"""Eval error analysis: decompose a value_mismatch into retrieval vs extraction
failure, and aggregate mismatches into systematic-vs-isolated patterns.

PHI note: both functions take/return raw chunk ids, fields, and clinical values — they
run on the PHI side (operator-facing eval output), never feed a NO_PHI artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jr_pipeline.evaluating_pipeline_performance.evaluation_metrics import bundle_sufficient
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


def gold_index(gold: list[dict]) -> dict[tuple[str, str], dict]:
    """Index retrieval-quality gold entries by (patient_id, variable) for O(1) lookup."""
    return {(g["patient_id"], g["variable"]): g for g in gold if g.get("patient_id") and g.get("variable")}


def chunks_fed_to_llm(run_id: str, patient_id: str, variable: str, dr: Path | None = None) -> set[str]:
    """The union of chunk ids fed to the LLM across this variable's retrieve_and_prompt
    steps, read from the step receipts' ``payload.evidence_packet.included_chunk_ids``."""
    steps_dir = phi_patient_run_dir(run_id, patient_id, dr) / "extract" / variable / "steps"
    fed: set[str] = set()
    if not steps_dir.is_dir():
        return fed
    for receipt in sorted(steps_dir.glob("*/receipt.json")):
        try:
            obj = json.loads(receipt.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        evidence_packet = (obj.get("payload") or {}).get("evidence_packet") or {}
        for chunk_id in evidence_packet.get("included_chunk_ids") or []:
            fed.add(chunk_id)
    return fed


def decompose_value_mismatch(
    *,
    run_id: str,
    patient_id: str,
    variable: str,
    gold_entry: dict | None,
    dr: Path | None = None,
) -> dict[str, Any]:
    """Classify why a value_mismatch happened by recomputing bundle sufficiency over the
    chunks ACTUALLY fed to the LLM vs the gold minimum-sufficient set:

      * ``extraction_miss`` — a sufficient set WAS fed; the model still got it wrong.
      * ``retrieval_miss``  — NO relevant chunk reached the model (it could not have answered).
      * ``both``            — some relevant chunks reached it but not a sufficient set
                              (retrieval was incomplete AND extraction failed on the rest).
      * ``no_retrieval_gold`` — no gold covers (patient, variable); not classifiable.
    """
    if gold_entry is None:
        return {"decomposition": "no_retrieval_gold"}
    fed = chunks_fed_to_llm(run_id, patient_id, variable, dr)
    min_sets = gold_entry.get("minimum_sufficient_set") or []
    relevant = set(gold_entry.get("relevant_chunk_ids") or [])
    bundle = bundle_sufficient(sorted(fed), min_sets, relevant_ids=relevant)
    relevant_fed = fed & relevant
    if bundle.sufficient:
        kind = "extraction_miss"
    elif not relevant_fed:
        kind = "retrieval_miss"
    else:
        kind = "both"
    return {
        "decomposition": kind,
        "bundle_sufficient": bundle.sufficient,
        "covered_set": bundle.covered_set,
        "n_chunks_fed": len(fed),
        "n_relevant_fed": len(relevant_fed),
        "n_relevant_gold": len(relevant),
    }


def _stable(value: Any) -> str:
    """Stable hashable key for grouping (lists/dicts can't be dict keys)."""
    return json.dumps(value, sort_keys=True, default=str)


def aggregate_mismatch_patterns(
    errors: list[dict], *, systematic_min_patients: int = 2
) -> list[dict]:
    """Group value_mismatch errors by (field, extracted, expected) across patients.

    A (field, extracted, expected) seen in >= ``systematic_min_patients`` DISTINCT patients
    is a SYSTEMATIC failure mode — the model gets the same thing wrong the same way, which
    is fixable at the recipe/prompt level — vs an ISOLATED one-off. With a single-patient
    review set every pattern is isolated by construction; the label degrades honestly
    rather than over-claiming a systematic trend it cannot support.
    """
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in errors:
        if e.get("error_type") != "value_mismatch":
            continue
        patient_id = e.get("patient_id")
        for field, mismatch in (e.get("mismatches") or {}).items():
            extracted, expected = mismatch.get("extracted"), mismatch.get("expected")
            key = (field, _stable(extracted), _stable(expected))
            bucket = buckets.setdefault(
                key,
                {"field": field, "extracted": extracted, "expected": expected, "patients": set()},
            )
            bucket["patients"].add(patient_id)
    patterns = [
        {
            "field": b["field"],
            "extracted": b["extracted"],
            "expected": b["expected"],
            "n_patients": len(b["patients"]),
            "pattern": "systematic" if len(b["patients"]) >= systematic_min_patients else "isolated",
        }
        for b in buckets.values()
    ]
    return sorted(patterns, key=lambda p: (-p["n_patients"], p["field"]))
