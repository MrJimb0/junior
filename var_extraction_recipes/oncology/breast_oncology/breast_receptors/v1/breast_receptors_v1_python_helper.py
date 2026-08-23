"""Final cleanup step for ``breast_receptors`` — tidy up the receptor / grade /
margin values and record where each came from in the chart. A value of null means
"not stated / not applicable" (for example, a chart with no breast disease at all),
so a chart with no breast findings comes out all-null and needs no source citation.
This step does no language-model call; it only normalizes what an earlier step found.

"Grounding" below means recording which chart passage a value came from (its
evidence_chunk_id), so every reported value is traceable back to the chart.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)

_RECEPTORS = frozenset({"positive", "negative"})
_MARGINS = frozenset({"clear", "close", "positive"})


def _receptor(value: Any) -> str | None:
    return value if value in _RECEPTORS else None


def finalize(ctx: StepContext) -> StepResult:
    steps = ctx.step_outputs or {}
    r = ((steps.get("receptors") or {}).get("data")) or {}
    repair = ((steps.get("clinical_receptor_repair") or {}).get("data")) or {}

    grade = r.get("grade")
    grade = grade if isinstance(grade, int) and not isinstance(grade, bool) and 1 <= grade <= 3 else None

    size = r.get("tumor_size_cm")
    size = size if isinstance(size, (int, float)) and not isinstance(size, bool) else None

    pathology_evidence = r.get("evidence_chunk_id")
    repair_evidence = repair.get("evidence_chunk_id")
    er = (
        _receptor(r.get("er_status")) if pathology_evidence else None
    ) or (_receptor(repair.get("er_status")) if repair_evidence else None)
    pr = (
        _receptor(r.get("pr_status")) if pathology_evidence else None
    ) or (_receptor(repair.get("pr_status")) if repair_evidence else None)
    her2 = (
        _receptor(r.get("her2_status")) if pathology_evidence else None
    ) or (_receptor(repair.get("her2_status")) if repair_evidence else None)
    margins = r.get("margins") if r.get("margins") in _MARGINS else None
    pathology_details = {
        "er_percent": r.get("er_percent"),
        "pr_percent": r.get("pr_percent"),
        "her2_ihc": r.get("her2_ihc"),
        "her2_ish": r.get("her2_ish"),
        "histologic_type": r.get("histologic_type"),
        "invasive_foci_count": r.get("invasive_foci_count"),
        "positive_nodes": r.get("positive_nodes"),
        "nodes_examined": r.get("nodes_examined"),
        "neoadjuvant_therapy_before_specimen": r.get(
            "neoadjuvant_therapy_before_specimen"
        ),
    }
    if not pathology_evidence:
        grade = size = margins = None
        pathology_details = {key: None for key in pathology_details}

    # Every receptor value we report must be traceable to a chart passage. If the
    # model found features but cited no passage, those values can't be backed up by
    # the chart -> drop them all to null (the stage helper does the same). This also
    # closes a loophole: the names er_status / pr_status / her2_status end in
    # "_status", which the generic provenance validator skips, so without this guard
    # they could slip through untraced; here we guarantee a receptor is never reported
    # without a source passage. An all-null result (e.g. a non-breast chart) has no
    # value to back up and so needs no citation.
    evidence = pathology_evidence or repair_evidence
    any_value = any(
        v is not None
        for v in (er, pr, her2, grade, size, margins, *pathology_details.values())
    )
    if any_value and not evidence:
        er = pr = her2 = grade = size = margins = None
        pathology_details = {key: None for key in pathology_details}
        any_value = False
    if not any_value:
        evidence = None  # nothing found -> no dangling evidence pointer
        pathology_evidence = repair_evidence = None

    out = {
        "er_status": er,
        "pr_status": pr,
        "her2_status": her2,
        "grade": grade,
        "tumor_size_cm": size,
        "margins": margins,
        **pathology_details,
        "evidence_chunk_id": evidence,
        "evidence_chunk_ids": [
            chunk_id
            for chunk_id in (pathology_evidence, repair_evidence)
            if isinstance(chunk_id, str) and chunk_id
        ],
        "rationale": (r.get("rationale") or repair.get("rationale") or "")[:400],
    }
    grounded = evidence is not None
    return StepResult(data=out, receipt_payload={"python_finalize": {"grounded": grounded}})
