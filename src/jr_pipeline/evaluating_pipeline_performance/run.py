"""Retrieval-quality eval harness.

Reads curated gold queries, runs each through the configured retriever, and
writes two artifacts under ``out/<run_id>/data/eval/``:

  * candidate_metrics — recall@k, MRR, nDCG over the per-query ranked list.
  * bundle_metrics    — does the top-k bundle contain at least one of the
                        query's ``min_sets`` answer chunks?

Retrieval-only; does not invoke the extract step.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from jr_pipeline.evaluating_pipeline_performance.evaluation_metrics import (
    BundleMetrics,
    CandidateMetrics,
    bundle_sufficient,
    candidate_metrics,
    mean,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import retrieve_one
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    eval_metrics_dir,
    phi_intermediate_run_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_log = get_logger("eval")

def load_gold(path: Path) -> list[dict]:
    """Load a JSONL gold file; skips blank lines and ``#`` comments."""
    gold: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                gold.append(json.loads(s))
    return gold

def run_eval(
    *,
    cfg: dict,
    gold_path: Path,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    code_lock_hash: str | None = None,
) -> dict:
    """Execute every gold query through the retriever and write summaries."""
    gold = load_gold(gold_path)
    _log.info("eval_start", extra_={"queries": len(gold)})

    per_query: list[dict[str, Any]] = []
    # Means are over SCORED queries only — ok AND retrieval-annotated. Failed queries
    # (retriever raised) and unannotated queries (no gold relevant set) are counted
    # separately, never silently averaged in as zeros (anti-silent-averaging).
    scored_recalls: dict[int, list[float]] = {k: [] for k in k_values}
    scored_ndcgs: dict[int, list[float]] = {k: [] for k in k_values}
    scored_mrr: list[float] = []
    n_ok = n_failed = n_unannotated = 0
    sufficient_count = 0
    n_bundle_scored = 0  # queries with a gold minimum_sufficient_set AND a successful retrieve

    top_k = int(cfg.get("k", max(k_values)))
    t0 = time.perf_counter()

    for entry in gold:
        query_id = entry["query_id"]
        patient_id = entry["patient_id"]
        text = entry["query_text"]
        relevant = set(entry.get("relevant_chunk_ids") or [])
        min_sets = entry.get("minimum_sufficient_set") or []

        q_cfg = dict(cfg)
        q_cfg["k"] = top_k
        q_cfg["write"] = False
        try:
            artifact = retrieve_one(
                cfg=q_cfg,
                patient_id=patient_id,
                text=text,
                query_id=query_id,
                variable=entry.get("variable"),
                code_lock_hash=code_lock_hash,
            )
            ranked = [c["chunk_id"] for c in artifact["payload"]["candidates"]]
            selected = list(artifact["payload"]["selected_chunk_ids"])
            ok = True
            error_kind = None
        except Exception as e:  # one broken query must not tank the eval
            ranked, selected = [], []
            ok = False
            # a stable error_kind (the exception class), never the raw message —
            # an exception string can carry a path/id/PHI; the class name cannot.
            error_kind = type(e).__name__

        cm: CandidateMetrics = candidate_metrics(ranked, relevant, k_values=k_values)
        bm: BundleMetrics = bundle_sufficient(selected, min_sets, relevant_ids=relevant)

        annotated = bool(relevant)   # has a gold relevance set -> recall/MRR/nDCG are defined
        scored = ok and annotated
        n_ok += int(ok)
        n_failed += int(not ok)
        n_unannotated += int(not annotated)
        if scored:
            for k, v in cm.recall_at_k.items():
                scored_recalls[k].append(v)
            for k, v in cm.ndcg_at_k.items():
                scored_ndcgs[k].append(v)
            scored_mrr.append(cm.mrr)
        if ok and min_sets:
            n_bundle_scored += 1
            if bm.sufficient:
                sufficient_count += 1

        per_query.append({
            "query_id": query_id,
            "patient_id": patient_id,
            "variable": entry.get("variable"),
            "ok": ok,
            "error_kind": error_kind,
            "scored": scored,
            "annotated": annotated,
            "n_relevant": len(relevant),  # the recall@k denominator for this query
            "recall_at_k": cm.recall_at_k,
            "mrr": cm.mrr,
            "ndcg_at_k": cm.ndcg_at_k,
            "bundle_sufficient": bm.sufficient,
            "bundle_covered_set": bm.covered_set,
            "bundle_unnecessary_chunks": bm.unnecessary_chunks,
        })

    elapsed = time.perf_counter() - t0
    n = len(gold)
    n_scored = len(scored_mrr)

    candidate_summary = {
        "n_queries": n,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_unannotated": n_unannotated,
        "n_scored": n_scored,  # denominator behind the means (excludes failed + unannotated)
        "mean_recall_at_k": {k: round(mean(scored_recalls[k]), 4) for k in k_values},
        "mean_ndcg_at_k": {k: round(mean(scored_ndcgs[k]), 4) for k in k_values},
        "mean_mrr": round(mean(scored_mrr), 4),
        "per_query": per_query,
        "elapsed_s": round(elapsed, 4),
    }
    bundle_summary = {
        "n_queries": n,
        "n_bundle_scored": n_bundle_scored,  # denominator behind the rate (ok + has min-set)
        "bundle_sufficient_rate": (
            round(sufficient_count / n_bundle_scored, 4) if n_bundle_scored else 0.0
        ),
        "per_query": [
            {
                "query_id": r["query_id"],
                "sufficient": r["bundle_sufficient"],
                "covered_set": r["bundle_covered_set"],
                "unnecessary_chunks": r["bundle_unnecessary_chunks"],
            }
            for r in per_query
        ],
    }

    run_root = phi_intermediate_run_dir(cfg["run_id"])
    eval_dir = eval_metrics_dir(run_root)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in (
        ("candidate_metrics", candidate_summary),
        ("bundle_metrics", bundle_summary),
    ):
        from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
            envelope_for,
        )

        # per-query rows carry the raw patient_id, so these metrics are PHI-side
        # (sensitivity:"high", matching the run_roster convention for raw ids) and stay in
        # the PHI tree — never the export/egress path. Aggregate means are PHI-clean but
        # ship in the same artifact. Validated against the base envelope until a dedicated
        # eval-metrics schema exists.
        env = envelope_for(
            artifact_type="run_metrics_shard",
            sensitivity="high",
            stream="data",
            run_id=cfg["run_id"],
            step="validate",
            payload=payload,
            code_lock_hash=code_lock_hash,
        )
        # registry -> run_root/data/eval/<name>.json; validate against the base envelope.
        write_artifact(env, run_root=run_root, name=name, schema_name="artifact_envelope")

    _log.info("eval_done", extra_={"n": n, "sufficient_rate": bundle_summary["bundle_sufficient_rate"]})
    return {"candidate": candidate_summary, "bundle": bundle_summary}
