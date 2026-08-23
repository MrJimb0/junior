"""Retriever A/B comparison — no LLM required.

Given a patient, a query text, and a list of retriever configs, run each,
report candidate-metric deltas side by side, and write a single report
artifact. Useful for chunk-size and retriever tuning (Gate 2 acceptance).
"""
from __future__ import annotations

import time
from typing import Any

from jr_pipeline.evaluating_pipeline_performance.evaluation_metrics import candidate_metrics
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import retrieve_one


def compare_retrievers(
    *,
    cfg: dict,
    patient_id: str,
    query_text: str,
    retriever_cfgs: list[dict],
    relevant_chunk_ids: list[str] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """Run each retriever against the same corpus and query; report deltas."""
    relevant = set(relevant_chunk_ids or [])
    results: list[dict[str, Any]] = []
    for rcfg in retriever_cfgs:
        q_cfg = dict(cfg)
        q_cfg["retriever"] = rcfg
        q_cfg["k"] = max(k_values)
        q_cfg["write"] = False
        t0 = time.perf_counter()
        art = retrieve_one(
            cfg=q_cfg,
            patient_id=patient_id,
            text=query_text,
            query_id=f"compare_{rcfg.get('kind', 'x')}",
        )
        elapsed = time.perf_counter() - t0
        ranked = [c["chunk_id"] for c in art["payload"]["candidates"]]
        metrics = candidate_metrics(ranked, relevant, k_values=k_values) if relevant else None
        results.append({
            "retriever_kind": rcfg.get("kind"),
            "retriever_config": rcfg,
            "candidates": art["payload"]["candidates"],
            "metrics": {
                "recall_at_k": metrics.recall_at_k if metrics else None,
                "mrr": metrics.mrr if metrics else None,
                "ndcg_at_k": metrics.ndcg_at_k if metrics else None,
            },
            "elapsed_s": round(elapsed, 6),
        })
    return {
        "patient_id": patient_id,
        "query_text": query_text,
        "relevant_chunk_ids": sorted(relevant),
        "results": results,
    }
