"""Retrieval quality metrics: candidate ranking (recall@k, MRR, nDCG) and
evidence-bundle sufficiency against a gold ``minimum_sufficient_set``."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateMetrics:
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]


@dataclass(frozen=True)
class BundleMetrics:
    sufficient: bool
    covered_set: list[str] | None
    unnecessary_chunks: int


def candidate_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> CandidateMetrics:
    """Compute recall@k, MRR, nDCG@k (binary relevance) for one query."""
    if not relevant_ids:
        zeros = {k: 0.0 for k in k_values}
        return CandidateMetrics(recall_at_k=zeros, mrr=0.0, ndcg_at_k=dict(zeros))

    relevant = set(relevant_ids)
    recalls = {k: sum(1 for c in ranked_ids[:k] if c in relevant) / len(relevant) for k in k_values}

    mrr = 0.0
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            mrr = 1.0 / i
            break

    ndcgs: dict[int, float] = {}
    for k in k_values:
        dcg = sum(
            (1.0 / math.log2(i + 1)) if cid in relevant else 0.0
            for i, cid in enumerate(ranked_ids[:k], start=1)
        )
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        ndcgs[k] = (dcg / idcg) if idcg > 0 else 0.0

    return CandidateMetrics(recall_at_k=recalls, mrr=mrr, ndcg_at_k=ndcgs)


def bundle_sufficient(
    selected_ids: list[str],
    minimum_sufficient_sets: list[list[str]],
    relevant_ids: set[str] | None = None,
) -> BundleMetrics:
    """Return whether ``selected_ids`` is a superset of any minimum sufficient set."""
    selected = set(selected_ids)
    covered = next((list(alt) for alt in minimum_sufficient_sets if selected.issuperset(alt)), None)
    if relevant_ids is None:
        relevant_ids = {c for alt in minimum_sufficient_sets for c in alt}
    unnecessary = sum(1 for c in selected_ids if c not in relevant_ids)
    return BundleMetrics(sufficient=covered is not None, covered_set=covered, unnecessary_chunks=unnecessary)


def mean(values: list[float]) -> float:
    """Arithmetic mean; 0.0 on empty input."""
    return sum(values) / len(values) if values else 0.0
