"""Rerank step — re-orders the chunks the search step found before extraction.

For one patient and one question, this takes the candidate chunks the search step
returned, filters them by the recipe's metadata conditions, ranks what survives
(by cross-encoder, by the rule-based combined score, or by document date), drops
chunks whose text merely repeats a higher-ranked one, keeps the best few, and
optionally re-sorts those into reading order before step 6 assembles them. The
numeric scores are written to the no-PHI side (numbers only); the actual chunk
text, which may contain protected health information, is kept on the
contains-PHI side.

Five modules:
  * rerank                     — the shell: build_reranker reads the recipe's
                                 reranking config and returns one configured ranker
                                 (the entry point used by step 7).
  * filter_candidates          — which candidates a recipe's ``filters:`` block lets
                                 through, including the partial-date rule its date
                                 comparisons use.
  * rank_candidates            — what happens to the survivors: rank them (by the
                                 rule-based combined score, the cross-encoder, or
                                 document date), collapse duplicate text, keep the
                                 top_n, optionally re-sort into reading order.
  * cross_encoder              — the toggleable model scorer (gte ModernBERT).
  * shared_reranking_contract  — the records every reranker passes around
                                 (EvidenceFilter, RerankInput, RerankedCandidate,
                                 RerankerInfo, RerankResult, the CandidateScorer
                                 Protocol) plus the ordering + text-resolution rules
                                 they all obey.
"""
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import (
    CandidateRanker,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import build_reranker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    CandidateScorer,
    EvidenceFilter,
    RerankedCandidate,
    RerankerInfo,
    RerankInput,
    RerankResult,
)

__all__ = [
    "build_reranker",
    "CandidateScorer",
    "RerankerInfo",
    "RerankInput",
    "RerankResult",
    "RerankedCandidate",
    "EvidenceFilter",
    "CandidateRanker",
]
