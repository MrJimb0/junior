"""Pin the default reranker build + the rule-based combined-score math + tie-break.

With the cross-encoder toggled off (the default), the candidate ranker scores
candidates with the rule-based combined score. These tests pin that math so the
many recipes that use the default keep producing the exact same order.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import (
    CandidateRanker,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import _DEFAULT_K0, build_reranker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import RerankInput
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate

_LEXICAL_ONLY = {"retriever_score": 0.0, "lexical_overlap": 1.0, "source_priority": 0.0}


class _FakeCorpus:
    """Minimal PatientChunkStore stand-in: chunk_id -> text (+ optional source_file)."""

    def __init__(self, texts: dict[str, str], sources: dict[str, str] | None = None):
        self._texts = texts
        self._sources = sources or {}

    def text_for(self, chunk_id: str) -> str:
        return self._texts[chunk_id]

    def metadata_for(self, chunk_id: str):
        source = self._sources.get(chunk_id)
        return {"source_file": source} if source is not None else None


def test_build_reranker_default_is_composed_with_cross_encoder_off():
    reranker = build_reranker({})
    assert isinstance(reranker, CandidateRanker)
    assert reranker.info.kind == "composed"
    assert reranker.info.config["cross_encoder"] is False


def test_combined_score_is_weighted_sum():
    # query {alpha, beta}; chunk {alpha, gamma} -> lexical_overlap = 1/2.
    reranker = build_reranker({})  # default weights 0.4 / 0.4 / 0.2, k0=60
    corpus = _FakeCorpus({"c1": "alpha gamma"})
    candidate = Candidate(chunk_id="c1", rank=1, score=0.0, retriever="bm25")

    out = reranker.rerank(corpus, RerankInput(candidates=[candidate], query_text="alpha beta"), top_n=5).candidates

    assert len(out) == 1
    features = out[0].features
    assert features["retriever_score"] == pytest.approx(1.0 / (_DEFAULT_K0 + 1))
    assert features["lexical_overlap"] == pytest.approx(0.5)
    assert features["source_priority"] == 0.0
    assert out[0].score == pytest.approx(
        0.4 * features["retriever_score"] + 0.4 * 0.5 + 0.2 * 0.0
    )
    assert out[0].rank == 1 and out[0].prior_rank == 1


def test_tie_break_chunk_id_when_score_and_prior_rank_equal():
    # lexical-only weights -> both chunks contain the whole query, so both score 1.0;
    # identical prior_rank forces the chunk_id tiebreak (ascending: "a" before "b").
    # The two texts must DIFFER or dedup would (rightly) drop one before scoring.
    reranker = build_reranker({"weights": _LEXICAL_ONLY})
    corpus = _FakeCorpus({"b": "x beta", "a": "x alpha"})
    cands = [
        Candidate(chunk_id="b", rank=5, score=0.0, retriever=None),
        Candidate(chunk_id="a", rank=5, score=0.0, retriever=None),
    ]
    out = reranker.rerank(corpus, RerankInput(candidates=cands, query_text="x"), top_n=5).candidates
    assert [c.chunk_id for c in out] == ["a", "b"]


def test_tie_break_prior_rank_before_chunk_id():
    # equal score (lexical-only; both texts contain the whole query, and they differ
    # from each other so dedup leaves both in). The lexically-LATER chunk_id ("z")
    # deliberately carries the BETTER prior_rank (1) so the two candidate sort keys
    # DIVERGE — this is what makes the test actually pin the precedence:
    #   correct  (-score, prior_rank, chunk_id) -> ["z", "a"]   (prior_rank wins)
    #   swapped  (-score, chunk_id, prior_rank) -> ["a", "z"]   (chunk_id wins)
    reranker = build_reranker({"weights": _LEXICAL_ONLY})
    corpus = _FakeCorpus({"a": "x alpha", "z": "x zeta"})
    cands = [
        Candidate(chunk_id="a", rank=9, score=0.0, retriever=None),
        Candidate(chunk_id="z", rank=1, score=0.0, retriever=None),
    ]
    out = reranker.rerank(corpus, RerankInput(candidates=cands, query_text="x"), top_n=5).candidates
    assert [c.chunk_id for c in out] == ["z", "a"]


def test_empty_candidates_returns_empty():
    reranker = build_reranker({})
    out = reranker.rerank(_FakeCorpus({}), RerankInput(candidates=[], query_text="q"), top_n=5).candidates
    assert out == []
