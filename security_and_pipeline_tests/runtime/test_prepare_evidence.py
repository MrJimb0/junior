"""Step 6 as the consolidator: include everything, measure it, flag if too big.

Step 6 no longer packs to a token budget — step 5 already selected the passages, so
step 6 includes them all, counts the tokens (estimated chars/4) total and per
document type, and raises loudly only if the assembled total exceeds the sanity
ceiling (rather than letting the model silently truncate an over-long prompt).
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_6_prepare_evidence_for_extraction.prepare_evidence import (
    EvidenceTooLargeError,
    prepare_evidence,
)


class _FakeCorpus:
    """chunk_id -> (text, row dict with doc_type/metadata)."""

    def __init__(self, items: dict[str, tuple[str, dict | None]]):
        self._items = items

    def text_for(self, chunk_id: str) -> str:
        return self._items[chunk_id][0]

    def metadata_for(self, chunk_id: str):
        return self._items.get(chunk_id, ("", None))[1]


def _cand(chunk_id, rank):
    return {"chunk_id": chunk_id, "rank": rank, "score": 1.0 / rank}


def test_includes_every_selected_passage_no_budget_drop():
    corpus = _FakeCorpus({
        "a": ("short note", {"doc_type": "progress_note"}),
        "b": ("word " * 500, {"doc_type": "progress_note"}),  # large; not dropped
        "c": ("another", {"doc_type": "pathology_report"}),
    })
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1), _cand("b", 2), _cand("c", 3)],
        variable="v/step",
        patient_id="P",
    )
    assert packet["block_count"] == 3
    assert set(packet["included_chunk_ids"]) == {"a", "b", "c"}


def test_token_total_and_per_doc_type_breakdown():
    corpus = _FakeCorpus({
        "a": ("alpha", {"doc_type": "pathology_report"}),
        "b": ("beta", {"doc_type": "imaging"}),
        "c": ("gamma", {"doc_type": "imaging"}),
    })
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1), _cand("b", 2), _cand("c", 3)],
        variable="v/step",
        patient_id="P",
    )
    by_type = packet["evidence_tokens_by_doc_type"]
    assert set(by_type) == {"pathology_report", "imaging"}
    # the breakdown sums to the reported total
    assert sum(by_type.values()) == packet["total_evidence_tokens"]
    assert packet["total_evidence_tokens"] > 0


def test_missing_doc_type_buckets_as_unknown():
    corpus = _FakeCorpus({"a": ("text", None)})  # :TABLE-style row, no metadata
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1)],
        variable="v/step",
        patient_id="P",
    )
    assert "unknown" in packet["evidence_tokens_by_doc_type"]


def test_over_ceiling_raises_loudly():
    corpus = _FakeCorpus({"a": ("word " * 1000, {"doc_type": "progress_note"})})
    with pytest.raises(EvidenceTooLargeError, match="over the"):
        prepare_evidence(
            corpus=corpus,
            reranked_candidates=[_cand("a", 1)],
            variable="v/step",
            patient_id="P",
            max_context_tokens=50,  # far below the ~1250-token block
        )


def test_under_ceiling_does_not_raise():
    corpus = _FakeCorpus({"a": ("small", {"doc_type": "progress_note"})})
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1)],
        variable="v/step",
        patient_id="P",
        max_context_tokens=10_000,
    )
    assert packet["total_evidence_tokens"] <= packet["max_context_tokens"]


def test_unresolved_text_chunk_is_dropped():
    corpus = _FakeCorpus({"a": ("", {"doc_type": "progress_note"}), "b": ("real", {"doc_type": "imaging"})})
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1), _cand("b", 2)],
        variable="v/step",
        patient_id="P",
    )
    assert packet["included_chunk_ids"] == ["b"]  # empty-text chunk dropped


def test_estimate_block_tokens_counts_rendered_header():
    # the estimate counts the whole rendered block (header + body), so a tiny body
    # still costs more than body/4 alone.
    corpus = _FakeCorpus({"a": ("hi", {"doc_type": "progress_note"})})
    packet = prepare_evidence(
        corpus=corpus, reranked_candidates=[_cand("a", 1)], variable="v/step", patient_id="P",
    )
    assert packet["total_evidence_tokens"] > len("hi") // 4


def test_all_unresolved_text_yields_empty_package():
    # Every selected passage resolves to empty/blank text -> the package is empty.
    # This is the input to step 7's empty_evidence guard (skip the model call), so
    # the empty packet shape must be exact: zero blocks, no chunk ids, zero tokens.
    corpus = _FakeCorpus({
        "a": ("", {"doc_type": "progress_note"}),
        "b": ("   ", {"doc_type": "imaging"}),
    })
    packet = prepare_evidence(
        corpus=corpus,
        reranked_candidates=[_cand("a", 1), _cand("b", 2)],
        variable="v/step",
        patient_id="P",
    )
    assert packet["block_count"] == 0
    assert packet["included_chunk_ids"] == []
    assert packet["total_evidence_tokens"] == 0
    assert packet["formatted_evidence_text"] == ""
