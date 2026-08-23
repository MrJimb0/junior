"""Hybrid member failures are recorded sanitized.

A failing member must never log the raw exception (which can carry PHI like an MRN/date):
per-member outcomes carry only error_kind + a message hash (never the raw text), and an
all-members-failing case raises rather than silently returning nothing.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.hybrid.hybrid_v1 import (
    HybridRetriever,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import RetrieverInfo

_PHI = "connection to MRN 12345678 on 2024-03-15 refused"


class _FailingMember:
    def __init__(self, kind, message):
        self.info = RetrieverInfo(kind=kind, version="v1", config={})
        self._message = message

    def query(self, corpus, *, text, k):
        raise RuntimeError(self._message)

    @property
    def score_normalization(self):
        return "raw"


class _OkMember:
    def __init__(self, kind):
        self.info = RetrieverInfo(kind=kind, version="v1", config={})

    def query(self, corpus, *, text, k):
        return []

    @property
    def score_normalization(self):
        return "raw"


def test_all_members_failing_raises_without_raw_phi():
    hybrid = HybridRetriever(members=[_FailingMember("bm25", _PHI), _FailingMember("embedding", _PHI)])
    with pytest.raises(RuntimeError) as excinfo:
        hybrid.query(corpus=None, text="q", k=5)
    message = str(excinfo.value)
    assert "12345678" not in message and "2024-03-15" not in message
    assert "RuntimeError" in message  # the error_kind, not the raw text


def test_member_outcomes_are_sanitized_and_one_failure_does_not_raise():
    hybrid = HybridRetriever(members=[_FailingMember("bm25", _PHI), _OkMember("embedding")])
    hybrid.query(corpus=None, text="q", k=5)  # one fails, one ok -> no raise
    bm25 = next(o for o in hybrid.last_member_outcomes if o["kind"] == "bm25")
    assert bm25["error_kind"] == "RuntimeError"
    assert bm25["sanitized_message_hash"]
    assert "12345678" not in str(hybrid.last_member_outcomes)


def test_member_outcomes_reach_the_retrieval_envelope_payload_sanitized():
    # the sanitized outcomes must actually reach the NO_PHI retrieval envelope, not just
    # sit on an in-memory attribute that the retrieval payload never reads.
    from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import _retrieval_payload

    hybrid = HybridRetriever(members=[_FailingMember("bm25", _PHI), _OkMember("embedding")])
    candidates = hybrid.query(corpus=None, text="q", k=5)
    payload = _retrieval_payload(
        hybrid, candidates,
        patient_id="p1", text="q", query_id="q0", variable="v", elapsed=0.0,
    )
    outcomes = payload["member_outcomes"]
    assert outcomes is not None and len(outcomes) == 2
    bm25 = next(o for o in outcomes if o["kind"] == "bm25")
    assert bm25["error_kind"] == "RuntimeError" and bm25["sanitized_message_hash"]
    # the raw PHI in the member exception never reaches the NO_PHI envelope payload
    assert "12345678" not in str(payload) and "2024-03-15" not in str(payload)


def test_non_hybrid_retriever_has_no_member_outcomes():
    from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import _retrieval_payload

    payload = _retrieval_payload(
        _OkMember("bm25"), [],
        patient_id="p1", text="q", query_id="q0", variable="v", elapsed=0.0,
    )
    assert payload["member_outcomes"] is None
