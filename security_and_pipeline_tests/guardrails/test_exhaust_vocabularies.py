"""The exhaust controlled vocabularies are closed StrEnums.

These enums are the wire values for categorical exhaust fields. Each must be a
``str`` subclass (so it serialises as its value) and ``VALID_DOC_TYPES`` must stay
derived from ``DocType`` so the two can never drift.
"""
from __future__ import annotations

from enum import StrEnum

from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    VALID_DOC_TYPES,
    VOCAB_VERSION,
    CorrectionType,
    DocType,
    HumanTier,
    RelevanceLabel,
    Retriever,
    SignalType,
    ValueShape,
)


def test_vocab_version_is_set():
    assert VOCAB_VERSION == "exhaust/v1"


def test_all_vocabularies_are_str_enums():
    for vocab in (DocType, Retriever, CorrectionType, SignalType, ValueShape, HumanTier,
                  RelevanceLabel):
        assert issubclass(vocab, StrEnum)
        for member in vocab:
            assert isinstance(member, str)  # serialises as its string value


def test_relevance_label_covers_relevant_not_relevant_unsure():
    assert {m.value for m in RelevanceLabel} == {"relevant", "not_relevant", "unsure"}


def test_valid_doc_types_is_derived_from_the_enum():
    assert VALID_DOC_TYPES == frozenset(m.value for m in DocType)
    assert "unknown" in VALID_DOC_TYPES


def test_retriever_vocab_covers_the_real_retriever_kinds():
    # Every RetrieverInfo.kind the pipeline actually emits must be a member, or a
    # real run's exhaust would fail validation.
    for kind in ("bm25", "embedding", "hybrid", "exact", "direct_parquet"):
        assert Retriever(kind).value == kind


def test_reviewer_tiers_match_the_feedback_layer():
    assert {m.value for m in HumanTier} == {"expert", "trainee", "unknown"}
