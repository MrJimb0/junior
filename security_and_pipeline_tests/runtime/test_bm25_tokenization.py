"""BM25 tokenizes on [a-z0-9]+ runs, not whitespace.

Splitting on any non-alphanumeric character keeps trailing punctuation from gluing to
words, so a query term "cancer" still matches a chunk saying "cancer," — protecting
recall.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.bm25.bm25_v2 import (
    _VERSION,
    _tok,
)


def test_tok_splits_punctuation_and_lowercases():
    assert _tok("Cancer, invasive (ductal)!") == ["cancer", "invasive", "ductal"]


def test_tok_query_term_matches_word_with_trailing_punctuation():
    # "cancer" appears as a token of "the cancer, staged"
    assert "cancer" in _tok("the cancer, staged")


def test_tok_splits_hyphenated_and_possessive():
    assert _tok("5-FU patient's") == ["5", "fu", "patient", "s"]


def test_tok_empty_and_none_safe():
    assert _tok("") == []
    assert _tok(None) == []


def test_version_bumped_for_tokenization_change():
    assert _VERSION == "v2"
