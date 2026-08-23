"""direct_parquet `contains` is a literal substring match, not a regex.

A recipe value like "5-FU (bolus)" has regex metacharacters (parens); treated as a
pattern it would silently match zero rows. ``literal=True`` makes it a substring match.
"""
from __future__ import annotations

import polars as pl

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.nonembedded_structured_parquet_table_looker_uper.direct_parquet_retriever_v1 import (  # noqa: E501
    _apply_filter,
)


def test_contains_matches_literal_parentheses():
    df = pl.DataFrame({"med": ["5-FU (bolus)", "carboplatin", "5-FU (infusion)"]})
    out = _apply_filter(df, "med contains '5-FU (bolus)'")
    assert out["med"].to_list() == ["5-FU (bolus)"]


def test_contains_does_not_treat_value_as_regex():
    df = pl.DataFrame({"code": ["a.b", "axb", "ayb"]})
    out = _apply_filter(df, "code contains 'a.b'")
    assert out["code"].to_list() == ["a.b"]  # literal dot, not a regex wildcard


def test_eq_and_ne_still_work():
    df = pl.DataFrame({"x": ["p", "q", "p"]})
    assert _apply_filter(df, "x == 'p'")["x"].to_list() == ["p", "p"]
    assert _apply_filter(df, "x != 'p'")["x"].to_list() == ["q"]
