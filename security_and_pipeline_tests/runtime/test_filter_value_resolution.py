"""Reranking filter-value resolution.

A filter value can be a literal (used verbatim) or a ``{{ vars.* }}`` template that is
filled in from the patient's upstream results. Only an unresolved template — one whose
referenced variable had no value for this patient — is skippable; a literal is always
applied, even the literal 'unknown'.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.retrieve_and_prompt_step import (
    _resolve_filter_value,
)


def test_literal_unknown_is_used_verbatim_not_treated_as_unresolved():
    # A recipe may legitimately filter on the literal 'unknown' — a None doc_type is
    # canonicalized to 'unknown' — so a literal is never read as an unresolved variable.
    resolved, ok = _resolve_filter_value("unknown", {})
    assert (resolved, ok) == ("unknown", True)


def test_template_resolving_to_unknown_is_skipped():
    resolved, ok = _resolve_filter_value("{{ status }}", {"status": "unknown"})
    assert ok is False
    assert resolved == "unknown"


def test_template_resolving_to_a_real_value_is_applied():
    resolved, ok = _resolve_filter_value(
        "{{ vars.diagnosis_date }}", {"vars": {"diagnosis_date": "2020-01-01"}}
    )
    assert (resolved, ok) == ("2020-01-01", True)
