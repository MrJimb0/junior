"""The exhaust record schemas are frozen, closed, and vocab-typed.

These pydantic models are the schema half of the write gate: a stray key,
an out-of-vocab categorical, or a full clinical date in ``emitted_month`` must fail
validation loudly, before any row reaches a shard.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from jr_pipeline.runtime_infrastructure.exhaust.schema import (
    RECORD_SCHEMAS,
    SCHEMA_VERSION,
    ExtractionOutcome,
    RecipeHeader,
    RunHeader,
    validate_record,
)
from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import VOCAB_VERSION


def _run_header_fields() -> dict:
    return dict(
        site_id="site_a",
        study_id="study_a",
        run_id="20260617_000000",
        emitted_month="2026-06",
        encoder_fingerprint="enc0",
        reranker_fingerprint="rer0",
        extractor_fingerprint="ext0",
        code_lock_hash="sha256:abc",
        chunking_config_id="chunk0",
    )


def _recipe_header_fields() -> dict:
    return {
        **_run_header_fields(),
        "recipe_id": "date_of_birth",
        "recipe_version": "v1",
        "variable_key": "date_of_birth",
        "archetype": "point_fact",
        "prompt_hash": "sha256:p",
        "schema_hash": "sha256:s",
    }


def _review_header_fields() -> dict:
    return {
        **_run_header_fields(),
        "recipe_id": "date_of_birth",
        "recipe_version": "v1",
        "step_id": "s0",
    }


def test_versions_default_onto_the_header():
    h = RunHeader(**_run_header_fields())
    assert h.schema_version == SCHEMA_VERSION
    assert h.vocab_version == VOCAB_VERSION


def test_headers_are_frozen_and_closed():
    h = RecipeHeader(**_recipe_header_fields())
    with pytest.raises(ValidationError):
        h.site_id = "other"  # frozen
    with pytest.raises(ValidationError):
        RecipeHeader(**_recipe_header_fields(), stray_key="x")  # extra='forbid'


def test_emitted_month_rejects_a_full_clinical_date():
    RunHeader(**{**_run_header_fields(), "emitted_month": "unknown"})  # allowed
    with pytest.raises(ValidationError):
        RunHeader(**{**_run_header_fields(), "emitted_month": "2026-06-15"})


def test_extraction_outcome_rejects_out_of_vocab_value_shape():
    base = {**_recipe_header_fields(), "patient_surrogate": "p0", "ok": True}
    ExtractionOutcome(**base, value_shape="scalar")  # valid member
    with pytest.raises(ValidationError):
        ExtractionOutcome(**base, value_shape="blob")  # not a ValueShape member
    with pytest.raises(ValidationError):
        ExtractionOutcome(**base, value_shape="scalar", error_kind="kaboom")  # bad ErrorKind


def test_extraction_outcome_closes_finish_reason_and_error_code():
    base = {**_recipe_header_fields(), "patient_surrogate": "p0", "ok": False, "value_shape": "null"}
    ExtractionOutcome(**base, finish_reason="length")            # valid FinishReason
    ExtractionOutcome(**base, error_code="sha256:" + "a" * 64)   # a hash is allowed
    with pytest.raises(ValidationError):
        ExtractionOutcome(**base, finish_reason="tumor_size_3cm")  # not a FinishReason
    with pytest.raises(ValidationError):
        ExtractionOutcome(**base, error_code="NameKittyBassett")   # raw text, not a hash


def test_clinical_invariant_severity_is_a_closed_set():
    from jr_pipeline.runtime_infrastructure.exhaust.schema import ClinicalInvariantOutcome
    ClinicalInvariantOutcome(rule_id="r", ok=False, severity="error")  # valid
    with pytest.raises(ValidationError):
        ClinicalInvariantOutcome(rule_id="r", ok=False, severity="patient_died_metastatic")


def test_validate_record_round_trips_each_registered_type():
    rows = [{
        "chunk_surrogate": "c0", "doc_type": "pathology", "rank": 1, "prior_rank": 1,
        "score": 0.9, "token_count": 10, "in_bundle": True, "cited": True,
    }]
    samples = {
        "selection_judgment": {
            **_recipe_header_fields(), "group": "g0", "step_id": "s0",
            "retriever": "hybrid", "total_evidence_tokens": 2000,
            "n_candidates": 1, "n_in_bundle": 1, "n_cited": 1, "n_doc_type_unmapped": 0,
            "rows": rows,
        },
        "clinical_invariant": {
            **_run_header_fields(), "labeler_id": "clinical_invariants/cancer_staging",
            "labeler_version": "v1", "patient_surrogate": "p0",
            "outcomes": [{"rule_id": "r1", "ok": True, "severity": "error"}],
            "n_violations": 0,
        },
        "method_provenance": {**_run_header_fields()},
        "extraction_outcome": {
            **_recipe_header_fields(), "patient_surrogate": "p0", "ok": True,
            "value_shape": "scalar",
        },
        "relevance_label": {
            **_review_header_fields(), "patient_surrogate": "p0", "reviewer_surrogate": "r0",
            "reviewer_tier": "expert", "chunk_surrogate": "c0", "label": "relevant",
        },
        "retriever_miss": {
            **_review_header_fields(), "patient_surrogate": "p0", "reviewer_surrogate": "r0",
            "reviewer_tier": "expert", "chunk_surrogate": "c0", "was_in_retriever_pool": True,
            "retriever_rank_if_present": 42, "cross_encoder_rank_if_present": 31,
            "manual_search_used": True,
        },
    }
    assert set(samples) == set(RECORD_SCHEMAS)
    for record_type, data in samples.items():
        out = validate_record(record_type, data)
        assert out["record_type"] == record_type
        assert out["schema_version"] == SCHEMA_VERSION


def test_validate_record_rejects_unknown_type_and_bad_retriever():
    with pytest.raises(KeyError):
        validate_record("not_a_record", {})
    bad = {
        **_recipe_header_fields(), "group": "g0", "step_id": "s0",
        "retriever": "quantum_search",  # not a Retriever member
        "n_candidates": 0, "n_in_bundle": 0, "n_cited": 0, "n_doc_type_unmapped": 0,
        "rows": [],
    }
    with pytest.raises(ValidationError):
        validate_record("selection_judgment", bad)
