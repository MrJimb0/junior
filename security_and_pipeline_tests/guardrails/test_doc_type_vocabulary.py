"""Raw EHR doc_type never reaches NO_PHI.

Raw doc_type strings are free text and can carry patient-identifying content.
``canonicalize_doc_type`` can only return a fixed ``VALID_DOC_TYPES`` member or
``"unknown"``; ``record_selection_trace`` must write only those, never the raw
value, and must count unmapped (non-empty unrecognised) values rather than silently
collapsing them.
"""
from __future__ import annotations

import json

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_selection_trace,
)
from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    VALID_DOC_TYPES,
    canonicalize_doc_type,
    is_unmapped,
)


def test_canonicalize_only_ever_returns_a_vocabulary_member():
    for raw in [
        "Pathology Final Report",
        "CT CHEST W/CONTRAST",
        "Progress Note - Dr. Jane Smith MRN 12345678",
        "some unrecognised vendor doc kind",
        "",
        None,
        12345,
        {"x": 1},
    ]:
        assert canonicalize_doc_type(raw) in VALID_DOC_TYPES


def test_phi_laden_doc_type_is_not_passed_through():
    out = canonicalize_doc_type("Progress Note for John Q Public, MRN 0001234")
    assert out == "progress_note"
    assert "John" not in out and "0001234" not in out


def test_is_unmapped_flags_nonempty_unrecognised_only():
    assert is_unmapped("totally unknown vendor type") is True
    assert is_unmapped("pathology") is False
    assert is_unmapped("") is False
    assert is_unmapped(None) is False


def test_record_selection_trace_emits_no_raw_doc_type(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    raw_phi = "Outpatient Progress Note - Jane Doe MRN 99887766"
    ranked = [
        {"chunk_id": "Test_Patient:notes:0:0", "doc_type": raw_phi, "rank": 1,
         "prior_rank": 1, "score": 0.9, "token_count": 10, "in_bundle": True},
        {"chunk_id": "Test_Patient:notes:1:0", "doc_type": "weird vendor kind", "rank": 2,
         "prior_rank": 2, "score": 0.5, "token_count": 8, "in_bundle": False},
    ]
    out_path = record_selection_trace(
        run_id="20260101_doctype", patient_id="Test_Patient",
        recipe_id="date_of_diagnosis", step_id="s0", archetype="event",
        retriever_kind="hybrid", total_evidence_tokens=2000, ranked=ranked,
    )
    text = out_path.read_text()
    payload = json.loads(text)
    assert {row["doc_type"] for row in payload["rows"]} <= VALID_DOC_TYPES
    assert raw_phi not in text
    assert "99887766" not in text
    assert payload["n_doc_type_unmapped"] == 1  # only "weird vendor kind"
