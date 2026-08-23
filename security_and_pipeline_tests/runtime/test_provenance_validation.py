"""Recursive value-bearing provenance validator.

Every object holding extracted clinical content must carry a non-empty evidence
pointer (`*evidence_chunk_id` / `evidence_chunk_ids`) at its own level; arrays are
checked item-by-item. Structural fields and null placeholders are not value-bearing.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths,
)


def test_scalar_with_evidence_is_provenanced():
    # the real date_of_birth shape: value + sibling pointer + structural fields
    data = {
        "date_of_birth": "1957-04-15",
        "dob_certainty": 95,
        "dob_evidence": "explicit",
        "dob_evidence_chunk_id": "Test_Patient:demographics:0:TABLE",
        "age_years": 67,
        "rationale": "stated in demographics",
    }
    assert find_unprovenanced_value_paths(data) == []


def test_scalar_without_evidence_is_flagged():
    data = {"date_of_birth": "1957-04-15", "dob_evidence_chunk_id": None}
    assert find_unprovenanced_value_paths(data) == ["data"]


def test_array_items_need_item_level_evidence():
    # a lines-of-therapy shape: value array + per-item pointer; structural top level
    data = {
        "lines": [
            {"line_number": 1, "regimen": "AC-T", "start_date": "2020-01-XX",
             "evidence_chunk_ids": ["c1", "c2"]},
            {"line_number": 2, "regimen": "tamoxifen", "start_date": "2021-03-XX",
             "evidence_chunk_ids": ["c3"]},
        ],
        "n_lines": 2,
        "rationale": "two lines of therapy",
    }
    assert find_unprovenanced_value_paths(data) == []


def test_array_item_missing_evidence_is_flagged_by_index():
    data = {
        "lines": [
            {"regimen": "AC-T", "evidence_chunk_ids": ["c1"]},
            {"regimen": "tamoxifen", "evidence_chunk_ids": []},  # empty -> ungrounded
        ],
        "n_lines": 2,
    }
    assert find_unprovenanced_value_paths(data) == ["data.lines[1]"]


def test_structural_only_object_is_not_flagged():
    # counts/rationale/ok are structural -> no evidence required
    assert find_unprovenanced_value_paths({"ok": True, "n_lines": 0, "rationale": "none found"}) == []


def test_null_and_empty_values_are_not_value_bearing():
    assert find_unprovenanced_value_paths({"date_of_birth": None, "dob_evidence_chunk_id": None}) == []
    assert find_unprovenanced_value_paths({}) == []
    assert find_unprovenanced_value_paths(None) == []


def test_nested_object_provenance_is_independent():
    data = {
        "n_lines": 1,
        "primary": {"site": "left breast", "evidence_chunk_id": "c1"},   # provenanced
        "secondary": {"site": "lung"},                                    # NOT provenanced
    }
    assert find_unprovenanced_value_paths(data) == ["data.secondary"]


# --- `_status` is not a structural suffix ------------------------------------------

def test_derived_aggregate_flags_need_no_evidence():
    # derived_* fields restate what a grounded sibling collection already says, so they
    # are not independent claims (same category as the n_ count prefix). The real
    # a per-item enumeration shape -- derived flags at the top, per-item grounded primaries[]
    # -- is fully provenanced.
    assert find_unprovenanced_value_paths({"derived_multiple_primaries": True}) == []
    data = {
        "n_primaries": 2,
        "derived_multiple_primaries": True,
        "derived_non_breast_primaries_present": True,
        "primaries": [
            {"site": "lung", "invasive_or_in_situ": "invasive", "is_index_primary": True,
             "evidence_chunk_id": "Pt:pathology_report:0:0"},
            {"site": "breast", "invasive_or_in_situ": "invasive", "is_index_primary": False,
             "evidence_chunk_id": "Pt:pathology_report:1:0"},
        ],
        "rationale": "two primaries",
    }
    assert find_unprovenanced_value_paths(data) == []
    # a per-item primary that loses its pointer is still flagged (the contract still bites)
    data["primaries"][1].pop("evidence_chunk_id")
    assert find_unprovenanced_value_paths(data) == ["data.primaries[1]"]


def test_unknown_is_a_non_answer_placeholder():
    # "unknown" is the model's non-determination -- a placeholder like null, not an
    # extracted claim -- so an object whose only non-structural content is "unknown"
    # needs no evidence (there is nothing to ground a non-answer on). A real categorical
    # value next to it (a date) still pulls the object back under the contract.
    assert find_unprovenanced_value_paths({"vital_status": "unknown", "date_of_death": None}) == []
    assert find_unprovenanced_value_paths({"status_at_diagnosis": "Unknown"}) == []  # case-insensitive
    assert find_unprovenanced_value_paths(
        {"vital_status": "unknown", "date_of_death": "2025-03-XX"}
    ) == ["data"]  # the date is a real claim and is ungrounded


def test_vital_status_is_value_bearing_and_needs_evidence():
    # date_of_death's real shape: vital_status is a clinical claim, not a marker. An
    # "alive"/"dead" determination must be grounded.
    assert find_unprovenanced_value_paths(
        {"date_of_death": None, "vital_status": "alive", "confidence": 90}
    ) == ["data"]
    assert find_unprovenanced_value_paths(
        {"date_of_death": None, "vital_status": "alive", "evidence_chunk_id": "Pt:demographics:0:TABLE"}
    ) == []
    # a confirmed death with a date is grounded by the same object-level pointer
    assert find_unprovenanced_value_paths(
        {"date_of_death": "2025-03-XX", "vital_status": "dead",
         "evidence_chunk_id": "Pt:clinical_note.csv:3:1"}
    ) == []


def test_breast_receptor_status_fields_need_evidence():
    # er/pr/her2_status share the object-level evidence_chunk_id the helper guarantees.
    grounded = {
        "er_status": "positive", "pr_status": "positive", "her2_status": "negative",
        "grade": 2, "tumor_size_cm": 1.4, "margins": "clear",
        "evidence_chunk_id": "Pt:pathology_report.csv:1:0", "rationale": "IHC ER 95%",
    }
    assert find_unprovenanced_value_paths(grounded) == []
    # the same receptors without a pointer are flagged (regression guard)
    ungrounded = dict(grounded)
    ungrounded["evidence_chunk_id"] = None
    assert find_unprovenanced_value_paths(ungrounded) == ["data"]
    # an all-null (non-breast) result has no value-bearing field -> no provenance needed
    assert find_unprovenanced_value_paths(
        {"er_status": None, "pr_status": None, "her2_status": None,
         "grade": None, "tumor_size_cm": None, "margins": None,
         "evidence_chunk_id": None, "rationale": ""}
    ) == []
