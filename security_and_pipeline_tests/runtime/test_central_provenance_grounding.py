"""Fabricated evidence citations are rejected pipeline-wide, not shipped as grounded.

The extract step nulls any evidence_chunk_id in the final data that cites a passage the
model was never shown and is not a well-formed structured evidence pointer. The nulled
value then flows through step 8's ordinary unprovenanced-value handling (it appears in
unprovenanced_value_paths and drives ok). A genuinely-shown chunk id and a structured
evidence pointer pass untouched.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import (
    _reject_ungrounded_evidence_pointers,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    build_structured_evidence_pointer,
)


def test_fabricated_pointer_rejected_shown_and_structured_pass():
    structured = build_structured_evidence_pointer(
        patient_id="p1", file_hash="sha256:abc", row_hash="sha256:def", column="race"
    )
    data = {
        "stage": {"value": "IIA", "stage_evidence_chunk_id": "p1:notes:9:9"},  # fabricated
        "grade": {"value": "2", "grade_evidence_chunk_id": "p1:notes:0:0"},    # genuinely shown
        "race": {"value": "White", "race_evidence_chunk_id": structured},      # structured pointer
    }
    shown = {"p1:notes:0:0"}  # only grade's chunk was ever put in front of the model
    rejected = _reject_ungrounded_evidence_pointers(data, shown, step_id="s1")

    # the fabricated citation is nulled and recorded; the shown + structured pointers stay
    assert data["stage"]["stage_evidence_chunk_id"] is None
    assert data["grade"]["grade_evidence_chunk_id"] == "p1:notes:0:0"
    assert data["race"]["race_evidence_chunk_id"] == structured
    assert rejected == [
        {"path": "data.stage.stage_evidence_chunk_id", "chunk_id": "p1:notes:9:9", "step": "s1"}
    ]

    # the now-pointer-less value is unprovenanced; the grounded ones are not
    unprovenanced = find_unprovenanced_value_paths(data)
    assert "data.stage" in unprovenanced
    assert "data.grade" not in unprovenanced
    assert "data.race" not in unprovenanced


def test_list_pointer_keeps_grounded_items_and_drops_fabricated():
    data = {
        "primaries": [
            {"site": "lung", "evidence_chunk_ids": ["p1:notes:0:0", "p1:notes:9:9"]},
        ]
    }
    shown = {"p1:notes:0:0"}
    rejected = _reject_ungrounded_evidence_pointers(data, shown, step_id="s1")

    assert data["primaries"][0]["evidence_chunk_ids"] == ["p1:notes:0:0"]  # fabricated item dropped
    assert len(rejected) == 1 and rejected[0]["chunk_id"] == "p1:notes:9:9"
    # a grounded item remains, so the object is still provenanced
    assert find_unprovenanced_value_paths(data) == []
