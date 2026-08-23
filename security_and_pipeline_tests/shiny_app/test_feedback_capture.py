"""The feedback writers' own contract, below the UI.

The sampling frame is the denominator of every accuracy number computed from a
review, so its file semantics are pinned directly: it accumulates, it deduplicates,
it keeps its opening date, it refuses a second sampling rule, and damage stops it
rather than silently starting it over. And every writer requires a run id — a label
that names no run can never be joined back to the extraction it judged.
"""
from __future__ import annotations

import json
from pathlib import Path

import feedback_capture
import pytest

RUN = "20260101_010101_aa"
RULE = "shiny_review_all_extracted_variables"


def _pair(patient_id: str, variable: str = "date_of_diagnosis") -> dict[str, str]:
    return {"patient_id": patient_id, "variable": variable}


def _frame(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── every writer requires the run it judges ──────────────────────────────────

def test_every_writer_refuses_a_missing_run_id(tmp_path):
    with pytest.raises(ValueError, match="run id"):
        feedback_capture.write_correction(
            patient_id="P", variable="v", correct_value="x", dr=tmp_path)
    with pytest.raises(ValueError, match="run id"):
        feedback_capture.write_confirmation(patient_id="P", variable="v", dr=tmp_path)
    with pytest.raises(ValueError, match="run id"):
        feedback_capture.write_chunk_relevance(
            patient_id="P", variable="v", chunk_id="c", relevant=True, dr=tmp_path)
    with pytest.raises(ValueError, match="run id"):
        feedback_capture.write_sampling_frame(drawn=[_pair("P")], rule=RULE, dr=tmp_path)
    assert not (tmp_path / "CONTAINS_PHI").exists(), "refused, but wrote anyway"


def test_a_correction_lands_in_the_shape_eval_values_reads(tmp_path):
    path = feedback_capture.write_correction(
        patient_id="Patient_A", variable="date_of_birth",
        correct_value="1999-12-31", original_value="1970-01-01",
        run_id=RUN, dr=tmp_path,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["patient_id"] == "Patient_A"
    assert record["run_id"] == RUN
    entry = record["feedback"][0]
    assert entry["type"] == "extraction_correction"
    assert entry["field"] == "date_of_birth"
    assert entry["correct_value"] == "1999-12-31"


def test_a_confirmation_lands_in_the_shape_eval_values_reads(tmp_path):
    path = feedback_capture.write_confirmation(
        patient_id="Patient_A", variable="date_of_birth",
        reviewed_value="1970-01-01", run_id=RUN, dr=tmp_path,
    )

    entry = json.loads(path.read_text(encoding="utf-8"))["feedback"][0]
    assert entry["type"] == "extraction_confirmation"
    assert entry["agreed"] is True
    assert entry["reviewed_value"] == "1970-01-01"


def test_two_entries_for_one_pair_append_rather_than_replace(tmp_path):
    feedback_capture.write_confirmation(
        patient_id="P", variable="v", reviewed_value="a", run_id=RUN, dr=tmp_path)
    path = feedback_capture.write_correction(
        patient_id="P", variable="v", correct_value="b", run_id=RUN, dr=tmp_path)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert [e["type"] for e in record["feedback"]] == [
        "extraction_confirmation", "extraction_correction",
    ]


# ── the sampling frame, one patient at a time ────────────────────────────────

def test_a_second_patients_draw_does_not_discard_the_first(tmp_path):
    """The reviewer records the sample per patient, so the frame has to accumulate.
    Rewriting it leaves a forty-patient review with a one-patient denominator."""
    feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule=RULE, run_id=RUN, dr=tmp_path)
    path = feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_B")], rule=RULE, run_id=RUN, dr=tmp_path)

    frame = _frame(path)
    assert [d["patient_id"] for d in frame["drawn"]] == ["Patient_A", "Patient_B"]
    assert frame["n_drawn"] == 2


def test_drawing_the_same_pair_twice_does_not_inflate_the_denominator(tmp_path):
    """Clicking Record twice on one patient must not make the sample look bigger than
    the number of patients actually drawn."""
    for _ in range(2):
        path = feedback_capture.write_sampling_frame(
            drawn=[_pair("Patient_A"), _pair("Patient_A", "stage")],
            rule=RULE, run_id=RUN, dr=tmp_path)

    frame = _frame(path)
    assert frame["n_drawn"] == 2
    assert len(frame["drawn"]) == 2


def test_the_frame_is_dated_when_the_review_opened_not_when_it_last_grew(tmp_path):
    """The draw is dated to establish that it preceded the reviewing. A frame restamped
    by its most recent append says the sample was drawn after some of the review."""
    path = feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule=RULE, run_id=RUN, dr=tmp_path)
    opened = _frame(path)
    opened["drawn_at"] = "2026-01-01T09:00:00"  # a review that began yesterday
    path.write_text(json.dumps(opened), encoding="utf-8")

    feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_B")], rule=RULE, run_id=RUN, dr=tmp_path)

    assert _frame(path)["drawn_at"] == "2026-01-01T09:00:00"


def test_a_draw_under_a_second_rule_is_refused_rather_than_relabelling_the_frame(tmp_path):
    """Merging two sampling rules into one frame would leave a denominator whose label
    describes only part of it — unusable, and unusable in a way nobody would notice."""
    feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule="random_10_per_variable", run_id=RUN, dr=tmp_path)

    with pytest.raises(RuntimeError, match="random_10_per_variable"):
        feedback_capture.write_sampling_frame(
            drawn=[_pair("Patient_B")], rule="all_low_confidence", run_id=RUN, dr=tmp_path)


def test_an_unreadable_frame_is_refused_rather_than_started_over(tmp_path):
    """Starting a fresh frame over a damaged one silently drops every patient already
    drawn, which is the whole defect. It has to stop and say so."""
    path = feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule=RULE, run_id=RUN, dr=tmp_path)
    path.write_text("{ truncated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot be read"):
        feedback_capture.write_sampling_frame(
            drawn=[_pair("Patient_B")], rule=RULE, run_id=RUN, dr=tmp_path)
    assert path.read_text(encoding="utf-8") == "{ truncated"  # left for recovery


def test_a_frame_of_the_wrong_shape_is_refused_like_any_other_damage(tmp_path):
    """Damage that still parses as JSON — a bare list where the record belongs — would
    reach ``previous.get`` and come back as a Python attribute error: a reviewer told a
    type name, not that the sample already drawn is at risk."""
    path = feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule=RULE, run_id=RUN, dr=tmp_path)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="cannot be read"):
        feedback_capture.write_sampling_frame(
            drawn=[_pair("Patient_B")], rule=RULE, run_id=RUN, dr=tmp_path)
    assert path.read_text(encoding="utf-8") == "[]"  # left for recovery


def test_a_frame_that_is_not_utf_8_text_is_refused_rather_than_started_over(tmp_path):
    """A frame that came back from a tool that mangled its encoding is not readable text
    at all. That is the same situation as a truncated one and has to stop the same way,
    rather than as a decoding error nobody can act on."""
    path = feedback_capture.write_sampling_frame(
        drawn=[_pair("Patient_A")], rule=RULE, run_id=RUN, dr=tmp_path)
    damaged = b'{"rule": "\xff\xfe"}'
    path.write_bytes(damaged)

    with pytest.raises(RuntimeError, match="cannot be read"):
        feedback_capture.write_sampling_frame(
            drawn=[_pair("Patient_B")], rule=RULE, run_id=RUN, dr=tmp_path)
    assert path.read_bytes() == damaged  # left for recovery
