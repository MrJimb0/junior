"""Pressing a feedback button from a view that is not a real run must write no file.

Every one of the four buttons is irreversible and lands in the expert-label corpus
that accuracy is later measured against. With nothing selected there is no run and no
patient — a correction filed there is a gold record attached to nobody. From a run
that produced no readable output the patient is real and the value is not.

The guards that stop this live inside the button handlers, and a test that only asks
the helper for its sentence shows that a non-empty string exists — not that the click
it guards left the disk alone. So these press the buttons, and look at the disk.

The counterpart is here too: from a real run each button must still write, or the
guard has closed the one case feedback capture exists for.
"""
from __future__ import annotations

RUN = "20260101_010101_aa"
PATIENT = "Patient_A"
VARIABLE = "date_of_birth"
CHUNK_ID = f"{PATIENT}:notes:3"


def _corrections_tree(tmp_path):
    """Everything any feedback writer could have created, under any run id."""
    root = tmp_path / "CONTAINS_PHI" / "expert_label_corrections"
    return sorted(p.name for p in root.rglob("*.json")) if root.exists() else []


def _real_run_screen(review_screen, write_extract_output):
    write_extract_output(RUN, PATIENT, VARIABLE, evidence_chunk_id=CHUNK_ID)
    return review_screen(RUN, PATIENT)


def _press_every_feedback_button(screen) -> None:
    """Everything a reviewer can do that reaches disk: correct a value, confirm one,
    record the sampling frame, and judge an evidence chunk."""
    screen.set_input("fb_variable", VARIABLE)
    screen.set_input("fb_correct", "1999-12-31")
    screen.click_button("fb_submit")
    screen.click_button("fb_confirm")
    screen.click_button("fb_draw")
    screen.click_button("cr_yes_0")


# ── nothing is written from a view that is not a real run ───────────────────────

def test_no_button_files_anything_when_nothing_is_selected(tmp_path, review_screen):
    """The failure this exists to stop: a judgement recorded with no run behind it is
    a fabricated gold record nothing downstream can tell from a real one."""
    screen = review_screen()  # nothing selected — no run on disk

    _press_every_feedback_button(screen)

    assert _corrections_tree(tmp_path) == [], "wrote expert labels with no run behind them"
    assert "No run is selected" in screen.panel_text("fb_status")
    assert "Nothing was saved" in screen.panel_text("fb_status")
    assert "Nothing was saved" in screen.panel_text("cr_status")


def test_no_button_files_anything_for_a_run_that_produced_no_output(tmp_path, review_screen):
    """The patient here is real and the value is not: the run has no readable extract
    output, so there is nothing that was ever extracted to correct or agree with."""
    screen = review_screen(RUN, PATIENT)

    _press_every_feedback_button(screen)

    assert _corrections_tree(tmp_path) == [], "recorded a judgement of a value nobody produced"
    assert "Nothing was saved" in screen.panel_text("fb_status")


def test_no_button_files_anything_for_a_run_with_no_patient_at_all(tmp_path, review_screen):
    """A run whose patient folders are gone leaves the patient select empty. A write from
    here lands under an empty patient id, which is a record of nobody."""
    screen = review_screen(RUN, "")

    _press_every_feedback_button(screen)

    assert _corrections_tree(tmp_path) == []
    assert "Nothing was saved" in screen.panel_text("fb_status")


# ── the counterpart: a real run still writes all four ───────────────────────────

def test_a_correction_from_a_real_run_reaches_disk(tmp_path, review_screen, write_extract_output):
    screen = _real_run_screen(review_screen, write_extract_output)

    screen.set_input("fb_variable", VARIABLE)
    screen.set_input("fb_correct", "1999-12-31")
    screen.click_button("fb_submit")

    assert _corrections_tree(tmp_path) == [f"{PATIENT}__{VARIABLE}.json"]
    assert "Saved correction" in screen.panel_text("fb_status")


def test_a_confirmation_from_a_real_run_reaches_disk(tmp_path, review_screen, write_extract_output):
    screen = _real_run_screen(review_screen, write_extract_output)

    screen.set_input("fb_variable", VARIABLE)
    screen.click_button("fb_confirm")

    assert _corrections_tree(tmp_path) == [f"{PATIENT}__{VARIABLE}.json"]
    assert "Confirmed" in screen.panel_text("fb_status")


def test_a_sampling_frame_from_a_real_run_reaches_disk(tmp_path, review_screen, write_extract_output):
    screen = _real_run_screen(review_screen, write_extract_output)

    screen.click_button("fb_draw")

    assert _corrections_tree(tmp_path) == ["_sampling_frame.json"]


def test_a_chunk_relevance_judgement_from_a_real_run_reaches_disk(
    tmp_path, review_screen, write_extract_output,
):
    screen = _real_run_screen(review_screen, write_extract_output)

    screen.set_input("fb_variable", VARIABLE)
    screen.click_button("cr_yes_0")

    assert _corrections_tree(tmp_path) == [f"{PATIENT}__{VARIABLE}.json"]
    assert CHUNK_ID in screen.panel_text("cr_status")
