"""Sentences the review screen prints that a reviewer acts on.

Each of these is a claim about what the app did, and each is pinned because it can
be wrong in a direction nobody checks:

  * the sampling frame accumulates across patients, so the status line must report
    the run-wide total the file holds — not the click's own count;
  * a frame that cannot be read is refused, and the refusal reaches the reviewer as
    a sentence with a next step, not a Python type name;
  * the empty-result panel knows the difference between "no run selected", "this
    run has no patient", and "this patient's output cannot be read" — three states
    with three different next steps.

These drive the app's own handlers and render functions (see the ``review_screen``
fixture), because every one of these sentences is composed inside ``server()``.
"""
from __future__ import annotations

import json

import feedback_capture

RUN = "20260101_010101_aa"


def _frame_on_disk(tmp_path) -> dict:
    path = feedback_capture.feedback_dir(RUN, tmp_path) / "_sampling_frame.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ── the sampling-frame status line ──────────────────────────────────────────────

def test_recording_a_second_patient_reports_the_run_wide_total_not_this_click(
    tmp_path, review_screen, write_extract_output,
):
    """The frame is the denominator of every accuracy number computed from this review.
    Reporting the click's own count tells a reviewer the sample is two pairs while the
    file they are building holds twice that."""
    for patient in ("Patient_A", "Patient_B"):
        write_extract_output(RUN, patient, "date_of_birth")
        write_extract_output(RUN, patient, "stage", value="IIA")
    screen = review_screen(RUN, "Patient_A")

    screen.click_button("fb_draw")
    screen.set_input("review_patient", "Patient_B")
    screen.click_button("fb_draw")
    status = screen.panel_text("fb_status")

    assert _frame_on_disk(tmp_path)["n_drawn"] == 4
    assert "4" in status, f"the screen and the file disagree about the sample: {status!r}"
    assert "denominator" in status, "does not say what the number is for"


def test_the_sample_the_screen_reports_is_the_one_that_reached_the_file(
    tmp_path, review_screen, write_extract_output,
):
    """Counting what was passed in rather than reading what was written cannot notice a
    pair the merge dropped — so the number reported has to come off the file."""
    write_extract_output(RUN, "Patient_A", "date_of_birth")
    write_extract_output(RUN, "Patient_A", "stage", value="IIA")
    screen = review_screen(RUN, "Patient_A")

    screen.click_button("fb_draw")
    screen.click_button("fb_draw")  # the same patient twice must not inflate the total
    status = screen.panel_text("fb_status")

    assert _frame_on_disk(tmp_path)["n_drawn"] == 2
    assert "holds 2 pairs" in status, f"reported a total the file does not hold: {status!r}"


def test_a_frame_that_cannot_be_read_is_reported_as_a_sentence_not_a_python_error(
    tmp_path, review_screen, write_extract_output,
):
    """The frame is refused rather than overwritten when it cannot be read, and the
    refusal reaches the reviewer through the status line. Left unhandled it surfaces as
    the exception's own text, which names a Python type and no next step."""
    write_extract_output(RUN, "Patient_A", "date_of_birth")
    screen = review_screen(RUN, "Patient_A")
    screen.click_button("fb_draw")
    frame_path = feedback_capture.feedback_dir(RUN, tmp_path) / "_sampling_frame.json"
    frame_path.write_bytes(b"[]")

    screen.click_button("fb_draw")
    status = screen.panel_text("fb_status")

    assert "Nothing was saved" in status
    assert "Move that file aside" in status, "does not say what to do next"
    for python_leak in ("AttributeError", "UnicodeDecodeError", "RuntimeError", "object"):
        assert python_leak not in status, f"shows the operator {python_leak}"
    assert frame_path.read_bytes() == b"[]", "overwrote the frame it could not read"


# ── the empty-result panel knows all three states ───────────────────────────────

def test_a_run_with_no_patient_is_not_told_to_re_run_extract_for_that_patient(review_screen):
    """A run whose patient folders are gone leaves the patient select empty. Telling the
    reviewer to re-run extract "for this patient" names somebody they never picked, and
    sends them to re-run a stage that is not what failed."""
    screen = review_screen(RUN, "")

    panel = screen.panel_text("extract_meta_panel")

    assert "for this patient" not in panel, panel
    assert "has no patient with readable extract output" in panel
    assert "Pick another run in the sidebar" in panel, "does not say what to do next"


def test_a_patient_whose_output_cannot_be_read_is_still_told_to_re_run_extract(review_screen):
    """The counterpart state: a patient WAS selected and their extract output is
    unreadable. That one really is fixed by re-running extract, and folding it into the
    no-patient message would lose the only accurate instruction of the three."""
    screen = review_screen(RUN, "Patient_A")

    panel = screen.panel_text("extract_meta_panel")

    assert "Patient_A" in panel
    assert "Re-run extract for this patient" in panel


def test_nothing_selected_is_told_to_produce_a_run(review_screen):
    """The third state: nothing selected because no run exists. Re-running extract is
    not the next step here — there is no run to re-run."""
    screen = review_screen()

    panel = screen.panel_text("extract_meta_panel")

    assert "Start tab" in panel
    assert "junior run" in panel
    assert "Re-run extract for this patient" not in panel
