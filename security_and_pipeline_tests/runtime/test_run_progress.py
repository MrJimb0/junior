"""Cohort position and time-left, read back from a run's own state log.

A stage command runs one patient and cannot count the cohort itself, so it reads the
run directory. Two things make that non-obvious and are pinned here: transitions live
in per-process fragments until a run is summarized (so reading only the merged
state.jsonl shows nothing mid-run, which is when progress is wanted), and a cached
patient does no work (so averaging its instant duration would promise a finish far
sooner than the truth).
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from jr_pipeline.runtime_infrastructure.run_progress import (
    count_patient_folders,
    format_duration,
    read_stage_progress,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _transition(patient: str, step: str, state: str, at: datetime) -> str:
    return json.dumps({
        "artifact_type": "state_transition",
        "payload": {
            "entity": {"kind": "step", "patient_id": patient, "run_id": "r", "step": step},
            "to_state": state,
            "ts": at.isoformat(),
        },
    })


def _write_patient(run_root: Path, patient: str, step: str, seconds: float, offset: int,
                   *, fragment: bool = True) -> None:
    """Record one completed (patient, step) pair taking `seconds`."""
    began = START + timedelta(seconds=offset)
    lines = [
        _transition(patient, step, "running", began),
        _transition(patient, step, "completed", began + timedelta(seconds=seconds)),
    ]
    if fragment:
        target = run_root / "state_fragments" / f"state.pid_{patient}.jsonl"
    else:
        target = run_root / "state.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def test_counts_completed_patients_from_unmerged_fragments(tmp_path):
    """Mid-run is exactly when progress is wanted, and mid-run nothing is merged yet."""
    for index, patient in enumerate(["P1", "P2", "P3"]):
        _write_patient(tmp_path, patient, "embed", seconds=4, offset=index * 10)

    progress = read_stage_progress(tmp_path, "embed", total=10)
    assert progress.completed == 3
    assert progress.remaining == 6  # after the one now running
    assert progress.mean_seconds == 4


def test_reads_merged_and_unmerged_together(tmp_path):
    _write_patient(tmp_path, "P1", "embed", seconds=4, offset=0, fragment=False)
    _write_patient(tmp_path, "P2", "embed", seconds=4, offset=10)
    assert read_stage_progress(tmp_path, "embed", total=5).completed == 2


def test_only_counts_the_stage_asked_for(tmp_path):
    _write_patient(tmp_path, "P1", "ingest", seconds=1, offset=0)
    _write_patient(tmp_path, "P1", "embed", seconds=4, offset=10)
    assert read_stage_progress(tmp_path, "embed", total=5).completed == 1


def test_cached_patients_do_not_drag_the_estimate_down(tmp_path):
    """A cached patient returns instantly. Averaging it in would predict a finish
    far sooner than the remaining real work will take."""
    _write_patient(tmp_path, "P1", "embed", seconds=0.0, offset=0)   # cache hit
    _write_patient(tmp_path, "P2", "embed", seconds=20.0, offset=10)  # real work

    progress = read_stage_progress(tmp_path, "embed", total=4)
    assert progress.mean_seconds == 20.0


def test_no_estimate_before_there_is_history(tmp_path):
    """The first patient of a stage has nothing to average; saying so beats guessing."""
    progress = read_stage_progress(tmp_path, "embed", total=5)
    assert progress.completed == 0
    assert progress.mean_seconds is None
    assert progress.estimated_seconds_left(3.0) is None


def test_estimate_covers_the_current_patient_and_the_rest(tmp_path):
    _write_patient(tmp_path, "P1", "embed", seconds=10.0, offset=0)

    progress = read_stage_progress(tmp_path, "embed", total=4)
    # 1 done, 1 running, 2 left. Four seconds into the running one:
    # 6s to finish it + 10s x 2 still queued.
    assert progress.estimated_seconds_left(4.0) == 26.0
    # Past the average, the current patient contributes nothing further.
    assert progress.estimated_seconds_left(30.0) == 20.0


def test_unknown_cohort_size_yields_no_estimate(tmp_path):
    _write_patient(tmp_path, "P1", "embed", seconds=10.0, offset=0)
    progress = read_stage_progress(tmp_path, "embed", total=None)
    assert progress.remaining is None
    assert progress.estimated_seconds_left(1.0) is None


def test_a_half_written_fragment_does_not_break_the_reading(tmp_path):
    """Fragments are appended to by live processes; a truncated last line is normal."""
    _write_patient(tmp_path, "P1", "embed", seconds=4, offset=0)
    fragment = tmp_path / "state_fragments" / "state.pid_partial.jsonl"
    fragment.write_text('{"payload": {"entity": {"step": "emb', encoding="utf-8")
    assert read_stage_progress(tmp_path, "embed", total=3).completed == 1


def test_counts_patient_folders(tmp_path):
    for name in ("P1", "P2", ".hidden"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert count_patient_folders(tmp_path) == 2
    assert count_patient_folders(tmp_path / "missing") is None
    assert count_patient_folders(None) is None


def test_durations_read_as_rough_english():
    assert format_duration(45) == "45s"
    assert format_duration(360) == "6m"
    assert format_duration(4800) == "1h20m"
