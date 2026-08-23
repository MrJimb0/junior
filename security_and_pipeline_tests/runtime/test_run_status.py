"""A run where extractions returned ok:false must NOT report a clean
`completed`. Run status is derived in write_summary from the per-patient
result.json `ok` flags, not just from whether the step executed."""
import json
from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
    build_manifest,
    write_manifest,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.run_summary import (
    _count_extraction_failures,
    write_summary,
)

CODE_HASH = "sha256:" + "0" * 64


def _seed(run_root: Path, run_id: str, pid: str, ok_flags: dict[str, bool]) -> None:
    """A minimal but real run: manifest + step transitions + per-variable result.json."""
    run_root.mkdir(parents=True, exist_ok=True)
    write_manifest(run_root, build_manifest(
        run_id=run_id, code_lock_hash=CODE_HASH,
        entry_point_name="test", config_alias="test", target_patients=[pid],
    ))
    for step in ("ingest", "embed", "extract"):
        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=pid, step=step),
            from_state="running", to_state="completed", reason="test",
            step_context=step, code_lock_hash=CODE_HASH,
        )
    for var, ok in ok_flags.items():
        rj = run_root / "patients" / pid / "extract" / var / "result.json"
        rj.parent.mkdir(parents=True, exist_ok=True)
        rj.write_text(json.dumps({"payload": {"variable": var, "ok": ok, "data": {}}}), encoding="utf-8")


def _status(run_root: Path) -> str:
    return json.loads((run_root / "manifest.json").read_text())["payload"]["status"]


def test_count_extraction_failures(tmp_path):
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_aa", "p1", {"date_of_diagnosis": False, "date_of_birth": True})
    assert _count_extraction_failures(rr) == 1


def test_all_extractions_failed_reports_completed_with_errors(tmp_path):
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_bb", "p1", {"date_of_diagnosis": False})
    write_summary(rr)
    assert _status(rr) == "completed_with_errors", "an all-failed run must not read as completed"


def test_clean_run_reports_completed(tmp_path):
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_cc", "p1", {"date_of_diagnosis": True, "date_of_birth": True})
    write_summary(rr)
    assert _status(rr) == "completed"


def _summary_status(run_root: Path) -> str:
    return json.loads((run_root / "summary.json").read_text())["status"]


def test_summary_json_carries_final_status_not_running(tmp_path):
    # summary.json must carry the final status, not the in-progress "running".
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_dd", "p1", {"date_of_diagnosis": True, "date_of_birth": True})
    write_summary(rr)
    assert _summary_status(rr) == "completed"
    assert _summary_status(rr) == _status(rr)  # summary.json agrees with the manifest


def test_summary_json_reflects_completed_with_errors(tmp_path):
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_ee", "p1", {"date_of_diagnosis": False})
    write_summary(rr)
    assert _summary_status(rr) == "completed_with_errors"


def test_a_run_that_could_not_write_its_answers_does_not_report_clean(tmp_path, monkeypatch):
    """The answer tables are what the run was for. A failure to write them was a bare
    print() on stdout, and `junior finish` redirects stdout into a buffer it drops on
    success — so the operator saw three green ticks over a run with no answers in it."""
    from jr_pipeline.runtime_infrastructure import values_table

    def _cannot_write(_run_root):
        raise OSError("read-only file system")

    monkeypatch.setattr(values_table, "write_values_tables", _cannot_write)
    rr = tmp_path / "run"
    _seed(rr, "20260101_000000_ff", "p1", {"date_of_birth": True})

    write_summary(rr)

    assert _status(rr) == "completed_with_errors"
    summary = json.loads((rr / "summary.json").read_text())
    assert "read-only file system" in summary["values_tables_not_written"]
