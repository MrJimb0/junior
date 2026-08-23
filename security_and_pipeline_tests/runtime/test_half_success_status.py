"""A half-built cohort must not read as a clean ``completed``.

embed/index/extract can be *skipped* (no torch, no embeddings, no allowlist) without
recording a state transition, so the state-based rollup misses them entirely. run_cohort
persists cohort_result.json with the per-stage status; write_summary reads it and keeps
a run with any skipped/failed stage off ``completed``.
"""
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
    _runner_stage_flags,
    write_summary,
)

CODE_HASH = "sha256:" + "0" * 64


def _seed_ingest_only(run_root: Path, run_id: str, pid: str) -> None:
    """A run that ingested cleanly (a real completed step) but did nothing else."""
    run_root.mkdir(parents=True, exist_ok=True)
    write_manifest(run_root, build_manifest(
        run_id=run_id, code_lock_hash=CODE_HASH,
        entry_point_name="test", config_alias="test", target_patients=[pid],
    ))
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=pid, step="ingest"),
        from_state="running", to_state="completed", reason="test",
        step_context="ingest", code_lock_hash=CODE_HASH,
    )


def _write_cohort_result(run_root: Path, **stages) -> None:
    rec = {"embed_status": {}, "index_status": {}, "extract_status": {}}
    rec.update(stages)
    (run_root / "cohort_result.json").write_text(json.dumps(rec), encoding="utf-8")


def _status(run_root: Path) -> str:
    return json.loads((run_root / "manifest.json").read_text())["payload"]["status"]


def test_runner_stage_flags_counts_skips_and_failures(tmp_path):
    rr = tmp_path / "run"
    rr.mkdir()
    _write_cohort_result(
        rr,
        embed_status={"p1": "skipped: no torch", "p2": "ok: 5 chunks"},
        extract_status={"p2": "failed: RuntimeError: boom"},
    )
    stage_status, n_skipped, n_failed = _runner_stage_flags(rr)
    assert n_skipped == 1
    assert n_failed == 1
    assert stage_status["embed_status"]["p1"] == "skipped: no torch"


def test_skipped_stage_keeps_run_off_clean_completed(tmp_path):
    rr = tmp_path / "run"
    _seed_ingest_only(rr, "20260101_000000_aa", "p1")
    _write_cohort_result(rr, embed_status={"p1": "skipped: no torch"})
    write_summary(rr)
    assert _status(rr) == "completed_with_errors", "a stage-skipped run must not read as completed"


def test_all_clean_stage_status_still_completes(tmp_path):
    rr = tmp_path / "run"
    _seed_ingest_only(rr, "20260101_000000_bb", "p1")
    _write_cohort_result(
        rr,
        embed_status={"p1": "ok: 5 chunks"},
        index_status={"p1": "ok: 5 vectors"},
        extract_status={"p1": "ok"},
    )
    write_summary(rr)
    assert _status(rr) == "completed"
