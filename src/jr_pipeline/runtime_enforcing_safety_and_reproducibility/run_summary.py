"""Post-run summary builder — ``out/<run_id>/summary.json``.

A rollup of what finished, what failed, how long it took, and how many
chunks each patient contributed. Not an artifact envelope (no schema):
this is a derived view over state.jsonl + per-patient chunk_index
sidecars, regenerable at any time.

``write_summary`` is also the single point where:

  * Per-task ``data/state_fragments/state.<id>.jsonl`` files are merged
    into the canonical ``data/state.jsonl``.
  * The manifest's ``status`` is flipped from ``running`` to ``completed``
    or ``failed`` (the only place that transition happens).

Lives at the top of the run tree so an operator can answer "did this run
succeed?" without grepping state.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    iter_transitions,
    merge_state_fragments,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
)


def _parse_ts(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _patient_chunk_count(run_root: Path, patient_id: str) -> int | None:
    """Read chunk_index.parquet.meta.json for ``patient_id``; return ``None`` if absent."""
    meta = run_root / "patients" / patient_id / "chunk_index.parquet.meta.json"
    if not meta.is_file():
        return None
    try:
        env = json.loads(meta.read_text(encoding="utf-8"))
        return int(env.get("payload", {}).get("row_count", 0))
    except Exception:
        return None


def _runner_stage_flags(run_root: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    """Read cohort_result.json (written by run_cohort) for per-patient per-stage status.

    Embed/index/extract can be *skipped* (no torch, no embeddings, no allowlist) without
    recording a state transition, so those half-successes are invisible to the state-based
    rollup above. Surfacing them here keeps a half-built cohort off a clean ``completed``.
    Returns (stage_status, n_skipped, n_failed); empty/zeros when the file is absent."""
    p = Path(run_root) / "cohort_result.json"
    if not p.is_file():
        return {}, 0, 0
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, 0, 0
    stage_status: dict[str, dict[str, str]] = {}
    n_skipped = n_failed = 0
    for stage in ("embed_status", "index_status", "extract_status"):
        statuses = rec.get(stage) or {}
        stage_status[stage] = statuses
        for v in statuses.values():
            s = str(v)
            if s.startswith("skipped:"):
                n_skipped += 1
            elif s.startswith("failed:"):
                n_failed += 1
    return stage_status, n_skipped, n_failed


def _patient_validate(run_root: Path, patient_id: str) -> dict[str, Any] | None:
    p = run_root / "patients" / patient_id / "validate_embed.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_summary(run_root: Path) -> dict[str, Any]:
    """Compute the summary dict. Does not write it."""
    run_root = Path(run_root)

    # Single point where SLURM-array fragments become the canonical state.jsonl
    # that downstream tools (health, validate, inspect) read.
    merge_report = merge_state_fragments(run_root)

    mpath = run_root / "manifest.json"
    if not mpath.is_file():
        raise FileNotFoundError(f"No manifest at {mpath} — cannot summarize.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    m_payload = manifest["payload"]

    step_counts: dict[str, dict[str, int]] = {}
    per_patient_steps: dict[str, dict[str, str]] = {}
    earliest: datetime | None = None
    latest: datetime | None = None

    for env in iter_transitions(run_root):
        payload = env["payload"]
        ent = payload["entity"]
        ts = _parse_ts(payload.get("ts", ""))
        if ts is not None:
            earliest = ts if earliest is None or ts < earliest else earliest
            latest = ts if latest is None or ts > latest else latest

        if ent["kind"] != "step":
            continue
        step = ent.get("step")
        pid = ent.get("patient_id")
        to_state = payload.get("to_state")
        if step is None or pid is None or to_state is None:
            continue
        step_counts.setdefault(step, {})
        step_counts[step][to_state] = step_counts[step].get(to_state, 0) + 1
        per_patient_steps.setdefault(pid, {})[step] = to_state

    # The raw roster lives in the PHI-side run_roster.json (not the low-sensitivity
    # manifest); read it here for the declared list (the summary lives PHI-side at
    # run_root, so the union with observed pids stays PHI-side).
    declared: list[str] = []
    roster_path = run_root / "run_roster.json"
    if roster_path.is_file():
        try:
            declared = list(json.loads(roster_path.read_text(encoding="utf-8")).get("target_patients") or [])
        except (OSError, json.JSONDecodeError):
            declared = []
    observed = sorted(per_patient_steps.keys())
    patients: list[str] = list(declared)
    for pid in observed:
        if pid not in patients:
            patients.append(pid)
    per_patient: list[dict[str, Any]] = []
    for pid in patients:
        entry: dict[str, Any] = {
            "patient_id": pid,
            "steps": per_patient_steps.get(pid, {}),
            "chunk_count": _patient_chunk_count(run_root, pid),
        }
        vr = _patient_validate(run_root, pid)
        if vr is not None:
            entry["validate_embed_ok"] = bool(vr.get("ok", False))
            if not vr.get("ok", True):
                entry["validate_embed_failures"] = sorted(
                    name for name, c in vr.get("checks", {}).items()
                    if not c.get("ok", False)
                )
        per_patient.append(entry)

    n_patients = len(patients)
    per_step_success: dict[str, int] = {}
    per_step_failure: dict[str, int] = {}
    for step in step_counts:
        per_step_success[step] = sum(
            1 for pid in patients
            if per_patient_steps.get(pid, {}).get(step) == "completed"
        )
        per_step_failure[step] = sum(
            1 for pid in patients
            if per_patient_steps.get(pid, {}).get(step) == "failed"
        )

    runtime_seconds: float | None = None
    if earliest and latest:
        runtime_seconds = round((latest - earliest).total_seconds(), 3)

    total_chunks = sum(
        (p["chunk_count"] or 0) for p in per_patient if p["chunk_count"] is not None
    )

    n_extraction_failed = _count_extraction_failures(run_root)
    runner_stage_status, n_stage_skipped, n_stage_failed = _runner_stage_flags(run_root)

    return {
        "run_id": m_payload["run_id"],
        "status": m_payload.get("status"),
        "model_sha256": m_payload.get("model_sha256"),
        "code_lock_hash": m_payload["code_lock_hash"],
        "n_patients": n_patients,
        # Extraction-level failures (a variable returned ok:false) are distinct
        # from a step crashing — they must keep the run off a clean "completed".
        "n_extraction_failed": n_extraction_failed,
        # A skipped/failed stage (no torch / no embeddings / no allowlist) is invisible to
        # the state-based counts but must also keep the run off a clean "completed".
        "runner_stage_status": runner_stage_status,
        "n_stage_skipped": n_stage_skipped,
        "n_stage_failed": n_stage_failed,
        "per_step_completed": per_step_success,
        "per_step_failed": per_step_failure,
        "step_state_counts": step_counts,
        "total_chunks": total_chunks,
        "runtime_seconds": runtime_seconds,
        "started_at": earliest.isoformat() if earliest else None,
        "ended_at": latest.isoformat() if latest else None,
        "per_patient": per_patient,
        # lines_dropped > 0 means a fragment had a malformed line — investigate.
        "state_merge": merge_report,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _count_extraction_failures(run_root: Path) -> int:
    """Count variable results with ``ok == false`` across all patients.

    A step can run cleanly while every extraction inside it returns ``ok:false``;
    that must not read as a clean run. The live tree is ``patients/<pid>/extract/
    <var>/result.json`` (envelope → payload.ok)."""
    patients_dir = Path(run_root) / "patients"
    if not patients_dir.is_dir():
        return 0
    n = 0
    for result_file in patients_dir.glob("*/extract/*/result.json"):
        try:
            env = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload = env.get("payload", env) if isinstance(env, dict) else {}
        if isinstance(payload, dict) and payload.get("ok") is False:
            n += 1
    return n


def write_summary(run_root: Path) -> Path:
    """Build + atomically write ``summary.json`` with the FINAL status, then flip the
    manifest status to completed / completed_with_errors / failed.

    The status is computed and stamped into the summary BEFORE the single write, so the
    operator-facing summary.json never gets stuck reading ``running``.
    """
    summary = build_summary(run_root)

    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
        mark_completed,
    )
    # Written BEFORE the status is decided, because the answers are what the run was
    # for: a run that could not write them has not finished cleanly and must not be
    # recorded as though it had.
    tables_not_written = _write_the_runs_answers(run_root)
    if tables_not_written:
        summary["values_tables_not_written"] = tables_not_written
    failed = any(int(n) > 0 for n in summary["per_step_failed"].values())
    # No transitions recorded = catastrophic failure; don't mark completed.
    no_work = (
        not summary.get("per_step_completed")
        or sum(int(n) for n in summary.get("per_step_completed", {}).values()) == 0
    )
    if failed or no_work:
        status = "failed"
    elif tables_not_written:
        # The run's primary deliverable is missing. It used to be a bare print() on a
        # channel `junior finish` captures and throws away on success, so the operator
        # saw three green ticks over a run with no answers in it.
        status = "completed_with_errors"
    elif summary.get("n_extraction_failed", 0) > 0:
        # every step ran, but at least one variable came back ok:false — honest
        # status so an all-extractions-failed run never reads as clean "completed".
        status = "completed_with_errors"
    elif summary.get("n_stage_skipped", 0) > 0 or summary.get("n_stage_failed", 0) > 0:
        # a whole stage was skipped (no torch / no embeddings / no allowlist) or failed for
        # some patient while the steps that DID run completed — a half-built cohort must not
        # read as a clean "completed".
        status = "completed_with_errors"
    else:
        status = "completed"

    summary["status"] = status
    path = Path(run_root) / "summary.json"
    atomic_write_json(path, summary)
    mark_completed(run_root, status=status)
    # LAST, once both the answers and the summary are on disk. The index lists what a
    # run holds worth opening, so rendering it any earlier describes a run that does
    # not exist yet: rendered between the tables and the summary, it named the tables
    # and left out "summary.json — what finished, what failed", which is the line an
    # operator looks for after a run they were not watching.
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        write_project_run_index,
    )

    write_project_run_index(Path(run_root).parent)
    return path


def _write_the_runs_answers(run_root: Path) -> str | None:
    """Write the run's answer tables, and say what went wrong if they could not be.

    The answers are why the run was run: answers/<run id>/<variable>_results.xlsx, one
    per recipe, one dated folder per run beside every other run's inside the project's
    CONTAINS_PHI folder. Still never fatal — the result.json files they are
    built from are all still there, and a table that cannot be written must not
    un-finish a finished run — but the reason is returned so it can reach both the
    summary and the run's status instead of only a print nobody sees.

    Writing them is all this does. The project index that points at them is rendered by
    the caller once the summary is written too.
    """
    from jr_pipeline.runtime_infrastructure.values_table import write_values_tables

    try:
        write_values_tables(run_root)
    except Exception as why_not:  # noqa: BLE001 — reported three ways, never fatal
        # stderr, not stdout: `junior finish` redirects stdout into a buffer it drops on
        # success, so this was the one message about a missing deliverable and it went
        # nowhere.
        print(f"  (note: the run's values tables were not written: {why_not})",
              file=sys.stderr)
        return f"{type(why_not).__name__}: {why_not}"
    return None
