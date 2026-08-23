"""How far through a cohort a stage is, read back from the run's own state log.

A stage command runs one patient, so on its own it cannot say "patient 12 of 40".
The run directory can: each process appends running/completed transitions per
(patient, step) with timestamps, so counting completed patients gives the position and
the gaps between the pairs give a per-patient duration to project from.

Reading the log rather than threading a counter through the call chain is what makes
this work on a cluster, where each patient is a separate process that never sees the
others.

Transitions live in ``state_fragments/state.pid_<n>.jsonl`` while a run is in flight —
one file per process, so concurrent writers never interleave — and are merged into
``state.jsonl`` when the run is summarized. Both are read here: a progress reading
taken mid-run would otherwise see nothing, which is exactly when it is wanted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class StageProgress:
    """Where a stage stands across the cohort."""
    completed: int
    total: int | None
    mean_seconds: float | None

    @property
    def remaining(self) -> int | None:
        """Patients still to do after the one now running."""
        if self.total is None:
            return None
        return max(0, self.total - self.completed - 1)

    def estimated_seconds_left(self, elapsed_on_current: float) -> float | None:
        """Projection for the rest of the cohort, including the patient in flight."""
        if self.mean_seconds is None or self.remaining is None:
            return None
        rest_of_current = max(0.0, self.mean_seconds - elapsed_on_current)
        return rest_of_current + self.mean_seconds * self.remaining


def read_stage_progress(run_root: Path, step: str, total: int | None = None) -> StageProgress:
    """Count patients that finished ``step`` in this run, and how long each took."""
    started: dict[str, datetime] = {}
    durations: list[float] = []
    finished: set[str] = set()

    for payload in _sorted_transitions(Path(run_root)):
        entity = payload.get("entity") or {}
        if entity.get("step") != step:
            continue
        patient_id = entity.get("patient_id")
        moment = _timestamp(payload.get("ts"))
        if not patient_id or moment is None:
            continue
        if payload.get("to_state") == "running":
            started[patient_id] = moment
        elif payload.get("to_state") == "completed":
            finished.add(patient_id)
            if (start := started.pop(patient_id, None)) is not None:
                durations.append((moment - start).total_seconds())

    # Ignore instant durations when averaging: a cached patient did no work, so
    # including it would predict the cohort finishing far sooner than it will.
    real_work = [seconds for seconds in durations if seconds > 0.05]
    return StageProgress(
        completed=len(finished),
        total=total,
        mean_seconds=(sum(real_work) / len(real_work)) if real_work else None,
    )


def count_patient_folders(source_root: str | Path | None) -> int | None:
    """Cohort size from the source directory: one visible subfolder per patient."""
    if not source_root:
        return None
    root = Path(source_root).expanduser()
    if not root.is_dir():
        return None
    return sum(1 for child in root.iterdir() if child.is_dir() and not child.name.startswith("."))


def format_duration(seconds: float) -> str:
    """`45s`, `6m`, `1h20m` — coarse on purpose, since it is an estimate."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours, leftover_minutes = divmod(int(minutes), 60)
    return f"{hours}h{leftover_minutes:02d}m"


def _sorted_transitions(run_root: Path) -> list[dict]:
    """Every state transition this run has recorded, oldest first.

    Merged and unmerged sources both contribute, and a patient can appear in both once
    a run has been summarized and then resumed, so pairing running/completed by
    timestamp order rather than file order is what keeps the durations honest."""
    sources = [run_root / "state.jsonl", *sorted((run_root / "state_fragments").glob("*.jsonl"))]
    payloads = [
        payload
        for source in sources
        if source.is_file()
        for payload in (_payload(line) for line in _lines(source))
        if payload is not None
    ]
    return sorted(payloads, key=lambda payload: str(payload.get("ts") or ""))


def _lines(path: Path) -> list[str]:
    # A fragment is being appended to by a live process; a truncated final line is
    # normal, and _payload drops it.
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []


def _payload(raw_line: str) -> dict | None:
    try:
        record = json.loads(raw_line)
    except ValueError:
        return None
    payload = record.get("payload") if isinstance(record, dict) else None
    return payload if isinstance(payload, dict) else None


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
