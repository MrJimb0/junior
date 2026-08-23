"""Execution state machine — append-only, SLURM-array safe.

Every unit of work logs state transitions to a per-writer fragment under
``data/state_fragments/state.<id>.jsonl``. The fragment id is derived from
SLURM array env vars when present, or falls back to PID otherwise — so a
laptop dev session, a single SLURM job, and a 500-task SLURM array all
produce non-overlapping fragments.

After the run finishes, ``summary.write_summary`` calls
``merge_state_fragments`` which merges every fragment into the canonical
``data/state.jsonl`` (sorted by timestamp). Readers go through
``iter_transitions``, which transparently reads either form — so health,
inspect, and validate work both before and after the merge.

Why fragments instead of a shared file with locks? POSIX ``O_APPEND``
atomicity only holds up to ``PIPE_BUF`` (4 KB on Linux) and our envelopes
routinely exceed that. ``fcntl.flock`` works on Lustre but is unreliable
on NFS, and a cluster filesystem has both. Fragments sidestep the question entirely.

Entities (identified by their keyed tuple):

* ``run``            -> (run_id,)
* ``patient``        -> (run_id, patient_id)
* ``step``          -> (run_id, patient_id, step)
* ``variable``       -> (run_id, patient_id, variable)
* ``step_instance`` -> (run_id, patient_id, variable, step_id)

Valid states: ``not_started``, ``queued``, ``running``, ``completed``,
``failed``, ``cancelled``, ``invalidated``.

Invalidation rules live in ``invalidation.py``; this module only records
and replays transitions.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)

_VALID_STATES = {
    "not_started", "queued", "running",
    "completed", "failed", "cancelled", "invalidated",
}

_VALID_KINDS = {"run", "patient", "step", "variable", "step_instance"}


@dataclass(frozen=True)
class Entity:
    """Keyed identifier for something the state machine tracks."""

    kind: str
    run_id: str
    patient_id: str | None = None
    step: str | None = None
    variable: str | None = None
    step_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Unknown entity kind: {self.kind}")
        needs = {
            "run": set(),
            "patient": {"patient_id"},
            "step": {"patient_id", "step"},
            "variable": {"patient_id", "variable"},
            "step_instance": {"patient_id", "variable", "step_id"},
        }[self.kind]
        for field_name in needs:
            if getattr(self, field_name) is None:
                raise ValueError(f"Entity kind={self.kind} requires {field_name}")

    def key(self) -> tuple:
        return (self.kind, self.run_id, self.patient_id, self.step, self.variable, self.step_id)

    def to_payload(self) -> dict:
        out = {"kind": self.kind, "run_id": self.run_id}
        if self.patient_id is not None:
            out["patient_id"] = self.patient_id
        if self.step is not None:
            out["step"] = self.step
        if self.variable is not None:
            out["variable"] = self.variable
        if self.step_id is not None:
            out["step_id"] = self.step_id
        return out


def _fragment_id() -> str:
    """Per-writer fragment id: SLURM array task, SLURM job, or PID — whichever is most specific."""
    array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_job and array_task:
        return f"{array_job}_{array_task}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        return f"slurm_{job}"
    return f"pid_{os.getpid()}"


def _fragment_dir(run_root: Path) -> Path:
    return Path(run_root) / "state_fragments"


def _state_path(run_root: Path) -> Path:
    return _fragment_dir(run_root) / f"state.{_fragment_id()}.jsonl"


def merged_state_path(run_root: Path) -> Path:
    """Final merged state file, written by ``summary.write_summary``."""
    return Path(run_root) / "state.jsonl"


def _transition_sensitivity(entity: Entity) -> str:
    return "medium" if entity.patient_id is not None else "low"


def record_transition(
    run_root: Path,
    *,
    entity: Entity,
    from_state: str | None,
    to_state: str,
    reason: str,
    step_context: str,
    child_run_id: str | None = None,
    code_lock_hash: str | None = None,
) -> dict:
    """Append one transition envelope to this writer's fragment and return it."""
    if to_state not in _VALID_STATES:
        raise ValueError(f"Unknown target state: {to_state}")
    if from_state is not None and from_state not in _VALID_STATES:
        raise ValueError(f"Unknown source state: {from_state}")

    path = _state_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "entity": entity.to_payload(),
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason,
        "ts": datetime.now(UTC).isoformat(),
        "child_run_id": child_run_id,
    }
    env = envelope_for(
        artifact_type="state_transition",
        sensitivity=_transition_sensitivity(entity),
        stream="data",
        run_id=entity.run_id,
        step=step_context,
        patient_id=entity.patient_id,
        variable=entity.variable,
        step_id=entity.step_id,
        payload=payload,
        code_lock_hash=code_lock_hash,
    )
    env["content_hash"] = hash_artifact_payload(env)
    validate_artifact(env, "state_transition")

    line = json.dumps(env, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return env


def iter_transitions(run_root: Path) -> Iterable[dict]:
    """Yield every transition envelope for this run — the merged file AND any fragments,
    deduplicated. A first merge must not hide later writers: a fragment written AFTER the
    merge would otherwise be missed because the merged file alone is already stale.

    Bad lines are skipped silently; callers that care about parse errors should
    use ``merge_state_fragments``, which returns a count of dropped rows.
    """
    seen: set[str] = set()

    def _iter_file(path: Path) -> Iterable[dict]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    env = json.loads(s)
                except json.JSONDecodeError:
                    continue
                # dedup on the canonical envelope so a transition already in the merged
                # file is not re-yielded from its still-present fragment.
                key = json.dumps(env, sort_keys=True, ensure_ascii=False, default=str)
                if key in seen:
                    continue
                seen.add(key)
                yield env

    merged = merged_state_path(run_root)
    if merged.is_file():
        yield from _iter_file(merged)

    frag_dir = _fragment_dir(run_root)
    if frag_dir.is_dir():
        for path in sorted(frag_dir.glob("state.*.jsonl")):
            yield from _iter_file(path)


def merge_state_fragments(run_root: Path) -> dict[str, int]:
    """Merge per-writer fragments into a single ``state.jsonl`` sorted by ``payload.ts``.

    Returns ``{"fragments": N, "lines_kept": M, "lines_dropped": K}``.
    """
    frag_dir = _fragment_dir(run_root)
    if not frag_dir.is_dir():
        return {"fragments": 0, "lines_kept": 0, "lines_dropped": 0}

    rows: list[tuple[str, str]] = []
    fragments = 0
    dropped = 0
    for path in sorted(frag_dir.glob("state.*.jsonl")):
        fragments += 1
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if not s.strip():
                    continue
                try:
                    env = json.loads(s)
                    ts = env["payload"].get("ts", "")
                except (json.JSONDecodeError, KeyError, TypeError):
                    dropped += 1
                    continue
                rows.append((ts, s))

    rows.sort(key=lambda r: r[0])
    out = merged_state_path(run_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        "\n".join(line for _, line in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    tmp.rename(out)
    return {"fragments": fragments, "lines_kept": len(rows), "lines_dropped": dropped}
