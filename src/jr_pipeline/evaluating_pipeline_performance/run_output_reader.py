"""Stable readers over a run's on-disk artifact tree.

Works for any run dir — a sealed CLI run or an unsealed quickstart/cohort run.
The read-side commands (``health``/``inspect``/``summarize``) need no seal; only
the per-stage *write* subcommands require a sealed bundle. Every consumer reads
through this module rather than grubbing through files directly, so layout
changes stay localized here.

The on-disk shape is ``<run_root>/{code, manifest.json, state.jsonl,
patients/<pid>/…}`` — there is no ``data/`` subdir (that was a reader bug).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    """Path resolver for the on-disk shape of one sealed run."""
    run_root: Path

    @property
    def code_dir(self) -> Path:
        return self.run_root / "code"

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "manifest.json"

    @property
    def hashes_path(self) -> Path:
        return self.code_dir / "hashes.json"

    def patient_dir(self, patient_id: str) -> Path:
        # phi_patient_run_dir == run_root / patients / <pid> (no 'data' subdir).
        return self.run_root / "patients" / patient_id

    def retrieval_artifacts(self, patient_id: str) -> list[Path]:
        d = self.patient_dir(patient_id) / "retrieve"
        return sorted(d.glob("*.json")) if d.is_dir() else []


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(run_root: Path) -> dict:
    return load_json(RunLayout(run_root).manifest_path)


def hashes(run_root: Path) -> dict:
    return load_json(RunLayout(run_root).hashes_path)


def source_snapshot(run_root: Path, patient_id: str) -> dict:
    return load_json(RunLayout(run_root).patient_dir(patient_id) / "source_snapshot.json")


def iter_state(run_root: Path):
    """Yield every state-transition envelope, preferring merged state.jsonl over per-writer fragments."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
        iter_transitions,
    )
    yield from iter_transitions(run_root)


def list_patients(run_root: Path) -> list[str]:
    base = Path(run_root) / "patients"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())
