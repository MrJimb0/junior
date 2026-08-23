"""Per-patient data-stream artifacts: source_snapshot.json.

Writes the sha256 fingerprint of every input file (csv / tsv / xlsx /
json / jsonl / parquet) at ingest time. This is what the invalidation
rules consult to decide whether a patient's ingest / embed / index
outputs are still valid.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.source_file_discovery import discover_source_files


def _count_rows_best_effort(path: Path) -> int | None:
    """Count data rows without materializing the file; None for unsupported
    formats or on any read error. csv/tsv use a lazy ``scan_csv`` row count so a
    quoted field with an embedded newline counts as one row, not several (the
    old physical-line count over-counted those)."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        try:
            return int(
                pl.scan_csv(
                    path, separator=separator, has_header=True, infer_schema_length=0
                ).select(pl.len()).collect().item()
            )
        except Exception:
            return None
    if suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return None
    return None


def write_source_snapshot(
    *,
    run_id: str,
    patient_id: str,
    patient_dir: Path,
    patient_out_dir: Path,
    code_lock_hash: str | None = None,
) -> Path:
    """Write ``data/patients/<pid>/source_snapshot.json``."""
    patient_dir = Path(patient_dir)
    patient_out_dir = Path(patient_out_dir)
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    # Fingerprint exactly the files ingest will read — the shared discovery's
    # per-stem winners, matched case-insensitively — so the snapshot can never
    # miss a file ingest opens (bug B3).
    files: list[dict] = []
    for path in sorted(discover_source_files(patient_dir).values(), key=lambda p: p.name):
        stat = path.stat()
        files.append({
            "relpath": path.relative_to(patient_dir).as_posix(),
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            "content_hash": hash_file(path),
            "row_count": _count_rows_best_effort(path),
        })

    payload = {
        "patient_id": patient_id,
        "source_root": str(patient_dir),
        "files": files,
    }
    env = envelope_for(
        artifact_type="source_snapshot",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="ingest",
        patient_id=patient_id,
        payload=payload,
        code_lock_hash=code_lock_hash,
    )
    return write_artifact(env, patient_root=patient_out_dir)  # -> <patient_root>/source_snapshot.json


def read_source_snapshot(patient_out_dir: Path) -> dict | None:
    """Read a patient's snapshot, or None if not yet written."""
    path = Path(patient_out_dir) / "source_snapshot.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def snapshot_matches(patient_out_dir: Path, patient_dir: Path) -> bool:
    """True iff on-disk source files still match the stored snapshot."""
    stored = read_source_snapshot(patient_out_dir)
    if stored is None:
        return False
    stored_files = {f["relpath"]: f["content_hash"] for f in stored["payload"]["files"]}

    patient_dir = Path(patient_dir)
    current = {
        path.relative_to(patient_dir).as_posix(): hash_file(path)
        for path in discover_source_files(patient_dir).values()
    }
    return stored_files == current
