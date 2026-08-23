"""Egress verification for NO_PHI exhaust Parquet tables.

The egress scanner fails closed on every binary it cannot read as text -- which is
correct, except for the one binary the pipeline legitimately ships: a finalized
``NO_PHI/<run>/exhaust/<record_type>.parquet``. Allowlisting it *by name* would skip
scanning entirely, so instead this verifies the table is genuinely ours and clean:

* it sits at ``.../exhaust/<record_type>.parquet`` with a known ``record_type``;
* the run's exhaust manifest lists it with a ``file_sha256`` that matches the file on
  disk — which pins the file to finalize's schema-validated output (rows are NOT
  re-validated here; see the in-code note on why that is stronger, not weaker); and
* every row passes the same forbidden-content scan as ``emit`` (no raw id/date/text),
  with no run patient id present anywhere in the row.

Any failure is returned as a finding (the export aborts). A non-exhaust binary is not
handled here and stays fail-closed in the caller.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.exhaust.forbidden_content import (
    scan_record_for_forbidden_content,
)
from jr_pipeline.runtime_infrastructure.exhaust.schema import RECORD_SCHEMAS


def verify_exhaust_parquet(path: Path, *, patient_ids: list[str]) -> list[str]:
    """Return forbidden-content/provenance findings for an exhaust parquet (empty =
    clean and allowed to egress). Derives run/manifest from the path -- no data_root."""
    if path.parent.name != "exhaust" or path.suffix != ".parquet":
        return [f"{path.name}: parquet outside the exhaust tree (fail closed)"]
    record_type = path.stem
    if record_type not in RECORD_SCHEMAS:
        return [f"{path.name}: unknown exhaust record_type {record_type!r}"]

    manifest_path = path.parent.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"{path.name}: no readable exhaust manifest beside the table"]
    entry = (manifest.get("record_types") or {}).get(record_type)
    if not entry:
        return [f"{path.name}: not listed in the exhaust manifest"]
    if entry.get("file_sha256") != hash_file(path):
        return [f"{path.name}: manifest hash mismatch (not produced by this writer / tampered)"]

    try:
        rows = pl.read_parquet(path).to_dicts()
    except Exception as exc:  # corrupt/unreadable parquet
        return [f"{path.name}: unreadable parquet ({type(exc).__name__})"]

    # NB: we do NOT re-validate each row against the pydantic schema here. The records
    # were schema-validated at emit AND at finalize (before the parquet was written), and
    # the manifest hash match above proves this file IS that finalize-validated parquet,
    # untampered -- a stronger guarantee than re-validation. Re-validation is also
    # infeasible after a parquet round-trip of variable-key dict fields (e.g. the
    # per-candidate `features` map becomes a union-struct with nulls). The egress job is
    # PHI containment: the recursive content scan + patient-id check below.
    findings: list[str] = []
    for i, row in enumerate(rows):
        findings.extend(f"{path.name}[row {i}]: {label}" for label in scan_record_for_forbidden_content(row))
        if patient_ids:
            blob = json.dumps(row, default=str)
            findings.extend(
                f"{path.name}[row {i}]: patient id present" for pid in patient_ids if pid and pid in blob
            )
    return findings
