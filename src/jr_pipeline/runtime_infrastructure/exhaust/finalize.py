"""Locked finalize of NO_PHI exhaust shards into Parquet + manifest.

``emit`` appends each record to a process-unique JSONL shard. At run end, exactly one
finalizer compacts every record type's shards into a single
``NO_PHI/<run>/exhaust/<record_type>.parquet`` and writes the shareable
``NO_PHI/<run>/manifest.json`` (row counts, per-table file hashes, schema/vocab/
surrogate versions, the non-reversible secret fingerprint, and the record-type
inventory -- never a patient roster).

Finalize holds a file lock (one finalizer wins; a concurrent finalize waits), re-validates
every shard record against its schema (a tampered/garbled line is counted as
``n_records_failed``, never silently dropped), compacts via Parquet through a temp file +
atomic rename, and writes the manifest atomically. Shards are left in place, immutable,
for audit. Finalize always writes the manifest -- even a run that emitted nothing leaves
a manifest with an empty inventory, so the NO_PHI tree is self-describing.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl
from filelock import FileLock

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    no_phi_exhaust_dir,
    no_phi_exhaust_shards_dir,
    no_phi_manifest_path,
)
from jr_pipeline.runtime_infrastructure.exhaust.cell_min import (
    apply_cell_min,
    assert_all_record_types_have_a_policy,
    policy_for,
)
from jr_pipeline.runtime_infrastructure.exhaust.schema import SCHEMA_VERSION, validate_record
from jr_pipeline.runtime_infrastructure.exhaust.secret_lifecycle import (
    _run_secret_path,
    run_secret,
    secret_fingerprint,
)
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import SURROGATE_VERSION
from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import VOCAB_VERSION


def _read_shard_records(record_type_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard in sorted(record_type_dir.glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _secret_fingerprint_if_present(run_id: str, dr: Path | None) -> str | None:
    """The run secret's non-reversible fingerprint (key continuity), or None if no
    record was emitted (the secret is created lazily by the first ``emit``)."""
    if not _run_secret_path(run_id, dr).exists():
        return None
    return secret_fingerprint(run_secret(run_id, dr))


def finalize_exhaust(run_id: str, dr: Path | None = None) -> dict[str, Any]:
    """Compact this run's exhaust shards into Parquet + a manifest. Idempotent
    (re-running recompacts the same shards). Returns the manifest dict."""
    exhaust_dir = no_phi_exhaust_dir(run_id, dr)
    shards_root = no_phi_exhaust_shards_dir(run_id, dr)
    exhaust_dir.mkdir(parents=True, exist_ok=True)

    assert_all_record_types_have_a_policy()  # CELL_MIN gate: no exhaust type without a policy
    with FileLock(str(exhaust_dir / ".finalize.lock"), timeout=120):
        inventory: dict[str, dict[str, Any]] = {}
        type_dirs = sorted(p for p in shards_root.iterdir() if p.is_dir()) if shards_root.is_dir() else []
        for type_dir in type_dirs:
            record_type = type_dir.name
            raw_records = _read_shard_records(type_dir)
            if not raw_records:
                continue
            validated: list[dict[str, Any]] = []
            n_failed = 0
            for record in raw_records:
                try:
                    validated.append(validate_record(record_type, record))
                except Exception:  # tampered/garbled shard line -- count, never drop silently
                    n_failed += 1
            if not validated:
                inventory[record_type] = {"n_rows": 0, "n_records_failed": n_failed, "parquet": None}
                continue
            # CELL_MIN suppression for aggregate record types (no-op for row-level).
            validated = apply_cell_min(validated, policy_for(record_type))
            final_parquet = exhaust_dir / f"{record_type}.parquet"
            tmp = exhaust_dir / f".{record_type}.{os.getpid()}.tmp.parquet"
            pl.DataFrame(validated).write_parquet(tmp)
            os.replace(tmp, final_parquet)
            inventory[record_type] = {
                "n_rows": len(validated),
                "n_records_failed": n_failed,
                "parquet": final_parquet.name,
                "file_sha256": hash_file(final_parquet),
                "schema_version": SCHEMA_VERSION,
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "vocab_version": VOCAB_VERSION,
            "surrogate_version": SURROGATE_VERSION,
            "run_id": run_id,
            "sensitivity": "no_phi",
            "secret_fingerprint": _secret_fingerprint_if_present(run_id, dr),
            "n_record_types": len(inventory),
            "record_types": inventory,
        }
        atomic_write_json(no_phi_manifest_path(run_id, dr), manifest)
    return manifest
