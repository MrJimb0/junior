"""Normalize one builder-produced intermediate file into a structured parquet.

A *builder* is an upstream process that produces extra per-patient tables (for
example a pathology CAP synoptic table) as CSVs, keyed by a run id. To use one as
evidence it must be shaped EXACTLY like every other ingested source, so this
module runs the same step-1 pipeline — read every cell as text, drop fully-empty
rows, canonicalize date-like columns to ISO-8601 — and writes the same kind of
snappy parquet. Reusing those functions (not re-implementing them) is what
guarantees a builder table is indistinguishable in format from a normal source.

Unlike step-1 ingest, the output lands in the patient's ``derived_tables/`` folder
(not ``structured/``), so it is never embedded or reconciled against the patient's
source files, and it carries a small purpose-built provenance sidecar that records
the ``origin_run_id`` (the run whose builder produced the CSV — which may be a
prior run). The schema-bound ``ingest_file`` sidecar can't hold that extra field,
so this is a separate, deliberately minimal sidecar read only by the
retrieve_and_prompt step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.date_canonicalization import (
    _canonicalize_dates,
)
from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
    _strip_empty_rows,
    write_dataframe_parquet_atomically,
)
from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.source_format_readers import _read_source
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
)

# Companion sidecar name for a derived builder parquet. Distinct from ingest's
# ``.parquet.meta.json`` (which is the schema-validated ingest_file artifact) so the
# two never collide and the builder sidecar can carry origin_run_id freely.
_SIDECAR_SUFFIX = ".parquet.builder_meta.json"


def _sidecar_path(dest_parquet: Path) -> Path:
    return dest_parquet.with_name(dest_parquet.stem + _SIDECAR_SUFFIX)


def normalize_builder_source(
    source_path: Path,
    dest_parquet: Path,
    *,
    run_id: str,
    patient_id: str,
    origin_run_id: str,
    code_lock_hash: str | None = None,
) -> dict[str, Any]:
    """Normalize ``source_path`` into ``dest_parquet`` using the step-1 pipeline and
    return the provenance sidecar payload. ``source_path`` may be any format the
    ingest step reads (``_read_source`` dispatches on extension: csv/tsv/xlsx/
    json/jsonl/parquet), so a builder is not restricted to CSV.

    Cached: if ``dest_parquet`` and its builder sidecar already exist and the
    sidecar's ``source_content_hash`` matches the current bytes of ``source_path``, the
    work is skipped and the existing sidecar payload is returned unchanged — so a
    repeated run (or a pinned prior-run source that hasn't changed) re-normalizes
    nothing.

    ``origin_run_id`` is the run whose builder produced ``source_path`` (it may differ
    from ``run_id``, the run doing the normalization); it is recorded so a cross-run
    draw stays traceable.
    """
    source_content_hash = hash_file(source_path)
    sidecar = _sidecar_path(dest_parquet)
    if dest_parquet.is_file() and sidecar.is_file():
        existing = _read_sidecar(sidecar)
        if existing is not None and existing.get("source_content_hash") == source_content_hash:
            return existing

    # Identical to step-1 ingest: read-as-text -> strip empty rows -> ISO dates.
    normalized_table = _read_source(source_path)
    normalized_table = _strip_empty_rows(normalized_table)
    normalized_table = _canonicalize_dates(normalized_table)

    # The SAME atomic write + compression + parquet metadata as ingest, so a builder
    # parquet is byte-identical to an ingested one for the same dataframe. The builder
    # sidecar adds origin_run_id (which the schema-bound ingest_file sidecar can't hold).
    common = write_dataframe_parquet_atomically(normalized_table, dest_parquet)
    payload = {
        "name": dest_parquet.stem,
        "source_file": source_path.name,
        "origin_run_id": origin_run_id,
        "run_id": run_id,
        "patient_id": patient_id,
        "code_lock_hash": code_lock_hash,
        "source_content_hash": source_content_hash,
        **common,
    }
    atomic_write_json(sidecar, payload)
    return payload


def _read_sidecar(sidecar: Path) -> dict[str, Any] | None:
    import json

    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
