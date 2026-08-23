"""Exhaust parquet egress is content-scanned, not name-allowlisted.

A finalized, manifest-matching, PHI-clean exhaust parquet may egress; a tampered one,
a hash-mismatched one, or one with a raw id/date in a cell is rejected. Non-exhaust
binaries stay fail-closed.
"""
from __future__ import annotations

import json

import polars as pl

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl
from jr_pipeline.runtime_infrastructure.exhaust.egress import verify_exhaust_parquet
from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust
from jr_pipeline.runtime_infrastructure.exhaust.writer import emit

RUN = "20260617_000000"


def setup_function(_):
    sl._CACHE.clear()


def _recipe_header() -> dict:
    return dict(
        site_id="site_a", study_id="study_a", run_id=RUN, emitted_month="2026-06",
        encoder_fingerprint="enc0", reranker_fingerprint="rer0", extractor_fingerprint="ext0",
        code_lock_hash="sha256:abc", chunking_config_id="chunk0",
        recipe_id="date_of_birth", recipe_version="v1", variable_key="date_of_birth",
        archetype="point_fact", prompt_hash="sha256:abcd", schema_hash="sha256:ef01",
    )


def _emit_and_finalize(tmp_path):
    emit("extraction_outcome", {**_recipe_header(), "ok": True, "value_shape": "scalar"},
         run_id=RUN, phi_keys={"patient_surrogate": "Patient_A"}, data_root=tmp_path)
    finalize_exhaust(RUN, tmp_path)
    return layout.no_phi_exhaust_dir(RUN, tmp_path) / "extraction_outcome.parquet"


def test_clean_exhaust_parquet_passes_egress(tmp_path):
    parquet = _emit_and_finalize(tmp_path)
    assert verify_exhaust_parquet(parquet, patient_ids=["Patient_A"]) == []


def test_hash_mismatch_is_rejected(tmp_path):
    parquet = _emit_and_finalize(tmp_path)
    # Re-write the parquet with an extra benign row so the file hash no longer matches
    # the manifest entry -> "not produced by this writer".
    df = pl.read_parquet(parquet)
    pl.concat([df, df.head(1)]).write_parquet(parquet)
    findings = verify_exhaust_parquet(parquet, patient_ids=[])
    assert any("manifest hash mismatch" in f for f in findings)


def test_raw_phi_cell_is_rejected_even_with_matching_hash(tmp_path):
    parquet = _emit_and_finalize(tmp_path)
    # Tamper a cell with a full clinical date, then re-stamp the manifest hash so only
    # the row-level content scan can catch it.
    df = pl.read_parquet(parquet).with_columns(pl.lit("seen 2026-01-15").alias("error_code"))
    df.write_parquet(parquet)
    manifest_path = layout.no_phi_manifest_path(RUN, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["record_types"]["extraction_outcome"]["file_sha256"] = hash_file(parquet)
    manifest_path.write_text(json.dumps(manifest))
    findings = verify_exhaust_parquet(parquet, patient_ids=[])
    assert any("error_code" in f for f in findings)  # the tampered cell is rejected


def test_patient_id_in_a_cell_is_rejected(tmp_path):
    parquet = _emit_and_finalize(tmp_path)
    df = pl.read_parquet(parquet).with_columns(pl.lit("Patient_A").alias("error_code"))
    df.write_parquet(parquet)
    manifest_path = layout.no_phi_manifest_path(RUN, tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["record_types"]["extraction_outcome"]["file_sha256"] = hash_file(parquet)
    manifest_path.write_text(json.dumps(manifest))
    findings = verify_exhaust_parquet(parquet, patient_ids=["Patient_A"])
    assert any("patient id present" in f for f in findings)


def test_export_run_metadata_ships_a_clean_exhaust_run(tmp_path):
    from jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata import (
        export_run_metadata,
    )
    _emit_and_finalize(tmp_path)
    out = export_run_metadata(run_id=RUN, output_path=tmp_path / "bundle.zip", dr=tmp_path)
    assert out.is_file()
