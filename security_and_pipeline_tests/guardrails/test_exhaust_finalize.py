"""Finalize compacts shards into Parquet + a roster-free manifest."""
from __future__ import annotations

import polars as pl

from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl
from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust
from jr_pipeline.runtime_infrastructure.exhaust.writer import emit

RUN = "20260617_000000"


def setup_function(_):
    sl._CACHE.clear()


def _run_header() -> dict:
    return dict(
        site_id="site_a", study_id="study_a", run_id=RUN, emitted_month="2026-06",
        encoder_fingerprint="enc0", reranker_fingerprint="rer0", extractor_fingerprint="ext0",
        code_lock_hash="sha256:abc", chunking_config_id="chunk0",
    )


def _recipe_header() -> dict:
    return {**_run_header(), "recipe_id": "date_of_birth", "recipe_version": "v1",
            "variable_key": "date_of_birth", "archetype": "point_fact",
            "prompt_hash": "sha256:abcd", "schema_hash": "sha256:ef01"}


def test_finalize_writes_parquet_and_roster_free_manifest(tmp_path):
    emit("method_provenance", _run_header(), run_id=RUN, data_root=tmp_path)
    for pid in ("Patient_A", "Patient_B"):
        emit("extraction_outcome", {**_recipe_header(), "ok": True, "value_shape": "scalar"},
             run_id=RUN, phi_keys={"patient_surrogate": pid}, data_root=tmp_path)

    manifest = finalize_exhaust(RUN, tmp_path)

    parquet = layout.no_phi_exhaust_dir(RUN, tmp_path) / "extraction_outcome.parquet"
    assert parquet.is_file()
    df = pl.read_parquet(parquet)
    assert df.height == 2
    surrogates = set(df["patient_surrogate"].to_list())
    assert all(s.startswith("surr:v1:run:patient_surrogate:") for s in surrogates)
    assert "Patient_A" not in parquet.read_bytes().decode("latin-1")  # raw id absent from the bytes

    # manifest: inventory + versions + secret fingerprint, NO patient roster
    assert manifest["sensitivity"] == "no_phi"
    assert manifest["record_types"]["extraction_outcome"]["n_rows"] == 2
    assert manifest["record_types"]["method_provenance"]["n_rows"] == 1
    assert manifest["record_types"]["extraction_outcome"]["file_sha256"].startswith("sha256:")
    assert len(manifest["secret_fingerprint"]) == 12
    assert "target_patients" not in manifest and "patient_ids" not in manifest
    mtext = layout.no_phi_manifest_path(RUN, tmp_path).read_text()
    assert "Patient_A" not in mtext and "Patient_B" not in mtext

    # idempotent: a second finalize over the same shards reproduces the row counts
    again = finalize_exhaust(RUN, tmp_path)
    assert again["record_types"]["extraction_outcome"]["n_rows"] == 2


def test_finalize_with_no_shards_writes_empty_manifest(tmp_path):
    manifest = finalize_exhaust(RUN, tmp_path)
    assert manifest["n_record_types"] == 0
    assert manifest["record_types"] == {}
    assert layout.no_phi_manifest_path(RUN, tmp_path).is_file()
    assert manifest["secret_fingerprint"] is None  # nothing emitted -> no secret created
