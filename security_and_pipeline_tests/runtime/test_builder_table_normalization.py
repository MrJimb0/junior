"""Builder-intermediate CSVs normalize identically to a plain step-1 ingest.

The load-bearing guarantee: a builder table fed through ``normalize_builder_source``
is byte-for-byte the same parquet that ``run_ingest_one`` would write for the same
CSV (same read-as-text, same empty-row strip, same ISO date canonicalization, same
snappy write) — so it is indistinguishable in format from any other source. Plus
the derived-table caching + provenance the retrieve_and_prompt step relies on.
"""
from __future__ import annotations

import polars as pl

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.builder_table_normalization import (
    normalize_builder_source,
)
from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import run_ingest_one
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)

_CSV = "specimen,dx_date\nleft breast,1/2/2024\nright breast,3/4/2023\n"


def test_builder_parquet_is_byte_identical_to_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    src_root = tmp_path / "src"
    pdir = src_root / "P1"
    pdir.mkdir(parents=True)
    (pdir / "path_cap.csv").write_text(_CSV)

    cfg = {"run_id": "20260101_b", "source_root": str(src_root), "files": "auto"}
    run_ingest_one(cfg=cfg, patient_id="P1")
    ingest_parquet = phi_patient_run_dir("20260101_b", "P1") / "structured" / "path_cap.parquet"

    dest = tmp_path / "derived" / "path_cap.parquet"
    normalize_builder_source(
        pdir / "path_cap.csv", dest,
        run_id="20260101_b", patient_id="P1", origin_run_id="20260101_b",
    )

    assert hash_file(dest) == hash_file(ingest_parquet)  # same bytes as a real ingest
    # and the date column was ISO-canonicalized, same as ingest
    assert pl.read_parquet(dest)["dx_date"].to_list() == ["2024-01-02", "2023-03-04"]


def test_sidecar_records_origin_run_id_and_hashes(tmp_path):
    src = tmp_path / "path_cap.csv"
    src.write_text(_CSV)
    dest = tmp_path / "derived" / "path_cap.parquet"

    payload = normalize_builder_source(
        src, dest, run_id="cur", patient_id="P1", origin_run_id="prior_run",
    )
    assert payload["origin_run_id"] == "prior_run"
    assert payload["source_content_hash"] == hash_file(src)
    assert payload["parquet_content_hash"] == hash_file(dest)
    assert payload["row_count"] == 2


def test_cache_skips_when_source_unchanged(tmp_path):
    src = tmp_path / "t.csv"
    src.write_text("a,b\n1,2\n")
    dest = tmp_path / "d" / "t.parquet"
    first = normalize_builder_source(src, dest, run_id="r", patient_id="P", origin_run_id="o")

    # Corrupt the parquet but leave the sidecar: an unchanged source is a cache hit,
    # so the function returns the recorded payload and does NOT rewrite the parquet.
    dest.write_bytes(b"CORRUPT")
    again = normalize_builder_source(src, dest, run_id="r", patient_id="P", origin_run_id="o")
    assert again == first
    assert dest.read_bytes() == b"CORRUPT"  # not re-normalized


def test_changed_source_re_normalizes(tmp_path):
    src = tmp_path / "t.csv"
    src.write_text("a,b\n1,2\n")
    dest = tmp_path / "d" / "t.parquet"
    normalize_builder_source(src, dest, run_id="r", patient_id="P", origin_run_id="o")
    dest.write_bytes(b"CORRUPT")

    src.write_text("a,b\n9,9\n")  # source bytes change -> sidecar hash mismatch -> rewrite
    normalize_builder_source(src, dest, run_id="r", patient_id="P", origin_run_id="o")
    assert dest.read_bytes() != b"CORRUPT"
    # ingest reads every cell as text on purpose, so the value is the string "9"
    assert pl.read_parquet(dest)["b"].to_list() == ["9"]
