"""Ingest and the builder-table write share one atomic parquet-write primitive.

write_dataframe_parquet_atomically is the single source of the tmp-file+rename write, the
snappy compression, and the common sidecar metadata — so an ingested parquet and a builder
parquet are byte-identical for the same dataframe instead of two hand-kept copies.
"""
from __future__ import annotations

import polars as pl

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
    _PARQUET_COMPRESSION,
    write_dataframe_parquet_atomically,
)


def test_writes_atomically_and_returns_common_metadata(tmp_path):
    df = pl.DataFrame({"stage": ["IIA", "IIIB"], "grade": ["2", "3"]})
    dest = tmp_path / "t.parquet"
    meta = write_dataframe_parquet_atomically(df, dest)
    assert dest.is_file()
    assert not dest.with_suffix(".parquet.tmp").is_file()  # temp file renamed away
    assert meta["row_count"] == 2
    assert meta["column_count"] == 2
    assert [c["name"] for c in meta["columns"]] == ["stage", "grade"]
    assert all("dtype" in c for c in meta["columns"])
    assert meta["compression"] == _PARQUET_COMPRESSION
    assert meta["parquet_content_hash"].startswith("sha256:")
    assert meta["polars_version"] == pl.__version__


def test_same_dataframe_yields_byte_identical_parquets(tmp_path):
    df = pl.DataFrame({"a": ["1", "2"], "b": ["x", "y"]})
    p1, p2 = tmp_path / "one.parquet", tmp_path / "two.parquet"
    m1 = write_dataframe_parquet_atomically(df, p1)
    m2 = write_dataframe_parquet_atomically(df, p2)
    assert p1.read_bytes() == p2.read_bytes()
    assert m1["parquet_content_hash"] == m2["parquet_content_hash"]
