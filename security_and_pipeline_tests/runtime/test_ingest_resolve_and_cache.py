"""Shared file-entry resolution + the ingest cache.

``resolve_file_entries`` is the one pure entry-resolver shared by ingest and
preflight. The cache predicate also requires each parquet's sidecar (a parquet
with no sidecar can't satisfy downstream readers, so it isn't a real cache hit),
and the cached ``files_written`` shape matches the fresh one.
"""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
    resolve_file_entries,
    run_ingest_one,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)

# --- resolve_file_entries (pure, shared by ingest + preflight) ---

def test_resolve_auto_returns_sorted_discovered_stems():
    discovered = {"b": Path("b.csv"), "a": Path("a.csv")}
    assert resolve_file_entries("auto", discovered) == [
        {"stem": "a", "optional": False},
        {"stem": "b", "optional": False},
    ]


def test_resolve_none_is_treated_as_auto():
    assert resolve_file_entries(None, {"a": Path("a.csv")}) == [{"stem": "a", "optional": False}]


def test_resolve_auto_with_nothing_on_disk_is_empty_list():
    assert resolve_file_entries("auto", {}) == []


def test_resolve_explicit_list_normalizes_entries():
    out = resolve_file_entries(
        [{"stem": "x"}, {"stem": "y", "optional": True}], {"x": Path("x.csv")}
    )
    assert out == [{"stem": "x", "optional": False}, {"stem": "y", "optional": True}]


# --- cache predicate requires the sidecar; cached/fresh shapes match ---

def _seed_one_patient_csv(tmp_path):
    src_root = tmp_path / "src"
    pdir = src_root / "P1"
    pdir.mkdir(parents=True)
    (pdir / "a.csv").write_text("text\nhello\n")
    return src_root


def test_cache_hit_returns_same_files_written_shape_as_fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    src_root = _seed_one_patient_csv(tmp_path)
    cfg = {"run_id": "20260101_c", "source_root": str(src_root), "files": "auto"}

    fresh = run_ingest_one(cfg=cfg, patient_id="P1")
    cached = run_ingest_one(cfg=cfg, patient_id="P1")

    assert fresh["cached"] is False
    assert cached["cached"] is True
    # both are lists of {stem, rows, cols, parquet} dicts — and identical here
    assert cached["files_written"] == fresh["files_written"]
    assert set(cached["files_written"][0]) == {"stem", "rows", "cols", "parquet"}


def test_missing_sidecar_is_not_a_cache_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    src_root = _seed_one_patient_csv(tmp_path)
    cfg = {"run_id": "20260101_s", "source_root": str(src_root), "files": "auto"}

    run_ingest_one(cfg=cfg, patient_id="P1")
    structured = phi_patient_run_dir("20260101_s", "P1") / "structured"
    # drop the sidecar; the parquet remains. A parquet without its sidecar is not a cache hit.
    (structured / "a.parquet.meta.json").unlink()

    result = run_ingest_one(cfg=cfg, patient_id="P1")

    assert result["cached"] is False
    assert (structured / "a.parquet.meta.json").is_file()  # rewritten on re-ingest
