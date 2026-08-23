"""One shared, case-insensitive source-file discovery.

Ingest, file resolution, and the source snapshot all share one discovery rule, so an
uppercase-extension file (``NOTES.CSV`` → stem ``NOTES``) is handled identically by
each. Without this, the snapshot could miss the file on macOS (a later edit never busts
the cache) and resolution could fail on case-sensitive Linux (FileNotFoundError
mid-run).

These tests pin the shared rule: ``discover_source_files`` matches case-insensitively
and returns the real on-disk path, and the snapshot — which consumes the same discovery
— records and re-checks uppercase-extension files.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files import source_format_readers
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.per_patient_source_snapshot import (
    read_source_snapshot,
    snapshot_matches,
    write_source_snapshot,
)
from jr_pipeline.runtime_infrastructure.source_file_discovery import (
    SUPPORTED_SOURCE_EXTENSIONS,
    discover_source_files,
    is_supported_source_file,
)


def _make_patient(directory, **files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content)


# --- discover_source_files ---

def test_discover_maps_stem_to_real_on_disk_path(tmp_path):
    pdir = tmp_path / "p"
    _make_patient(pdir, **{"labs.csv": "a,b\n1,2\n", "notes.tsv": "a\tb\n3\t4\n"})
    found = discover_source_files(pdir)
    assert set(found) == {"labs", "notes"}
    assert found["labs"].name == "labs.csv"
    assert found["labs"].read_text() == "a,b\n1,2\n"


def test_discover_matches_extension_case_insensitively_and_preserves_case(tmp_path):
    pdir = tmp_path / "p"
    _make_patient(pdir, **{"NOTES.CSV": "a,b\n1,2\n"})
    found = discover_source_files(pdir)
    assert set(found) == {"NOTES"}
    # The returned path is the real directory entry — original name and case —
    # so a caller on a case-sensitive filesystem opens the file that exists.
    assert found["NOTES"].name == "NOTES.CSV"


def test_discover_skips_hidden_and_unsupported(tmp_path):
    pdir = tmp_path / "p"
    _make_patient(
        pdir,
        **{"keep.csv": "a\n1\n", ".hidden.csv": "a\n1\n", "readme.txt": "hi"},
    )
    assert set(discover_source_files(pdir)) == {"keep"}


def test_discover_extension_priority_csv_beats_parquet(tmp_path):
    # When one stem has several supported files, the earliest extension in
    # SUPPORTED_SOURCE_EXTENSIONS wins — a fresh csv must not lose to a stale
    # parquet copy left in the folder.
    pdir = tmp_path / "p"
    pdir.mkdir(parents=True)
    (pdir / "notes.csv").write_text("a\n1\n")
    (pdir / "notes.parquet").write_bytes(b"not really parquet")
    found = discover_source_files(pdir)
    assert set(found) == {"notes"}
    assert found["notes"].suffix == ".csv"


def test_is_supported_source_file_directory_is_not_a_source(tmp_path):
    sub = tmp_path / "sub.csv"  # a directory whose name ends in .csv
    sub.mkdir()
    assert is_supported_source_file(sub) is False


# --- snapshot consumes the shared discovery ---

def test_snapshot_records_uppercase_extension_file(tmp_path):
    # An uppercase-extension source file is recorded in the snapshot's file list.
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"NOTES.CSV": "a,b\n1,2\n"})
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    stored = read_source_snapshot(out)
    relpaths = [f["relpath"] for f in stored["payload"]["files"]]
    assert relpaths == ["NOTES.CSV"]


def test_snapshot_matches_false_when_uppercase_file_content_changes(tmp_path):
    # An edit to an uppercase-extension source file must bust the cache: if discovery
    # missed the file, snapshot_matches would stay True after the edit and serve a
    # stale cache.
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"NOTES.CSV": "a,b\n1,2\n"})
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    assert snapshot_matches(out, pdir) is True
    (pdir / "NOTES.CSV").write_text("a,b\n9,9\n")
    assert snapshot_matches(out, pdir) is False


# --- the discovery/reader-dispatch agreement invariant (mirrors ingest's import assert) ---

def test_reader_dispatch_covers_exactly_the_supported_extensions():
    assert set(source_format_readers._FORMAT_READERS) == set(SUPPORTED_SOURCE_EXTENSIONS)
