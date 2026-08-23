"""Pin ingest date canonicalization + the snapshot predicate.

  * the century-refusal policy — a 2-digit slash-date year raises rather than
    guessing the century; and
  * snapshot_matches() is a content-hash equality over the patient's source files
    (False when unchanged-snapshot is absent, a file's content changes, or a file
    is added/removed).
"""
from __future__ import annotations

import polars as pl
import pytest

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.date_canonicalization import (
    AmbiguousYearError,
    _canonicalize_dates,
    _try_iso,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.per_patient_source_snapshot import (
    read_source_snapshot,
    snapshot_matches,
    write_source_snapshot,
)

# --- date canonicalization / century refusal ---

def test_try_iso_four_digit_slash_date_canonicalizes():
    assert _try_iso("4/15/1957") == "1957-04-15"


def test_try_iso_iso_date_passes_through():
    assert _try_iso("1957-04-15") == "1957-04-15"


def test_try_iso_two_digit_year_refuses_to_guess_century():
    with pytest.raises(AmbiguousYearError):
        _try_iso("4/15/57")


def test_try_iso_non_date_returns_none():
    assert _try_iso("not a date") is None


def test_canonicalize_dates_four_digit_column():
    df = pl.DataFrame({"dx_date": ["1/2/2024", "3/4/2023", "5/6/2022"]})
    out = _canonicalize_dates(df)
    assert out["dx_date"].to_list() == ["2024-01-02", "2023-03-04", "2022-05-06"]


def test_canonicalize_dates_two_digit_year_column_raises():
    df = pl.DataFrame({"dx_date": ["1/2/45", "3/4/99", "5/6/88"]})
    with pytest.raises(AmbiguousYearError):
        _canonicalize_dates(df)


# --- snapshot cache predicate ---

def _make_patient(directory, **files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (directory / name).write_text(content)


def test_snapshot_matches_true_when_unchanged(tmp_path):
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"a.csv": "x,y\n1,2\n", "b.csv": "p,q\n3,4\n"})
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    assert snapshot_matches(out, pdir) is True


def test_snapshot_matches_false_when_file_content_changes(tmp_path):
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"a.csv": "x,y\n1,2\n"})
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    (pdir / "a.csv").write_text("x,y\n9,9\n")
    assert snapshot_matches(out, pdir) is False


def test_snapshot_matches_false_when_no_snapshot(tmp_path):
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"a.csv": "x,y\n1,2\n"})
    assert snapshot_matches(out, pdir) is False


def test_snapshot_matches_false_when_file_added(tmp_path):
    pdir, out = tmp_path / "p", tmp_path / "out"
    _make_patient(pdir, **{"a.csv": "x,y\n1,2\n"})
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    (pdir / "c.csv").write_text("m,n\n5,6\n")
    assert snapshot_matches(out, pdir) is False


# --- vectorized date canonicalization ---

def test_canonicalize_preserves_time_bearing_values():
    df = pl.DataFrame({"ts": ["4/15/1957 09:05:07", "1957-04-15T13:30", "1957-04-15"]})
    out = _canonicalize_dates(df)
    assert out["ts"].to_list() == [
        "1957-04-15T09:05:07+00:00",
        "1957-04-15T13:30:00+00:00",
        "1957-04-15",
    ]


def test_canonicalize_passes_through_non_dates_in_date_column():
    # >=60% date-shaped -> treated as a date column; the stray non-date stays.
    df = pl.DataFrame({"dx": ["1/2/2024", "3/4/2023", "5/6/2022", "pending"]})
    out = _canonicalize_dates(df)
    assert out["dx"].to_list() == ["2024-01-02", "2023-03-04", "2022-05-06", "pending"]


def test_canonicalize_is_idempotent():
    df = pl.DataFrame({"d": ["1/2/2024", "3/4/2023"]})
    once = _canonicalize_dates(df)
    twice = _canonicalize_dates(once)
    assert twice["d"].to_list() == once["d"].to_list() == ["2024-01-02", "2023-03-04"]


def test_canonicalize_refuses_two_digit_year_past_sample_window():
    # is_date is decided on the sampled head (first 100 rows); the 2-digit
    # offender sits past it, so only a whole-column scan catches it.
    rows = ["1/2/2024"] * 200 + ["3/4/55"]
    df = pl.DataFrame({"d": rows})
    with pytest.raises(AmbiguousYearError):
        _canonicalize_dates(df)


def test_preflight_catches_two_digit_year_past_sample_window(tmp_path):
    # preflight scans the whole column, like _canonicalize_dates. A clean-headed
    # column with a 2-digit-year offender past row 100 must be flagged by preflight
    # (report.blocked) rather than passing and then crashing run_ingest_one with
    # AmbiguousYearError.
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    pdir = tmp_path / "P1"
    pdir.mkdir(parents=True)
    rows = ["1/2/2024"] * 110 + ["3/4/55"]
    pl.DataFrame({"row_id": list(range(len(rows))), "dx_date": rows}).write_csv(pdir / "diagnoses.csv")

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto"}, patients=["P1"]
    )
    assert ("P1", "ambiguous_year") in {(i.patient_id, i.kind) for i in report.issues}, report.summary()
    assert "P1" in report.blocked and "P1" not in report.ok


def test_preflight_passes_clean_four_digit_column(tmp_path):
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    pdir = tmp_path / "P2"
    pdir.mkdir(parents=True)
    pl.DataFrame({
        "row_id": [1, 2, 3],
        "dx_date": ["1/2/2024", "3/4/2023", "5/6/2022"],
        "text": ["a", "b", "c"],       # something embeddable, else preflight blocks
    }).write_csv(pdir / "diagnoses.csv")

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto"}, patients=["P2"]
    )
    # This toy file has a text column, so it is runnable; it carries none of the chart
    # metadata, which preflight reports as advisories that never block a patient.
    assert "P2" in report.ok
    assert [i for i in report.issues if i.blocking] == []
    assert {i.kind for i in report.advisories} == {"metadata_column_not_found"}


# --- snapshot row counts are quoted-newline safe (scan_csv) ---

def test_snapshot_row_count_handles_quoted_newline(tmp_path):
    pdir, out = tmp_path / "p", tmp_path / "out"
    pdir.mkdir(parents=True)
    # 2 logical rows; the first has a quoted embedded newline (3 physical lines).
    (pdir / "notes.csv").write_text('id,text\n1,"line one\nline two"\n2,plain\n')
    write_source_snapshot(run_id="r", patient_id="p", patient_dir=pdir, patient_out_dir=out)
    stored = read_source_snapshot(out)
    row_counts = {f["relpath"]: f["row_count"] for f in stored["payload"]["files"]}
    assert row_counts["notes.csv"] == 2


def test_preflight_reports_one_metadata_advisory_per_file_not_per_field(tmp_path):
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    # Seven fields x seven files was 49 advisory lines for a single patient; a cohort
    # scan multiplied that by every patient and buried the blockers.
    pdir = tmp_path / "P3"
    pdir.mkdir(parents=True)
    pl.DataFrame({"row_id": [1], "text": ["note body"]}).write_csv(pdir / "clinical_note.csv")
    pl.DataFrame({"row_id": [1], "text": ["path body"]}).write_csv(pdir / "pathology_report.csv")

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto"}, patients=["P3"]
    )

    metadata = [i for i in report.advisories if i.kind == "metadata_column_not_found"]
    assert len(metadata) == 2                       # one per file
    assert {i.file for i in metadata} == {"clinical_note.csv", "pathology_report.csv"}
    assert "author" in metadata[0].detail            # the fields are named in the one line
    assert "P3" in report.ok                         # and none of it blocks the patient


def test_preflight_flags_columns_that_differ_only_by_case(tmp_path):
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    # Only one of them can ever be read, so the operator has to be told which.
    pdir = tmp_path / "P4"
    pdir.mkdir(parents=True)
    (pdir / "clinical_note.csv").write_text("Author,AUTHOR,text\na,b,note body\n")

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto"}, patients=["P4"]
    )

    ambiguous = [i for i in report.advisories if i.kind == "ambiguous_columns"]
    assert len(ambiguous) == 1
    assert "Author" in ambiguous[0].detail and "AUTHOR" in ambiguous[0].detail
    assert "P4" in report.ok


def test_preflight_blocks_a_patient_with_no_free_text_anywhere(tmp_path):
    # The failure a new site hits first: their notes column is not called "text", so
    # nothing is embeddable. Caught before a run, not after a whole cohort is ingested.
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    pdir = tmp_path / "PT1"
    pdir.mkdir(parents=True)
    (pdir / "progress_notes.csv").write_text("MRN,NOTE_DATE,NOTE_TEXT\nX1,2024-03-01,a note\n")

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto"}, patients=["PT1"]
    )

    assert "PT1" in report.blocked
    blocker = next(i for i in report.issues if i.kind == "no_text_to_embed")
    assert "junior columns" in blocker.detail            # tells them how to fix it


def test_a_site_map_naming_the_text_column_clears_the_blocker(tmp_path):
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    pdir = tmp_path / "PT1"
    pdir.mkdir(parents=True)
    (pdir / "progress_notes.csv").write_text("MRN,NOTE_DATE,NOTE_TEXT\nX1,2024-03-01,a note\n")
    site_map = tmp_path / "site.yaml"
    site_map.write_text(
        "chunk_metadata_columns:\n"
        "  progress_notes:\n"
        "    text_columns: [NOTE_TEXT]\n"
        "    document_date: NOTE_DATE\n"
    )

    report = preflight_patients(
        cfg={"run_id": "t", "source_root": str(tmp_path), "files": "auto",
             "chart_columns_file": str(site_map)},
        patients=["PT1"],
    )

    assert "PT1" in report.ok
