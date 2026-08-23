"""Start tab Step-1 operations (cohort_ingest) over the bundled example folder.

scan/preflight/ingest are pure Step 1 — polars only, no encoder, no model — so
they run in a fraction of a second against ``examples/``. The example cohort is
the app's own default input (``default_input_folder``): one patient folder
(``Test_Patient``) with its source CSVs directly inside.
"""
from __future__ import annotations

import cohort_ingest

from jr_pipeline.runtime_infrastructure.cohort_runner import RUN_ID_PATTERN
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    pipeline_run_receipts_root,
)

EXAMPLES = cohort_ingest.default_input_folder()


# ── scan_cohort ──────────────────────────────────────────────────────────────

def test_scan_cohort_lists_the_bundled_patient_folder():
    result = cohort_ingest.scan_cohort(EXAMPLES)

    assert result.error is None
    assert [p.patient_id for p in result.patients] == ["Test_Patient"]
    assert result.patients[0].file_count >= 5  # 7 source CSVs sit directly inside


def test_scan_cohort_flags_a_nested_layout_as_an_empty_dir(tmp_path):
    # A project layout that keeps its files one level down reads as an "empty" top
    # folder — reported as a hint to point deeper, never silently dropped.
    patient = tmp_path / "Patient_A"
    patient.mkdir()
    (patient / "clinical_note.csv").write_text("note_text\nhello\n", encoding="utf-8")
    nested = tmp_path / "SomeProject" / "ready" / "Patient_B"
    nested.mkdir(parents=True)
    (nested / "clinical_note.csv").write_text("note_text\nhi\n", encoding="utf-8")

    result = cohort_ingest.scan_cohort(tmp_path)

    assert [p.patient_id for p in result.patients] == ["Patient_A"]
    assert "SomeProject" in result.empty_dirs


def test_scan_cohort_on_a_single_patient_folder_points_at_the_parent():
    # A common mis-point: selecting one patient's own folder. Its files sit directly
    # inside, not in subfolders — the scan must say so, not report a bare "found nothing".
    result = cohort_ingest.scan_cohort(EXAMPLES / "Test_Patient")

    assert result.patients == []
    assert result.error is not None
    assert "PARENT" in result.error
    assert "Test_Patient" in result.error


def test_scan_cohort_missing_folder_reports_error(tmp_path):
    result = cohort_ingest.scan_cohort(tmp_path / "does_not_exist")

    assert result.patients == []
    assert result.error is not None
    assert "Not a folder" in result.error


# ── preflight_cohort ─────────────────────────────────────────────────────────

def test_preflight_cohort_passes_the_example_patient():
    rows = cohort_ingest.preflight_cohort(EXAMPLES, ["Test_Patient"])

    assert len(rows) == 1
    assert rows[0].patient_id == "Test_Patient"
    assert rows[0].ok is True
    assert rows[0].problems == []


def test_preflight_cohort_blocks_a_patient_that_is_not_there():
    rows = cohort_ingest.preflight_cohort(EXAMPLES, ["No_Such_Patient"])

    assert len(rows) == 1
    assert rows[0].ok is False
    assert rows[0].problems  # at least one concrete reason


def test_preflight_cohort_empty_selection_returns_no_rows():
    assert cohort_ingest.preflight_cohort(EXAMPLES, []) == []


# ── ingest_cohort ────────────────────────────────────────────────────────────

def test_ingest_cohort_writes_structured_output_under_the_data_root(tmp_path, monkeypatch):
    # Route Step-1 output into a throwaway data root so the test never writes into
    # the repo's data/ tree. run_ingest_one resolves JR_DATA_ROOT at call time.
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))

    outcome = cohort_ingest.ingest_cohort(EXAMPLES, ["Test_Patient"])

    assert outcome.error is None
    assert RUN_ID_PATTERN.fullmatch(outcome.run_id)
    assert len(outcome.rows) == 1
    row = outcome.rows[0]
    assert row.patient_id == "Test_Patient"
    assert row.ok is True
    assert row.files_written >= 1
    assert row.total_rows >= 1
    # The output landed under the tmp data root, not the repo.
    assert (pipeline_run_receipts_root(tmp_path) / outcome.run_id).is_dir()


def test_ingest_cohort_empty_selection_is_an_error_not_a_crash():
    outcome = cohort_ingest.ingest_cohort(EXAMPLES, [])

    assert outcome.run_id == ""
    assert outcome.rows == []
    assert outcome.error is not None
