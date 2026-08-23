"""The review app's run/patient picker (run_results) over a fabricated receipts tree.

These exercise the *selection* logic the Review picker depends on — which runs are
offered, which patients under a run, newest-first ordering — without the encoder or
any model. The receipts tree is built with the same path helpers run_results reads
through, so the fabrication and the loader can never drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_results

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
    pipeline_run_receipts_root,
)

# A minimal, valid per-variable artifact envelope: one flat scalar the loader
# renders directly. No evidence chunk ids, so no PatientChunkStore is needed.
_DOB_ENVELOPE = {
    "payload": {
        "variable": "date_of_birth",
        "ok": True,
        "data": {"date_of_birth": {"type": "string", "value": "1970-01-01"}},
        "recipe": {"name": "demographics", "version": "1"},
    }
}


def _write_extract(dr: Path, run_id: str, patient_id: str,
                   variable: str = "date_of_birth", envelope: dict | str | None = None) -> None:
    """Fabricate ``<receipts>/<run>/patients/<pid>/extract/<var>/result.json``."""
    var_dir = extract_output_dir(phi_patient_run_dir(run_id, patient_id, dr)) / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    body = _DOB_ENVELOPE if envelope is None else envelope
    text = body if isinstance(body, str) else json.dumps(body)
    (var_dir / "result.json").write_text(text, encoding="utf-8")


def _make_patient_without_extract(dr: Path, run_id: str, patient_id: str) -> None:
    """A patient folder that exists but produced no extract output."""
    phi_patient_run_dir(run_id, patient_id, dr).mkdir(parents=True, exist_ok=True)


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Point run_results at an empty tmp data root for the duration of one test."""
    monkeypatch.setattr(run_results, "DATA_ROOT", tmp_path)
    return tmp_path


# ── list_reviewable_runs ─────────────────────────────────────────────────────

def test_list_reviewable_runs_newest_first_and_excludes_runs_without_extract(data_root):
    _write_extract(data_root, "20260101_010101_aa", "Patient_A")
    _write_extract(data_root, "20260102_020202_bb", "Patient_A")
    # A canonical run whose only patient produced no extract → not reviewable.
    _make_patient_without_extract(data_root, "20260103_030303_cc", "Patient_A")
    # A non-canonical directory (e.g. a compare-scratch dir) → never offered.
    (pipeline_run_receipts_root(data_root) / "not_a_run_dir").mkdir(parents=True)

    assert run_results.list_reviewable_runs() == [
        "20260102_020202_bb",
        "20260101_010101_aa",
    ]


def test_list_reviewable_runs_empty_when_no_receipts_root(data_root):
    assert run_results.list_reviewable_runs() == []


# ── list_patients_with_extract ───────────────────────────────────────────────

def test_list_patients_with_extract_sorted_and_only_those_with_output(data_root):
    run_id = "20260101_010101_aa"
    _write_extract(data_root, run_id, "Patient_B")
    _write_extract(data_root, run_id, "Patient_A")
    _make_patient_without_extract(data_root, run_id, "Patient_C")

    assert run_results.list_patients_with_extract(run_id) == ["Patient_A", "Patient_B"]


def test_list_patients_with_extract_unknown_run_is_empty(data_root):
    assert run_results.list_patients_with_extract("20260101_010101_aa") == []


def test_picker_rejects_a_non_canonical_run_id_even_if_the_dir_exists(data_root):
    # A spoofed picker input must not walk out of the receipts root: a run id that is not
    # the canonical YYYYMMDD_HHMMSS_<hex> shape is rejected before any path join, even when
    # a directory by that name exists on disk with extract output.
    _write_extract(data_root, "sneaky_dir", "Patient_A")

    assert run_results.list_patients_with_extract("sneaky_dir") == []
    assert run_results.load_run_variables("sneaky_dir", "Patient_A") == []


# ── find_latest_run_id (auto-selected default) ───────────────────────────────

def test_find_latest_run_id_skips_a_newer_run_without_extract(data_root):
    _write_extract(data_root, "20260101_010101_aa", "Patient_A")
    _write_extract(data_root, "20260102_020202_bb", "Patient_A")
    # Newest by id, but no extract output → must not be auto-selected.
    _make_patient_without_extract(data_root, "20260103_030303_cc", "Patient_A")

    assert run_results.find_latest_run_id() == "20260102_020202_bb"


# ── load_run_variables ───────────────────────────────────────────────────────

def test_load_run_variables_reads_a_flat_scalar_envelope(data_root):
    _write_extract(data_root, "20260101_010101_aa", "Patient_A")

    rows = run_results.load_run_variables("20260101_010101_aa", "Patient_A")

    assert len(rows) == 1
    row = rows[0]
    assert row.variable == "date_of_birth"
    assert "1970-01-01" in row.value
    assert row.ok is True
    assert row.evidence == []  # no evidence pointer in the envelope


def test_load_run_variables_unknown_patient_is_empty(data_root):
    assert run_results.load_run_variables("20260101_010101_aa", "Patient_A") == []


def test_load_run_variables_skips_a_malformed_result_json(data_root):
    run_id, pid = "20260101_010101_aa", "Patient_A"
    _write_extract(data_root, run_id, pid, variable="date_of_birth")
    # A second variable whose result.json is not valid JSON must be skipped, not fatal.
    _write_extract(data_root, run_id, pid, variable="broken", envelope="{ not json")

    rows = run_results.load_run_variables(run_id, pid)

    assert [r.variable for r in rows] == ["date_of_birth"]


# ── resolve_run env overrides ────────────────────────────────────────────────

def test_resolve_run_honors_env_run_and_patient(data_root, monkeypatch):
    _write_extract(data_root, "20260101_010101_aa", "Patient_A")
    _write_extract(data_root, "20260102_020202_bb", "Patient_A")
    monkeypatch.setenv("JR_REVIEW_RUN_ID", "20260101_010101_aa")
    monkeypatch.setenv("JR_REVIEW_PATIENT_ID", "Patient_A")

    resolved = run_results.resolve_run()

    assert resolved is not None
    assert resolved.run_id == "20260101_010101_aa"  # env pick, not the newest
    assert resolved.patient_id == "Patient_A"
