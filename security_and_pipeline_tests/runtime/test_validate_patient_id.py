"""Patient ids are validated as a safe single path segment BEFORE any path
construction, so a '/', '..', ':' or absolute-shaped id cannot escape the
per-patient dir or collide with another patient."""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    no_phi_run_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    validate_patient_id,
)


@pytest.mark.parametrize(
    "good", ["Test_Patient", "p1", "P1", "MRN-123.4", "Deceased_Synth_Patient"]
)
def test_accepts_safe_ids(good):
    assert validate_patient_id(good) == good


@pytest.mark.parametrize(
    "bad",
    ["../x", "a/b", "..", ".", "", "a:b", "a b", "a\x00b", "/etc/passwd", "~root", "a\\b"],
)
def test_rejects_unsafe_ids(bad):
    with pytest.raises(ValueError):
        validate_patient_id(bad)


def test_path_helper_rejects_traversal(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    with pytest.raises(ValueError):
        phi_patient_run_dir("R", "../evil")
    # a safe id still resolves under the run's patients/ dir
    p = phi_patient_run_dir("R", "P1")
    assert p.name == "P1" and p.parent.name == "patients"


@pytest.mark.parametrize(
    "bad",
    ["../ESCAPED", "../../../../ESCAPED", "/abs/escape", "a/b", "..", ".", "", "a b",
     "run\n", "..\n"],
)
def test_run_dirs_reject_an_id_that_would_leave_the_data_root(bad, tmp_path, monkeypatch):
    """The run id reaches these helpers from --run-id, from JR_RUN_ID (which the SLURM
    scripts export) and from a config file. validate_run_id existed for this and was
    wired into one caller that builds a filename, while the two helpers that build the
    run's directories took it raw — so an absolute or traversing id relocated the whole
    run, every patient parquet and receipt included, outside CONTAINS_PHI. Nothing
    raised, and check_phi_containment only walks what is inside the tree, so it could
    not see what had been written outside it."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    for helper in (phi_intermediate_run_dir, no_phi_run_dir):
        with pytest.raises(ValueError):
            helper(bad)


def test_a_safe_run_id_still_resolves_under_each_root(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    assert phi_intermediate_run_dir("20260101_000000_ab").parent.name == "pipeline_run_receipts"
    assert no_phi_run_dir("20260101_000000_ab").name == "20260101_000000_ab"
