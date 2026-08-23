"""A stage command runs the whole cohort; --patient narrows it to one.

The cohort is the unit of work, so that is the default. One patient is the special
case — it exists because a SLURM array task is handed exactly one, and those scripts
pass --patient explicitly, which is why making the flag optional is safe.

The two properties that matter beyond convenience: which patients a stage picks up
(the source tree for ingest, the run for everything after it, since a patient that
failed ingest has nothing to embed), and that one bad patient does not strand the
rest of the cohort while still failing the command.
"""
from __future__ import annotations

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main, patients_for_stage

REPO = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path, source_root: Path) -> Path:
    path = tmp_path / "junior.yaml"
    path.write_text(
        f"run_id: R1\nsource_root: {source_root}\noutput_root: {tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    return path


def _stub_pipeline(monkeypatch, seen: list[str]):
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface._ensure_sealed_bundle",
        lambda **_: "sha256:" + "0" * 64,
    )

    def _record(cfg, patient_id, code_lock_hash, force):
        seen.append(patient_id)
        if patient_id == "bad":
            raise RuntimeError("this chart is broken")
        return {"patient_id": patient_id}

    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.STAGES", {"index": _record}
    )


def test_a_stage_runs_every_patient_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.patients_for_stage",
        lambda stage, cfg, run_root: ["P1", "P2", "P3"],
    )
    seen: list[str] = []
    _stub_pipeline(monkeypatch, seen)

    result = CliRunner().invoke(
        main, ["index", "--config", str(_config(tmp_path, tmp_path))]
    )
    assert result.exit_code == 0, result.output
    assert seen == ["P1", "P2", "P3"]


def test_patient_narrows_to_one(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.patients_for_stage",
        lambda stage, cfg, run_root: ["P1", "P2", "P3"],
    )
    seen: list[str] = []
    _stub_pipeline(monkeypatch, seen)

    result = CliRunner().invoke(
        main, ["index", "--config", str(_config(tmp_path, tmp_path)), "--patient", "P2"]
    )
    assert result.exit_code == 0, result.output
    assert seen == ["P2"], "--patient did not narrow the cohort"


def test_one_bad_patient_does_not_strand_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.patients_for_stage",
        lambda stage, cfg, run_root: ["P1", "bad", "P3"],
    )
    seen: list[str] = []
    _stub_pipeline(monkeypatch, seen)

    result = CliRunner().invoke(
        main, ["index", "--config", str(_config(tmp_path, tmp_path))]
    )
    assert seen == ["P1", "bad", "P3"], "the cohort stopped at the failing patient"
    assert result.exit_code != 0, "a failed patient was reported as success"
    assert "bad" in result.output


# --- which patients a stage picks up ---------------------------------------------


def test_ingest_reads_the_source_tree_and_screens_it(tmp_path):
    """A directory of directories sitting beside the patients is not a patient.
    Reporting it as a failure every run trains the operator to ignore failures."""
    source_root = tmp_path / "src"
    (source_root / "P1").mkdir(parents=True)
    (source_root / "P1" / "clinical_note.csv").write_text(
        "patient_id,text\nP1,hello\n", encoding="utf-8"
    )
    (source_root / "not_a_patient" / "nested").mkdir(parents=True)

    cfg = {"run_id": "R1", "source_root": str(source_root), "project": "t"}
    picked = patients_for_stage("ingest", cfg, tmp_path / "run")
    assert "P1" in picked
    assert "not_a_patient" not in picked, "an export bundle was treated as a patient"


def test_later_stages_read_the_run_not_the_source_tree(tmp_path):
    """A patient that failed ingest has nothing to embed, so the cohort for every
    stage after ingest is whoever actually made it through."""
    run_root = tmp_path / "run"
    for patient in ("P1", "P3"):
        (run_root / "patients" / patient).mkdir(parents=True)
    source_root = tmp_path / "src"
    for patient in ("P1", "P2", "P3"):
        (source_root / patient).mkdir(parents=True)

    cfg = {"run_id": "R1", "source_root": str(source_root)}
    assert patients_for_stage("embed", cfg, run_root) == ["P1", "P3"]


def test_a_later_stage_before_any_ingest_says_so(tmp_path):
    cfg = {"run_id": "R1", "source_root": str(tmp_path)}
    with pytest.raises(click.ClickException, match="ingest"):
        patients_for_stage("embed", cfg, tmp_path / "never_ingested")
