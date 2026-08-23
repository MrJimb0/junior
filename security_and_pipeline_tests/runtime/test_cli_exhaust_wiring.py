"""The CLI's exhaust wiring: ``summarize`` finalizes the run's NO_PHI shards,
``collect-feedback`` re-finalizes so review-session relevance labels land in the
parquet, and ``export-metadata`` ships the egress-scanned bundle.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_method_provenance,
    record_relevance_label,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
    build_manifest,
    write_manifest,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
    no_phi_exhaust_dir,
    no_phi_manifest_path,
    no_phi_run_dir,
    phi_intermediate_run_dir,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle
from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust

RUN = "20260101_000000"
CODE_HASH = "sha256:" + "0" * 64


def setup_function(_):
    secret_lifecycle._CACHE.clear()


@pytest.fixture(autouse=True)
def _no_ambient_project(tmp_path, monkeypatch):
    """Run each of these from a directory that is not inside anybody's project.

    `export-metadata` resolves which cohort it is packaging the same way every other
    command does, by walking upward for a settings file. A stray one above the checkout
    — a `junior.yaml` in a home directory captures everything below it — would
    otherwise decide which run these tests exported."""
    monkeypatch.chdir(tmp_path)
    for name in ("JUNIOR_CONFIG", "JR_RUN_ID"):
        monkeypatch.delenv(name, raising=False)


def _seed_run(run_root: Path) -> None:
    """A minimal but real run: manifest + one completed step transition."""
    run_root.mkdir(parents=True, exist_ok=True)
    write_manifest(run_root, build_manifest(
        run_id=RUN, code_lock_hash=CODE_HASH,
        entry_point_name="test", config_alias="test", target_patients=["p1"],
    ))
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=RUN, patient_id="p1", step="ingest"),
        from_state="running", to_state="completed", reason="test",
        step_context="ingest", code_lock_hash=CODE_HASH,
    )


def test_summarize_finalizes_exhaust_from_run_root(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    run_root = phi_intermediate_run_dir(RUN)
    _seed_run(run_root)
    record_method_provenance(RUN)

    # Ambient data root points elsewhere: summarize must derive the run's data root
    # from --run-root (the canonical receipts layout), not from the environment.
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "elsewhere"))
    res = CliRunner().invoke(main, ["summarize", "--run-root", str(run_root)])
    assert res.exit_code == 0, res.output
    assert "method_provenance" in res.output
    assert no_phi_manifest_path(RUN, tmp_path).is_file()
    assert (no_phi_exhaust_dir(RUN, tmp_path) / "method_provenance.parquet").is_file()


def test_collect_feedback_refinalizes_review_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    record_method_provenance(RUN)
    finalize_exhaust(RUN)  # run-end finalize, BEFORE the review session

    # A review session: the PHI-side chunk_relevance entry + its NO_PHI twin
    # (exactly what the review app writes per judgment).
    fb = clinician_feedback_dir(RUN)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "p1__v.json").write_text(json.dumps({
        "patient_id": "p1", "variable": "v", "run_id": RUN, "annotator": "r1",
        "feedback": [{"type": "chunk_relevance", "chunk_id": "p1:notes:0:0", "relevant": True}],
    }), encoding="utf-8")
    record_relevance_label(
        run_id=RUN, patient_id="p1", reviewer_id="r1",
        chunk_id="p1:notes:0:0", label="relevant", recipe_id="v", step_id="unknown",
    )

    res = CliRunner().invoke(
        main, ["collect-feedback", "--run-id", RUN, "--output-dir", str(tmp_path / "export")]
    )
    assert res.exit_code == 0, res.output
    assert (no_phi_exhaust_dir(RUN) / "relevance_label.parquet").is_file(), \
        "collect-feedback must fold the review-session shard into the parquet"
    manifest = json.loads(no_phi_manifest_path(RUN).read_text(encoding="utf-8"))
    assert manifest["record_types"]["relevance_label"]["n_rows"] == 1


def test_collect_feedback_demo_run_creates_no_no_phi_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    demo_run = "shiny_demo_20260101"
    fb = clinician_feedback_dir(demo_run)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "p1__v.json").write_text(json.dumps({
        "patient_id": "p1", "variable": "v", "run_id": demo_run, "annotator": "r1",
        "feedback": [{"type": "extraction_correction", "correct_value": "1957-04-15"}],
    }), encoding="utf-8")

    res = CliRunner().invoke(
        main, ["collect-feedback", "--run-id", demo_run, "--output-dir", str(tmp_path / "export")]
    )
    assert res.exit_code == 0, res.output
    assert not no_phi_run_dir(demo_run).exists(), \
        "a synthetic demo run must not get a NO_PHI tree manufactured"


def test_export_metadata_cli_finalizes_then_ships_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    record_method_provenance(RUN)  # shards only — the command must finalize first

    out = tmp_path / "bundle.zip"
    res = CliRunner().invoke(main, ["export-metadata", "--run-id", RUN, "--output", str(out)])
    assert res.exit_code == 0, res.output
    names = zipfile.ZipFile(out).namelist()
    assert any(n.endswith("manifest.json") for n in names)
    assert any(n.endswith("method_provenance.parquet") for n in names)


def test_export_metadata_unknown_run_fails_clean(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    bogus = "20990101_000000"
    res = CliRunner().invoke(
        main, ["export-metadata", "--run-id", bogus, "--output", str(tmp_path / "b.zip")]
    )
    assert res.exit_code != 0
    assert not no_phi_run_dir(bogus).exists()
    assert not (tmp_path / "b.zip").exists()
