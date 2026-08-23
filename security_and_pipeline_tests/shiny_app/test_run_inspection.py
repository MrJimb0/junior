"""The Workbench tab's readers (run_inspection) over a fabricated run.

Everything is fabricated through the same path helpers the readers read with, so
the fixtures and the readers cannot drift apart. The export test records the
arguments handed to the pipeline's own export function rather than exercising the
scan — the CLI's export path owns that behavior; what the app owns is calling the
identical function on the identical artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import run_inspection

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    evidence_selection_metadata_dir,
    extract_output_dir,
    no_phi_manifest_path,
    no_phi_run_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    prepared_evidence_text_dir,
)

RUN = "20260101_010101_aa"
PATIENT = "Patient_A"
VARIABLE = "date_of_birth"


@pytest.fixture
def dr(tmp_path) -> Path:
    return tmp_path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── the run id gate ──────────────────────────────────────────────────────────

def test_a_non_canonical_run_id_reads_nothing_even_if_the_dir_exists(dr):
    sneaky = "not_a_run"
    _write(phi_patient_run_dir(RUN, PATIENT, dr) / "x.json", "{}")
    (dr / "CONTAINS_PHI" / "pipeline_run_receipts" / sneaky).mkdir(parents=True)

    assert run_inspection.list_patient_files(sneaky, PATIENT, dr) == []
    assert run_inspection.read_prepared_evidence(sneaky, PATIENT, VARIABLE, dr) == []
    assert run_inspection.read_run_summary(sneaky, dr) is None
    assert run_inspection.read_exhaust_manifest(sneaky, dr) is None
    with pytest.raises(ValueError, match="not a run id"):
        run_inspection.export_shareable_zip(sneaky, dr)


# ── files ────────────────────────────────────────────────────────────────────

def test_list_patient_files_reports_relative_paths_and_sizes(dr):
    _write(phi_patient_run_dir(RUN, PATIENT, dr) / "extract" / VARIABLE / "result.json", "{}")

    files = run_inspection.list_patient_files(RUN, PATIENT, dr)

    assert [f.rel_path for f in files] == [f"extract/{VARIABLE}/result.json"]
    assert files[0].size_bytes == 2


# ── prepared evidence + selection metadata ───────────────────────────────────

def test_read_prepared_evidence_returns_one_bundle_per_step(dr):
    base = prepared_evidence_text_dir(RUN, PATIENT, dr) / VARIABLE
    _write(base / "step_a" / "formatted_evidence.txt", "what the model saw")
    _write(base / "step_b" / "formatted_evidence.txt", "second pass")

    bundles = run_inspection.read_prepared_evidence(RUN, PATIENT, VARIABLE, dr)

    assert [(b.step_id, b.text) for b in bundles] == [
        ("step_a", "what the model saw"), ("step_b", "second pass"),
    ]


def test_a_huge_evidence_bundle_is_clipped_with_a_pointer_to_the_file(dr):
    base = prepared_evidence_text_dir(RUN, PATIENT, dr) / VARIABLE
    _write(base / "step_a" / "formatted_evidence.txt", "x" * (run_inspection.MAX_TEXT_CHARS + 500))

    bundle = run_inspection.read_prepared_evidence(RUN, PATIENT, VARIABLE, dr)[0]

    assert len(bundle.text) < run_inspection.MAX_TEXT_CHARS + 200
    assert "clipped" in bundle.text
    assert "500" in bundle.text  # says how much is left in the file


def test_read_evidence_selection_reports_the_bundle_sizes(dr):
    base = evidence_selection_metadata_dir(RUN, PATIENT, dr) / VARIABLE
    _write(base / "step_a" / "evidence_selection.json", json.dumps({
        "block_count": 4, "total_evidence_tokens": 900, "max_context_tokens": 3000,
        "evidence_tokens_by_doc_type": {"clinical_note": 700, "pathology_report": 200},
    }))

    summary = run_inspection.read_evidence_selection(RUN, PATIENT, VARIABLE, dr)[0]

    assert summary.step_id == "step_a"
    assert summary.block_count == 4
    assert summary.evidence_tokens == 900
    assert summary.max_context_tokens == 3000
    assert summary.tokens_by_doc_type == {"clinical_note": 700, "pathology_report": 200}


# ── the LLM exchange ─────────────────────────────────────────────────────────

def test_read_llm_exchanges_returns_the_messages_and_the_raw_response(dr):
    steps = extract_output_dir(phi_patient_run_dir(RUN, PATIENT, dr)) / VARIABLE / "steps"
    _write(steps / "step_a" / "receipt.json", json.dumps({"payload": {
        "messages_sent": [{"role": "user", "content": "the rendered prompt"}],
        "response_raw": "{\"date_of_birth\": \"1970-01-01\"}",
    }}))

    exchange = run_inspection.read_llm_exchanges(RUN, PATIENT, VARIABLE, dr)[0]

    assert exchange.step_id == "step_a"
    assert "the rendered prompt" in exchange.messages
    assert "1970-01-01" in exchange.response


# ── validation verdicts ──────────────────────────────────────────────────────

def test_read_invariants_returns_both_check_files_when_present(dr):
    patient_root = phi_patient_run_dir(RUN, PATIENT, dr)
    _write(extract_output_dir(patient_root) / VARIABLE / "invariants.json",
           json.dumps({"passed": True}))
    _write(patient_root / "clinical_invariants.json", json.dumps({"outcomes": []}))

    reports = run_inspection.read_invariants(RUN, PATIENT, VARIABLE, dr)

    assert [r.name for r in reports] == ["invariants.json", "clinical_invariants.json"]


# ── the recipe that ran ──────────────────────────────────────────────────────

def test_the_recipe_shown_is_the_runs_sealed_copy_when_it_has_one(dr):
    sealed = phi_intermediate_run_dir(RUN, dr) / "code" / "recipes" / "basic" / VARIABLE / "v1"
    _write(sealed / f"{VARIABLE}_v1_recipe.yaml", "name: the sealed copy\n")

    recipe = run_inspection.read_recipe_text(RUN, VARIABLE, dr)

    assert recipe is not None
    assert recipe.source == "this run's sealed code bundle"
    assert "the sealed copy" in recipe.text


def test_without_a_sealed_copy_the_working_tree_recipe_is_shown_and_labeled(dr):
    recipe = run_inspection.read_recipe_text(RUN, VARIABLE, dr)

    assert recipe is not None
    assert "working tree" in recipe.source
    assert recipe.path.startswith("var_extraction_recipes/")


def test_list_recipes_names_every_variable_on_disk():
    lines = run_inspection.list_recipes()

    named = {line.split(" ")[0].rsplit("/", 1)[-1] for line in lines}
    assert {"date_of_birth", "date_of_death", "stage"} <= named
    assert all("(" in line and line.strip().endswith(")") for line in lines)


# ── run rollup + exhaust + export ────────────────────────────────────────────

def test_read_run_summary_returns_the_summary_dict(dr):
    _write(phi_intermediate_run_dir(RUN, dr) / "summary.json",
           json.dumps({"status": "completed", "per_step_completed": {"ingest": 1}}))

    summary = run_inspection.read_run_summary(RUN, dr)

    assert summary == {"status": "completed", "per_step_completed": {"ingest": 1}}


def test_read_exhaust_manifest_summarizes_the_record_types(dr):
    _write(no_phi_manifest_path(RUN, dr), json.dumps({
        "schema_version": 3, "vocab_version": 2, "surrogate_version": 1,
        "secret_fingerprint": "abcd",
        "record_types": {"selection_judgment": {"n_rows": 12, "n_records_failed": 0}},
    }))

    manifest = run_inspection.read_exhaust_manifest(RUN, dr)

    assert manifest is not None
    assert manifest.schema_version == 3
    assert manifest.record_types == [("selection_judgment", 12, 0)]


def test_export_calls_the_pipelines_own_export_with_this_runs_identity(dr, monkeypatch):
    """The app must produce the identical artifact `junior export-metadata` writes —
    the same function, the same run, the same destination convention — not a zip
    dialect of its own."""
    no_phi_run_dir(RUN, dr).mkdir(parents=True)
    recorded = {}

    def fake_export(*, run_id, output_path, dr):
        recorded.update(run_id=run_id, output_path=output_path, dr=dr)
        return output_path

    import jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata as export_module

    monkeypatch.setattr(export_module, "export_run_metadata", fake_export)

    bundle = run_inspection.export_shareable_zip(RUN, dr)

    assert recorded["run_id"] == RUN
    assert recorded["dr"] == dr
    assert bundle.name == f"{RUN}_metadata.zip"


def test_export_refuses_a_run_with_no_shareable_tree(dr):
    with pytest.raises(FileNotFoundError, match="no shareable summary"):
        run_inspection.export_shareable_zip(RUN, dr)


def test_run_values_csv_covers_the_whole_run_and_guards_the_run_id(dr):
    _write(extract_output_dir(phi_patient_run_dir(RUN, "P1", dr)) / "dob" / "result.json",
           json.dumps({"payload": {"ok": True, "data": {"dob": "1970-01-01"}}}))
    _write(extract_output_dir(phi_patient_run_dir(RUN, "P2", dr)) / "dob" / "result.json",
           json.dumps({"payload": {"ok": True, "data": {"dob": "1980-02-02"}}}))

    wide = run_inspection.run_values_csv(RUN, shape="wide", dr=dr)

    assert "P1" in wide and "P2" in wide, "one patient at a time was the old shape"
    assert "1970-01-01" in wide and "1980-02-02" in wide
    assert run_inspection.run_values_csv("sneaky_dir", dr=dr) == ""
