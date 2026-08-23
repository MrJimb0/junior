"""The artifact-path registry + write_artifact.

artifact_path is the single source of truth for run-hierarchy artifact destinations;
write_artifact hashes + validates + atomically writes an envelope (registry path, or an
explicit path for data-co-located sidecars).
"""
from __future__ import annotations

import json

import pytest

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path, write_artifact
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    retrieval_output_dir,
)

RUN = "20260101_000000"
PID = "p1"


def test_artifact_path_registry_matches_the_writers(tmp_path):
    dr = tmp_path
    inter = phi_intermediate_run_dir(RUN, dr)
    pr = phi_patient_run_dir(RUN, PID, dr)
    assert artifact_path("manifest", run_id=RUN, dr=dr) == inter / "manifest.json"
    assert artifact_path("run_roster", run_id=RUN, dr=dr) == inter / "run_roster.json"
    assert artifact_path("run_metrics_shard", run_id=RUN, dr=dr, name="candidate_metrics") == \
        inter / "data" / "eval" / "candidate_metrics.json"
    assert artifact_path("variable_result", run_id=RUN, dr=dr, patient_id=PID, variable="dob") == \
        extract_output_dir(pr) / "dob" / "result.json"
    assert artifact_path("invariants", run_id=RUN, dr=dr, patient_id=PID, variable="dob") == \
        extract_output_dir(pr) / "dob" / "invariants.json"
    assert artifact_path("step_receipt", run_id=RUN, dr=dr, patient_id=PID, variable="dob", step_id="s0") == \
        extract_output_dir(pr) / "dob" / "steps" / "s0" / "receipt.json"
    assert artifact_path("retrieval", run_id=RUN, dr=dr, patient_id=PID, query_id="q/1") == \
        retrieval_output_dir(pr) / "q_1.json"  # the '/' in a qid is sanitized to '_'
    assert artifact_path("source_snapshot", run_id=RUN, dr=dr, patient_id=PID) == pr / "source_snapshot.json"


def test_artifact_path_honors_explicit_base_overrides(tmp_path):
    # A writer passes the base it holds (a possibly non-default root); the registry
    # still owns the subpath -> byte-identical even off the default data_root.
    proot = tmp_path / "patient"
    inter = tmp_path / "inter"
    assert artifact_path("variable_result", patient_root=proot, variable="dob") == \
        extract_output_dir(proot) / "dob" / "result.json"
    assert artifact_path("source_snapshot", patient_root=proot) == proot / "source_snapshot.json"
    assert artifact_path("manifest", run_root=inter) == inter / "manifest.json"


def test_artifact_path_unknown_type_raises():
    with pytest.raises(KeyError):
        artifact_path("not_a_registry_type", run_id=RUN)


def test_write_artifact_resolves_path_hashes_and_validates(tmp_path):
    env = envelope_for(
        artifact_type="manifest", sensitivity="low", stream="data", run_id=RUN, step="code_seal",
        payload={
            "run_id": RUN, "code_lock_hash": "sha256:" + "a" * 64,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_by": {"user": "t", "host": "h", "slurm_job_id": None},
            "entry_point_name": "t", "config_alias": "t", "n_target_patients": 0,
            "completed_at": None, "status": "running", "model_sha256": None,
        },
    )
    out = write_artifact(env, dr=tmp_path)
    assert out == phi_intermediate_run_dir(RUN, tmp_path) / "manifest.json" and out.is_file()
    written = json.loads(out.read_text())
    assert written["content_hash"].startswith("sha256:")
    assert written["content_hash"] != "sha256:" + "0" * 64  # a real hash, not the stub


def test_write_artifact_honors_explicit_path_for_sidecars(tmp_path):
    env = envelope_for(
        artifact_type="retrieval", sensitivity="medium", stream="data", run_id=RUN,
        step="retrieve", patient_id=PID, payload={"x": 1},
    )
    side = tmp_path / "structured" / "notes.parquet.meta.json"
    out = write_artifact(env, path=side, schema_name="artifact_envelope")
    assert out == side and side.is_file()


def test_an_empty_evidence_step_receipt_writes_an_honest_record(tmp_path):
    """A retrieval that surfaces nothing calls no model, and the step writes
    resolved_model: null (retrieve_and_prompt). The write gate rejecting that
    receipt killed the whole variable — including the recipe's own fallback
    passes, which exist precisely for the chart this happens on."""
    env = envelope_for(
        artifact_type="step_receipt", sensitivity="high", stream="data", run_id=RUN,
        step="extract", patient_id=PID, variable="date_of_death",
        step_id="demographics_grab",
        payload={
            "kind": "retrieve_and_prompt",
            "retrieval": {"kind": "direct_parquet", "candidates": []},
            "messages_sent": [],
            "resolved_model": None,
            "response_raw": None,
            "response_parsed": None,
        },
    )
    out = write_artifact(env, dr=tmp_path)
    assert out.is_file()
    assert json.loads(out.read_text())["payload"]["resolved_model"] is None


def test_a_called_model_receipt_still_names_the_model(tmp_path):
    """Nullable must not mean shapeless: when a model DID answer, the receipt
    still has to carry the full model identity."""
    import jsonschema

    env = envelope_for(
        artifact_type="step_receipt", sensitivity="high", stream="data", run_id=RUN,
        step="extract", patient_id=PID, variable="date_of_death", step_id="death_search",
        payload={
            "kind": "retrieve_and_prompt",
            "resolved_model": {"endpoint_name": "local_qwen"},
        },
    )
    with pytest.raises(jsonschema.ValidationError):
        write_artifact(env, dr=tmp_path)
