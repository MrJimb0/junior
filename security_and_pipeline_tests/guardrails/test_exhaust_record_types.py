"""method_provenance + extraction_outcome records are PHI-clean.

extraction_outcome carries why/what-it-cost (ok, value_shape, bucketed error kind,
token economics) with the patient as an HMAC surrogate and NO raw error text;
method_provenance is the run's single RunHeader join key (no patient data).
"""
from __future__ import annotations

import json

import pytest

from jr_pipeline.pipeline_steps.step_8_organize_output.extraction_outcome import (
    _value_shape,
    emit_extraction_outcome,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_method_provenance,
)
from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl

RUN = "20260617_000000"


def setup_function(_):
    sl._CACHE.clear()


def _transcript():
    return {
        "variable": "date_of_birth",
        "recipe": {"name": "date_of_birth", "version": "v1"},
        "provider_config_hash": "sha256:" + "a" * 64,
        "code_lock_hash": "sha256:" + "b" * 64,
        "step_evidence": [{"trace": {
            "archetype": "point_fact", "encoder_fingerprint": "a1b2c3d4e5f6a7b8",
            "reranker_fingerprint": "b1b2c3d4e5f6a7b8", "chunking_config_id": "c1b2c3d4e5f6a7b8",
        }}],
        "steps": [],
    }


def _records(record_type, tmp_path):
    d = layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / record_type
    out = []
    for shard in d.glob("*.jsonl"):
        out += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return out


def test_extraction_outcome_ok_is_clean_and_surrogated(tmp_path):
    emit_extraction_outcome(
        run_id=RUN, patient_id="Test_Patient", transcript=_transcript(),
        ok=True, data_final={"date_of_birth": "1957-04-15", "dob_evidence_chunk_id": "x"},
        errors=[], data_root=tmp_path,
    )
    recs = _records("extraction_outcome", tmp_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["record_type"] == "extraction_outcome"
    assert rec["ok"] is True and rec["value_shape"] == "object"
    assert rec["error_kind"] == "none" and rec["parse_failure_kind"] == "none"
    assert rec["patient_surrogate"].startswith("surr:v1:run:patient_surrogate:")
    # the raw extracted value never enters the record
    blob = json.dumps(rec)
    assert "1957-04-15" not in blob and "Test_Patient" not in blob


def test_extraction_outcome_buckets_error_without_leaking_text(tmp_path):
    emit_extraction_outcome(
        run_id=RUN, patient_id="Test_Patient", transcript=_transcript(),
        ok=False, data_final=None,
        errors=["JSONDecodeError: Expecting value near 'Patient John Doe MRN 0012345'"],
        data_root=tmp_path,
    )
    rec = _records("extraction_outcome", tmp_path)[0]
    assert rec["ok"] is False and rec["value_shape"] == "null"
    assert rec["error_kind"] == "parse_failure" and rec["parse_failure_kind"] == "json_decode_error"
    blob = json.dumps(rec)
    assert "John Doe" not in blob and "0012345" not in blob  # raw error text never travels


@pytest.mark.parametrize("data,shape", [
    ({"a": 1}, "object"), ([1, 2], "list"), ("x", "scalar"), (3, "scalar"), (None, "null"),
])
def test_value_shape_derivation(data, shape):
    assert _value_shape(data) == shape


def test_method_provenance_is_run_level_and_patient_free(tmp_path):
    record_method_provenance(
        RUN, encoder_fingerprint="a1b2c3d4e5f6a7b8",
        chunking_config_id="c1b2c3d4e5f6a7b8", code_lock_hash="sha256:" + "b" * 64,
        data_root=tmp_path,
    )
    recs = _records("method_provenance", tmp_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["record_type"] == "method_provenance"
    assert rec["encoder_fingerprint"] == "a1b2c3d4e5f6a7b8"
    assert "patient_surrogate" not in rec  # run-level: no patient
