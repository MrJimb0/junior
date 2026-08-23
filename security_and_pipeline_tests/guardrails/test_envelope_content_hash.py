"""envelope_for computes a real content_hash, never the all-zeros stub.

An all-zeros placeholder ('sha256:'+'0'*64) is schema-valid, so an envelope built and
validated without a hash of its own payload would ship a zero hash and still pass -- a
silent integrity hazard. envelope_for hashes the payload itself.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)

_ZEROS = "sha256:" + "0" * 64


def _make(payload):
    return envelope_for(
        artifact_type="manifest", sensitivity="low", stream="data",
        run_id="20260617_000000", step="code_seal", payload=payload,
    )


def test_envelope_content_hash_is_real_not_placeholder():
    env = _make({"a": 1})
    assert env["content_hash"] != _ZEROS
    assert env["content_hash"].startswith("sha256:") and len(env["content_hash"]) == len(_ZEROS)


def test_envelope_validates_without_a_manual_recompute():
    # a manifest payload, validated straight out of envelope_for -- no caller re-stamp
    env = envelope_for(
        artifact_type="manifest", sensitivity="low", stream="data",
        run_id="20260617_000000", step="code_seal",
        payload={
            "run_id": "20260617_000000", "code_lock_hash": "sha256:" + "a" * 64,
            "created_at": "2026-06-17T00:00:00+00:00",
            "started_by": {"user": "t", "host": "h", "slurm_job_id": None},
            "entry_point_name": "t", "config_alias": "t", "n_target_patients": 0,
            "completed_at": None, "status": "running", "model_sha256": None,
        },
    )
    validate_artifact(env, "manifest")  # must not raise


def test_content_hash_is_deterministic_for_the_same_payload():
    assert _make({"x": [1, 2, 3]})["content_hash"] == _make({"x": [1, 2, 3]})["content_hash"]
    assert _make({"x": 1})["content_hash"] != _make({"x": 2})["content_hash"]
