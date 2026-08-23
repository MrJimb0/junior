"""Weight drift busts the embed cache.

model.safetensors is the only weight format the encoder loads and hashes, so
fingerprint()'s model_sha256 always covers the exact file transformers reads;
an in-place weight swap changes the hash and forces a re-embed.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import HFEncoder
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)


def _fake_model_dir(tmp_path, weight_bytes=b"WEIGHTS"):
    d = tmp_path / "model"
    d.mkdir()
    (d / "model.safetensors").write_bytes(weight_bytes)
    (d / "config.json").write_text("{}")
    return d


def test_fingerprint_hashes_safetensors(tmp_path):
    d = _fake_model_dir(tmp_path)
    fp = HFEncoder(model_id=str(d)).fingerprint()
    assert fp["model_sha256"] == hash_file(d / "model.safetensors")


def test_weight_swap_changes_model_sha256(tmp_path):
    d = _fake_model_dir(tmp_path, weight_bytes=b"OLD-WEIGHTS")
    before = HFEncoder(model_id=str(d)).fingerprint()["model_sha256"]
    # different length so the (path, size, mtime) hash cache can't serve the old value
    (d / "model.safetensors").write_bytes(b"NEW-WEIGHTS!")
    after = HFEncoder(model_id=str(d)).fingerprint()["model_sha256"]
    assert before != after


def test_verify_local_files_sets_weight_sha(tmp_path):
    d = _fake_model_dir(tmp_path)
    good = hash_file(d / "model.safetensors")
    enc = HFEncoder(model_id=str(d), expected_file_sha256={"model.safetensors": good})
    enc._verify_local_files()  # matching pin -> no raise
    assert enc._model_weight_sha256 == good


def test_verify_local_files_raises_on_mismatch(tmp_path):
    d = _fake_model_dir(tmp_path)
    enc = HFEncoder(
        model_id=str(d), expected_file_sha256={"model.safetensors": "sha256:deadbeef"}
    )
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        enc._verify_local_files()
