"""A tokenizer swap under the same model_id busts the embed cache.

A tokenizer swap changes tokenization — chunk boundaries, token counts, and the
special-token-reserved default window — without touching model_id/config/weights.
The encoder fingerprint carries a combined hash of the tokenizer files on disk,
and that hash is part of the cache-key subset (which retrieval-side encoder
alignment also reuses). All load-free: no model load, no torch.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
    VECTOR_AFFECTING_FINGERPRINT_FIELDS,
    HFEncoder,
)


def _model_dir(tmp_path, *, tokenizer=True):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"W")
    if tokenizer:
        (d / "tokenizer.json").write_text('{"version":"1"}')
        (d / "tokenizer_config.json").write_text('{"model_max_length":512}')
        (d / "special_tokens_map.json").write_text('{"cls_token":"[CLS]"}')
    return d


def test_tokenizer_hash_in_cache_key_fields():
    # belt-and-suspenders: the bust below relies on this field being compared.
    assert "tokenizer_hash" in VECTOR_AFFECTING_FINGERPRINT_FIELDS


def test_fingerprint_includes_tokenizer_hash(tmp_path):
    fp = HFEncoder(model_id=str(_model_dir(tmp_path))).fingerprint()
    assert fp["tokenizer_hash"] is not None
    assert fp["tokenizer_hash"].startswith("sha256:")


def test_tokenizer_swap_changes_fingerprint(tmp_path):
    d = _model_dir(tmp_path)
    before = HFEncoder(model_id=str(d)).fingerprint()["tokenizer_hash"]
    (d / "tokenizer.json").write_text('{"version":"2"}')  # in-place tokenizer edit
    after = HFEncoder(model_id=str(d)).fingerprint()["tokenizer_hash"]
    assert before != after


def test_tokenizer_config_swap_changes_fingerprint(tmp_path):
    # not just tokenizer.json: any identity file changes the combined hash
    d = _model_dir(tmp_path)
    before = HFEncoder(model_id=str(d)).fingerprint()["tokenizer_hash"]
    (d / "special_tokens_map.json").write_text('{"cls_token":"[BOS]"}')
    after = HFEncoder(model_id=str(d)).fingerprint()["tokenizer_hash"]
    assert before != after


def test_tokenizer_hash_is_cached_across_calls_and_instances(tmp_path, monkeypatch):
    # fingerprint() runs per patient in embed and per variable via EmbeddingRetriever.__init__
    # (a fresh encoder each time), so the multi-MB tokenizer files must be hashed once, not on
    # every call — and the cache must be shared across encoder instances on the same path.
    from jr_pipeline.pipeline_steps.step_2_embed_chunks import encoder as encoder_mod

    calls = {"n": 0}
    real_hash_file = encoder_mod.hash_file

    def counting_hash_file(path):
        calls["n"] += 1
        return real_hash_file(path)

    monkeypatch.setattr(encoder_mod, "hash_file", counting_hash_file)

    model_dir = str(_model_dir(tmp_path))
    first = HFEncoder(model_id=model_dir)._tokenizer_identity_hash()
    after_first = calls["n"]
    assert after_first > 0  # the tokenizer files were hashed on the first call

    second = HFEncoder(model_id=model_dir)._tokenizer_identity_hash()
    assert calls["n"] == after_first  # a second instance on the same path re-uses the cache
    assert first == second is not None  # and the value is unchanged


def test_tokenizer_hash_none_without_local_files(tmp_path):
    # hub id: no local dir to hash (consistent with model_sha256/local_config_hash)
    assert HFEncoder(model_id="org/model").fingerprint()["tokenizer_hash"] is None
    # local dir with weights + config but no tokenizer files
    d = _model_dir(tmp_path, tokenizer=False)
    assert HFEncoder(model_id=str(d)).fingerprint()["tokenizer_hash"] is None
