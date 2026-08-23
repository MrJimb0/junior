"""The embed cache keys on the chunker config too.

A chunk-boundary change (an explicit window, or the default-window logic) must
rebuild the embeddings. The chunk_index sidecar records a chunker identity (raw
config + a logic version) and the cache compares it alongside the encoder
fingerprint and source-parquet hashes.
"""
from __future__ import annotations

import json
import re

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import embed as embed_mod
from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import (
    _EMBED_OUTPUT_LOGIC_VERSION,
    _chunker_cache_identity,
    run_embed_one,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


class _ConstEncoder:
    model_id = "fake-const"
    max_tokens = 512
    num_special_tokens = 2

    def __init__(self, dim: int = 4):
        self.dim = dim

    def embed_batch(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    def tokenize_with_offsets(self, text):
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def fingerprint(self):
        return {"model_id": self.model_id, "max_tokens": self.max_tokens, "dim": self.dim}


def _write_source(structured):
    structured.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"text": ["one two three four five six"]}).write_parquet(
        structured / "notes.parquet"
    )
    (structured / "notes.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": "notes.csv"}})
    )


def test_chunker_cache_identity_has_logic_version_and_raw_fields():
    ident = _chunker_cache_identity(
        {"chunker": {"kind": "token_window", "window": 100, "overlap": 10}}
    )
    assert ident["logic_version"] == _EMBED_OUTPUT_LOGIC_VERSION
    assert ident["kind"] == "token_window"
    assert ident["window"] == 100
    assert ident["overlap"] == 10


def test_chunker_window_change_busts_embed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _ConstEncoder())
    run_id, patient_id = "20260101_ck", "P"
    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    _write_source(structured)

    base = {"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}}
    assert run_embed_one(cfg=base, patient_id=patient_id).get("cached") is not True
    assert run_embed_one(cfg=base, patient_id=patient_id).get("cached") is True  # unchanged

    # same source + encoder, but an explicit chunker window -> must rebuild
    changed = {**base, "chunker": {"kind": "token_window", "window": 3, "overlap": 0}}
    assert run_embed_one(cfg=changed, patient_id=patient_id).get("cached") is not True


# ── tokenizer drift busts the embed cache + resolved window recorded ──


class _TokEncoder(_ConstEncoder):
    """const encoder whose fingerprint carries a settable tokenizer_hash, so a
    tokenizer swap (same model_id/weights) can be simulated without torch."""

    def __init__(self, tokenizer_hash: str, dim: int = 4):
        super().__init__(dim)
        self._tokenizer_hash = tokenizer_hash

    def fingerprint(self):
        fp = super().fingerprint()
        fp["tokenizer_hash"] = self._tokenizer_hash
        return fp


def test_tokenizer_hash_change_busts_embed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _TokEncoder("sha256:aaa"))
    run_id, patient_id = "20260101_tk", "P"
    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    _write_source(structured)

    base = {"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}}
    assert run_embed_one(cfg=base, patient_id=patient_id).get("cached") is not True
    assert run_embed_one(cfg=base, patient_id=patient_id).get("cached") is True  # unchanged

    # swap the tokenizer (same model_id, weights, chunker, source) -> must rebuild
    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _TokEncoder("sha256:bbb"))
    assert run_embed_one(cfg=base, patient_id=patient_id).get("cached") is not True


def test_resolved_chunker_window_recorded_in_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _ConstEncoder())
    run_id, patient_id = "20260101_rw", "P"
    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    _write_source(structured)

    base = {"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}}
    run_embed_one(cfg=base, patient_id=patient_id)

    idx_meta = phi_patient_run_dir(run_id, patient_id) / "chunk_index.parquet.meta.json"
    # omitted window -> resolved content budget max_tokens 512 - num_special_tokens 2,
    # recorded as 510 (not None, the raw chunker identity's value).
    assert embed_mod._cached_sidecar_field(idx_meta, "chunker_resolved_window") == 510
