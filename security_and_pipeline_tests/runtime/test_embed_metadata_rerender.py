"""Embed is always a full per-patient pass.

A full re-embed re-derives metadata from the current parquet every time, so a
metadata-only edit (e.g. document_date) on a row whose text was unchanged still
surfaces the new document_date rather than reusing a stored chunk that carries a
stale value.
"""
from __future__ import annotations

import json
import re

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import embed as embed_mod
from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import run_embed_one
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


class _ConstEncoder:
    """Minimal Encoder: whitespace content tokens, constant unit vectors. No model."""

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


def _write_source(structured, *, date):
    structured.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"text": ["a clinical note"], "date": [date]}).write_parquet(
        structured / "notes.parquet"
    )
    # Sidecar WITHOUT parquet_content_hash -> embed hashes the file itself, so a
    # parquet edit busts the embed cache (as a real re-ingest would).
    (structured / "notes.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": "notes.csv"}})
    )


def _document_dates(patient_out):
    return pl.read_parquet(patient_out / "chunk_index.parquet")["document_date"].to_list()


def test_metadata_only_edit_rerenders_document_date(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _ConstEncoder())
    run_id, patient_id = "20260101_meta", "Test_Patient"
    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    cfg = {"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}}

    _write_source(structured, date="2020-01-01")
    run_embed_one(cfg=cfg, patient_id=patient_id)
    patient_out = phi_patient_run_dir(run_id, patient_id)
    assert _document_dates(patient_out) == ["2020-01-01"]

    # change ONLY the metadata column; the text is byte-identical
    _write_source(structured, date="2021-12-31")
    second = run_embed_one(cfg=cfg, patient_id=patient_id)

    assert second.get("cached") is not True  # source hash changed -> full re-embed
    assert _document_dates(patient_out) == ["2021-12-31"]  # re-rendered, not stale
