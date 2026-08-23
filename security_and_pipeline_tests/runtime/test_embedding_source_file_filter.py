"""Regression: the embedding retriever's ``source_file`` filter must return [] when
the requested file matches no chunks — NOT fall through to the global nearest
neighbors, which would mislabel other files' evidence as the requested file's. An
absent or misspelled source_file must return nothing, the same contract bm25/exact hold.

Drives the real embed + index steps with a fake one-hot marker encoder (no torch),
so the corpus, sidecars, and hnsw hashes are all production-valid.
"""
import json
import re

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import embed as embed_mod
from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import run_embed_one
from jr_pipeline.pipeline_steps.step_3_build_vector_index.build_index import run_index_one
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.embedding.embedding_v1 import (
    EmbeddingRetriever,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore


class _MarkerEncoder:
    """Fake Encoder: one-hot row k for a chunk text containing 'marker k'. Satisfies
    the Encoder protocol without torch or a real model."""

    model_id = "fake-marker-encoder"
    max_tokens = 512
    num_special_tokens = 0

    def __init__(self, dim: int):
        self.dim = dim

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            m = re.search(r"marker (\d+)", text)
            if m:
                out[i, int(m.group(1))] = 1.0
        return out

    def tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def fingerprint(self) -> dict:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens, "dim": self.dim}


def _write_source_parquet(structured, stem: str, source_file: str, markers: list[int]) -> None:
    texts = [f"clinical note marker {i} end" for i in markers]
    pl.DataFrame({"text": texts}).write_parquet(structured / f"{stem}.parquet")
    (structured / f"{stem}.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": source_file, "parquet_content_hash": "x"}})
    )


def _build_corpus(tmp_path, monkeypatch) -> tuple[PatientChunkStore, int]:
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    run_id, patient_id, dim = "20260101_srcfilter", "Test_Patient", 6
    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    # Two source files so a filter can select one and a typo can select neither.
    _write_source_parquet(structured, "notes", "notes.csv", [0, 1, 2])
    _write_source_parquet(structured, "labs", "labs.csv", [3, 4, 5])

    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _MarkerEncoder(dim))
    run_embed_one(
        cfg={"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}},
        patient_id=patient_id,
    )
    run_index_one(cfg={"run_id": run_id}, patient_id=patient_id)
    return PatientChunkStore(patient_root=phi_patient_run_dir(run_id, patient_id)), dim


def test_missing_source_file_returns_empty_not_global_neighbors(tmp_path, monkeypatch):
    store, dim = _build_corpus(tmp_path, monkeypatch)
    retriever = EmbeddingRetriever(encoder=_MarkerEncoder(dim), source_file="does_not_exist.csv")
    hits = retriever.query(store, text="clinical marker 0 end", k=3)
    assert hits == []


def test_source_file_filter_keeps_only_that_file(tmp_path, monkeypatch):
    store, dim = _build_corpus(tmp_path, monkeypatch)
    notes_ids = set(
        store.chunk_index.filter(pl.col("source_file") == "notes.csv")["chunk_id"].to_list()
    )
    retriever = EmbeddingRetriever(encoder=_MarkerEncoder(dim), source_file="notes.csv")
    hits = retriever.query(store, text="clinical marker 1 end", k=6)
    assert hits, "expected at least one notes.csv hit"
    assert all(h.chunk_id in notes_ids for h in hits)
