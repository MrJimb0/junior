"""Guardrail: embeddings.npy row i == chunk_index.parquet row i (write-order invariant).

embed writes both artifacts from one ``chunk_manifest`` in the same order; retrieval
(embedding_v1) maps an HNSW label to chunk_index BY POSITION, so a bug that reordered
one artifact relative to the other would silently mis-attribute every vector.

This test DRIVES the real embed step (run_embed_one) with a fake encoder
that embeds each chunk's ``marker <k>`` into one-hot row k. Asserting
``argmax(embeddings[i]) == chunk_index['row_id'][i]`` ties each written vector to the
chunk_index row it must correspond to. A second test covers the consumption-side
check that an HNSW label maps back to its own row by position.
"""
import json
import re

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import embed as embed_mod
from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import run_embed_one
from jr_pipeline.pipeline_steps.step_3_build_vector_index.build_index import HnswlibIndex
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


class _MarkerEncoder:
    """Fake Encoder: one-hot row k for a chunk text containing 'marker k'.

    Satisfies the Encoder protocol (dim/max_tokens/model_id + embed_batch /
    tokenize_with_offsets / fingerprint) without torch or a real model.
    """

    model_id = "fake-marker-encoder"
    max_tokens = 512
    num_special_tokens = 0  # the fake's embed_batch adds no [CLS]/[SEP]

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


def test_embed_writes_embeddings_aligned_to_chunk_index(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    run_id, patient_id, n = "20260101_alignment", "Test_Patient", 6

    structured = phi_patient_run_dir(run_id, patient_id) / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    # one short row per marker -> exactly one chunk per row, in row order
    texts = [f"clinical note marker {i} end of note" for i in range(n)]
    pl.DataFrame({"text": texts}).write_parquet(structured / "notes.parquet")
    (structured / "notes.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": "notes.csv", "parquet_content_hash": "x"}})
    )

    monkeypatch.setattr(embed_mod, "build_encoder", lambda cfg: _MarkerEncoder(n))

    summary = run_embed_one(
        cfg={"run_id": run_id, "encoder": {"model_id": "fake", "max_tokens": 512}},
        patient_id=patient_id,
    )
    assert summary["chunks"] == n

    patient_out = phi_patient_run_dir(run_id, patient_id)
    embeddings = np.load(patient_out / "embeddings.npy")
    index = pl.read_parquet(patient_out / "chunk_index.parquet")

    assert embeddings.shape == (n, n)
    assert index.height == n
    row_ids = index["row_id"].to_list()
    # each embeddings row is the one-hot for the marker in the chunk the SAME-position
    # chunk_index row describes; a write-order regression breaks this per-position tie.
    for position in range(n):
        assert embeddings[position].sum() == 1.0  # the fake produced a real one-hot
        assert int(np.argmax(embeddings[position])) == row_ids[position]


def test_hnsw_label_maps_back_to_its_own_row_by_position(tmp_path):
    n, dim = 8, 16
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((n, dim)).astype("float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    chunk_ids = [f"c{i}" for i in range(n)]

    idx = HnswlibIndex()
    idx.build(mat, space="ip", m=16, ef_construction=200)
    hnsw_path = tmp_path / "hnsw.bin"
    idx.save(hnsw_path)

    import hnswlib

    h = hnswlib.Index(space="ip", dim=dim)
    h.load_index(str(hnsw_path))
    h.set_ef(50)

    for i in range(n):
        labels, _ = h.knn_query(mat[i : i + 1], k=1)
        label = int(labels[0][0])
        assert label == i, f"row {i}'s own vector returned label {label}"
        assert chunk_ids[label] == f"c{i}"
