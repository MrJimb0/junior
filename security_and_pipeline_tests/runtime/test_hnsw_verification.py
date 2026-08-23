"""Retrieval verifies the hnsw index matches the artifacts it indexes.

verify_hnsw_consistency reads the sidecar we already write (hnsw.bin.meta.json)
and rejects a stale index (embeddings changed since it was built) or a row-count
mismatch between the index, embeddings.npy, and chunk_index — either would
mislabel every neighbor, since retrieval maps an index label to a chunk_index row
by position.
"""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.patient_chunk_store import verify_hnsw_consistency


def test_consistent_index_passes():
    verify_hnsw_consistency(
        sidecar_meta={"embeddings_sha256": "sha256:abc"},
        current_emb_hash="sha256:abc",
        n_index=5, n_emb=5, n_chunks=5,
    )  # no raise


def test_stale_index_when_embeddings_hash_differs():
    with pytest.raises(RuntimeError, match="stale"):
        verify_hnsw_consistency(
            sidecar_meta={"embeddings_sha256": "sha256:OLD"},
            current_emb_hash="sha256:NEW",
            n_index=5, n_emb=5, n_chunks=5,
        )


def test_missing_sidecar_raises():
    with pytest.raises(RuntimeError, match="sidecar"):
        verify_hnsw_consistency(
            sidecar_meta=None,
            current_emb_hash="sha256:abc",
            n_index=5, n_emb=5, n_chunks=5,
        )


def test_row_count_mismatch_raises():
    with pytest.raises(RuntimeError, match="row-count mismatch"):
        verify_hnsw_consistency(
            sidecar_meta={"embeddings_sha256": "sha256:abc"},
            current_emb_hash="sha256:abc",
            n_index=5, n_emb=4, n_chunks=5,
        )
