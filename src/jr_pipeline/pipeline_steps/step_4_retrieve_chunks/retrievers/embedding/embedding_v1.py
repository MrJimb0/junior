"""Similarity search over the embedding vectors — finds chunks whose meaning is
close to the query, not just its exact words (bm25/exact cover those).

The query is encoded and searched against the patient's HNSW index. The query
encoder must be the same one that built the corpus: if the fingerprints disagree
we refuse and raise immediately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import build_encoder
from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
    VECTOR_AFFECTING_FINGERPRINT_FIELDS,
    Encoder,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    RetrieverInfo,
)

_VERSION = "v1"


def _fingerprint_subset(fingerprint: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Pick out just the fields in ``keys`` from an encoder fingerprint — the
    alignment comparison looks only at fields that actually change the vectors."""
    return {k: fingerprint.get(k) for k in keys}


def verify_corpus_encoder_alignment(
    query_fingerprint: dict[str, Any] | None,
    stored_fingerprint: dict[str, Any] | None,
    *,
    source_present: bool,
) -> str:
    """Check that the model that vectorized this patient's chart is the same one
    being used to vectorize the query. Returns one of 'ok' / 'mismatch' /
    'rebuild' / 'fail'.

    Why this matters: comparing only the vector length (e.g. dim=768) is not
    enough. We compare the encoder *fingerprints*
    (the identifying values recorded for each model). Outcomes:
      'ok'      — fingerprints agree on every vector-affecting field.
      'mismatch'— both fingerprints present but they disagree somewhere.
      'rebuild' — the chart's companion metadata file (sidecar) has no encoder
                  fingerprint, but the original structured source is still on disk,
                  so the chart could be re-vectorized to recover it.
      'fail'    — no fingerprint AND no source to rebuild from.

    Only the fields the STORED chart actually recorded are compared. A field that
    is simply absent from the stored fingerprint (e.g. tokenizer_hash on a chart
    vectorized before that field existed) is unverifiable for that chart, not a
    mismatch. Treating an absent field as the value None and comparing it against a
    live encoder's real value would wrongly reject such a chart on the
    retrieval-only path — where we read an existing chart and do not re-vectorize.
    "Field absent entirely" is different from "field present but set to None" (a
    model identified only by hub id), and the latter IS compared.

    The encoder's ``model_id`` (the file *path* it was loaded from) is deliberately
    EXCLUDED from this comparison: a path does not affect the vectors, and including
    it would reject a chart vectorized on one machine (e.g. the GPU cluster
    node) from being searched on another (e.g. the laptop) even when the model
    weights are byte-for-byte identical — which would break the intended
    "build on the cluster, search on the laptop" split. The other fields —
    ``model_sha256`` (a content hash of the weight file), ``tokenizer_hash``, and
    pooling/max_tokens/normalize/dtype — capture the true model identity and still
    fully tell two different models apart.
    """
    if stored_fingerprint is None:
        return "rebuild" if source_present else "fail"
    # Skip model_id (the load path) — see docstring: the same model weights stored
    # at a different path on another machine must count as a match, not a mismatch.
    recorded_keys = tuple(
        k for k in VECTOR_AFFECTING_FINGERPRINT_FIELDS if k in stored_fingerprint and k != "model_id"
    )
    if _fingerprint_subset(stored_fingerprint, recorded_keys) == _fingerprint_subset(
        query_fingerprint or {}, recorded_keys
    ):
        return "ok"
    return "mismatch"

# ── recipe config options ─────────────────────────────────────────────────────
# Set in your recipe YAML under retrieval:
#
#   kind:       embedding
#   k:          10     how many chunks to return (set at the recipe step level)
#   ef_search:   80     how hard the index looks — higher finds more of the true
#                       nearest neighbors (better recall) but runs slower
#   oversample:  4      pull k × oversample candidates first, then filter down to k
#                       (so per-file filtering below still leaves enough results)
#   space:       ip     how similarity is measured — must match what was used when
#                       the index was built; "ip" (inner product) is the default for junior
#   source_file: (none) restrict search to one file, e.g. pathology_report.csv
#
# Example:
#   retrieval:
#     kind: embedding
#     query: "HER2 receptor status immunohistochemistry"
#     k: 8
#     source_file: pathology_report.csv


@dataclass
class EmbeddingRetriever:
    """Finds chunks by semantic similarity — best when the query and relevant text don't share exact words."""

    info: RetrieverInfo

    def __init__(
        self,
        *,
        encoder: Encoder | None = None,
        encoder_cfg: dict[str, Any] | None = None,
        space: str = "ip",
        ef_search: int = 80,
        oversample: int = 4,
        source_file: str | None = None,
    ):
        if encoder is None:
            if encoder_cfg is None:
                raise ValueError("EmbeddingRetriever needs encoder or encoder_cfg")
            encoder = build_encoder(encoder_cfg)
        self._encoder = encoder
        self._space = space
        self._ef = ef_search
        self._oversample = max(1, int(oversample))
        self._source_file = source_file
        self.info = RetrieverInfo(
            kind="embedding",
            version=_VERSION,
            config={
                "encoder": encoder.fingerprint(),
                "space": space,
                "ef_search": ef_search,
                "oversample": oversample,
                "source_file": source_file,
            },
        )

    @property
    def score_normalization(self) -> str:
        return "l2_normalized_ip"

    def query(
        self,
        corpus: PatientChunkStore,
        *,
        text: str,
        k: int,
    ) -> list[Candidate]:
        """Return the k chunks whose meaning is closest to `text`, closest first."""
        if not text.strip():
            return []

        vec = self._encoder.embed_batch([text])
        if vec.size == 0:
            return []
        dim = int(vec.shape[1])

        emb = corpus.embeddings()
        if emb.shape[0] == 0:
            return []
        if emb.shape[1] != dim:
            raise ValueError(
                f"embedding dim mismatch: query encoder produces dim={dim} but "
                f"corpus embeddings.npy has dim={emb.shape[1]}. The query encoder "
                "fingerprint disagrees with the stored embeddings — re-embed the "
                "corpus or use the matching encoder."
            )
        alignment = verify_corpus_encoder_alignment(
            self.info.config["encoder"],
            corpus.stored_encoder_fingerprint(),
            source_present=corpus.has_structured_source(),
        )
        if alignment == "mismatch":
            raise ValueError(
                "encoder mismatch: the query encoder fingerprint disagrees with the "
                "encoder that embedded the corpus (same dim, different model/config). "
                "Re-embed the corpus or query with the matching encoder."
            )
        if alignment == "rebuild":
            raise ValueError(
                "corpus embeddings carry no encoder fingerprint (legacy embed); re-embed "
                "the corpus so retrieval can verify encoder alignment."
            )
        if alignment == "fail":
            raise ValueError(
                "corpus embeddings carry no encoder fingerprint and the structured "
                "source is gone — cannot verify encoder alignment or re-embed."
            )

        idx = corpus.hnsw(dim=dim, space=self._space, ef_search=self._ef)

        pull = min(emb.shape[0], k * self._oversample)
        labels, distances = idx.knn_query(vec.astype(np.float32), k=pull)
        labels = labels[0]
        distances = distances[0]

        # A requested source_file that matches no chunks (a typo, or a file this
        # patient lacks) must yield NO results — not fall through to the global
        # nearest neighbors, which would mislabel other files' evidence as this
        # file's. So gate on whether a filter was requested, not on whether
        # `allowed` happens to be non-empty. Matches bm25/exact, which return []
        # in the same case.
        apply_source_filter = bool(self._source_file)
        allowed: set[str] = set()
        if apply_source_filter:
            allowed = set(
                corpus.chunk_index.filter(
                    pl.col("source_file") == self._source_file
                )["chunk_id"].to_list()
            )

        chunk_ids = corpus.chunk_index["chunk_id"].to_list()
        out: list[Candidate] = []
        for vector_id, dist in zip(labels, distances, strict=True):
            if vector_id >= len(chunk_ids):
                continue
            cid = chunk_ids[int(vector_id)]
            if apply_source_filter and cid not in allowed:
                continue
            # The index reports a distance (1 - cosine); turn it back into a
            # similarity score so that higher means more similar.
            score = 1.0 - float(dist)
            out.append(Candidate(chunk_id=cid, rank=len(out) + 1, score=score))
            if len(out) >= k:
                break
        return out
