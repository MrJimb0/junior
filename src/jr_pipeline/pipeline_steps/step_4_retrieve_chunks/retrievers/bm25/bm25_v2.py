"""Find chunks by keyword overlap with the query — classic word-matching search.

This retriever ranks chunks by BM25
(classic keyword / term-frequency search)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    RetrieverInfo,
)

# Words are letter/digit runs (see _tok). This version string is recorded on every
# saved retrieval result, so any change to how text is split into words must bump it
# to force results to be recomputed rather than served stale.
_VERSION = "v2"

# ── recipe config options ─────────────────────────────────────────────────────
# Set in your recipe YAML under retrieval:
#
#   kind:        bm25
#   k:           10      chunks to return (set at the recipe step level)
#   source_file: (none)  restrict search to one file, e.g. clinical_note.csv;
#                        omit to search all embedded files
#   k1:          1.5     how much repeated query terms boost the score
#   b:           0.75    length normalization; lower = longer notes penalized less
#
# Example:
#   retrieval:
#     kind: bm25
#     query: "diagnosis invasive ductal carcinoma pathology"
#     k: 5
#     source_file: pathology_report.csv

_TOKEN = re.compile(r"[a-z0-9]+")

def _tok(s: str) -> list[str]:
    # Split text into words = runs of letters/digits in the lowercased text. This
    # keeps trailing punctuation off the word ("cancer," would otherwise not match
    # "cancer", separating a query term from the same word in the chunk and causing
    # real matches to be missed).
    return _TOKEN.findall((s or "").lower())

@dataclass
class BM25Retriever:
    """Keyword (term-frequency) retriever; k1/b are the standard BM25 knobs
    (documented in the config block above)."""
    info: RetrieverInfo
    def __init__(self, *, k1: float = 1.5, b: float = 0.75, source_file: str | None = None):
        self.info = RetrieverInfo(
            kind="bm25", version=_VERSION, config={"k1": k1, "b": b, "source_file": source_file}
        )
        self._k1 = k1
        self._b = b
        self._source_file = source_file

    @property
    def score_normalization(self) -> str:
        return "raw"

    def query(
        self,
        corpus: PatientChunkStore,
        *,
        text: str,
        k: int,
    ) -> list[Candidate]:
        """Return the k best-matching chunks for ``text`` by BM25 keyword score.

        Chunks with a zero-or-negative score (no real overlap) are dropped, and
        exact-score ties are broken by chunk_id so the order is stable across runs.
        """
        if not text.strip():
            return []
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RuntimeError("rank_bm25 is required for BM25Retriever") from e

        chunks = (
            corpus.chunk_index.filter(pl.col("source_file") == self._source_file)
            if self._source_file else corpus.chunk_index
        )
        # Only the chunk_id column is needed; pull it directly rather than
        # materializing every column of every row into dicts.
        ids = chunks.get_column("chunk_id").to_list()
        if not ids:
            return []
        docs = [_tok(corpus.text_for(cid)) for cid in ids]
        if not any(docs):
            return []
        # Rebuild the BM25 model on every call so its word-rarity and
        # average-length statistics reflect exactly this (possibly filtered) set
        # of chunks, not the whole corpus.
        bm25 = BM25Okapi(docs, k1=self._k1, b=self._b)
        scores = bm25.get_scores(_tok(text))
        scored = sorted(zip(ids, scores, strict=True), key=lambda x: (-float(x[1]), x[0]))
        out: list[Candidate] = []
        for i, (cid, score) in enumerate(scored[:k]):
            if float(score) <= 0:
                break  # list is sorted descending — nothing after this can be positive
            out.append(Candidate(chunk_id=cid, rank=i + 1, score=float(score)))
        return out
