"""Finds chunks containing an exact substring of the query — no scoring model,
just literal matches ranked by hit count."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    RetrieverInfo,
)

_VERSION = "v1"

# ── recipe config options ─────────────────────────────────────────────────────
# Set in your recipe YAML under retrieval:
#
#   kind:             exact
#   k:                10      how many chunks to return (set at the recipe step level)
#   source_file:      (none)  restrict search to one file, e.g. clinical_note.csv
#   case_insensitive: true    match regardless of capitalization (default)
#
# Example:
#   retrieval:
#     kind: exact
#     query: "capecitabine"
#     k: 10
#     source_file: med_orders.csv


@dataclass
class ExactRetriever:
    """Substring-match retriever."""

    info: RetrieverInfo

    def __init__(self, *, case_insensitive: bool = True, source_file: str | None = None):
        self.info = RetrieverInfo(
            kind="exact",
            version=_VERSION,
            config={"case_insensitive": case_insensitive, "source_file": source_file},
        )
        self._ci = case_insensitive
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
        """Return the k chunks where `text` appears most often, most hits first."""
        if not text.strip():
            return []

        needle = text.lower() if self._ci else text
        filtered_chunks = (
            corpus.chunk_index.filter(pl.col("source_file") == self._source_file)
            if self._source_file else corpus.chunk_index
        )
        scored: list[tuple[str, float]] = []
        # Only the chunk_id column is needed; pull it directly rather than
        # materializing every column of every row into dicts.
        for cid in filtered_chunks.get_column("chunk_id").to_list():
            hay = corpus.text_for(cid)
            if not hay:
                continue
            if self._ci:
                hay = hay.lower()
            hits = hay.count(needle)
            if hits > 0:
                scored.append((cid, float(hits)))

        # When two chunks have the same hit count, order them by chunk_id so the
        # ranking is identical every run (reproducible, not arbitrary).
        scored.sort(key=lambda x: (-x[1], x[0]))

        return [
            Candidate(chunk_id=cid, rank=i + 1, score=score)
            for i, (cid, score) in enumerate(scored[:k])
        ]
