"""BM25 (keyword) and exact-substring retrievers: ranking, the source_file filter,
and the empty-query guard. Both read chunk ids straight off ``corpus.chunk_index``
and the text via ``corpus.text_for`` — the two members a PatientChunkStore exposes,
stubbed here so the test needs no embed/index artifacts and no model.
"""
from __future__ import annotations

import polars as pl

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.bm25.bm25_v2 import BM25Retriever
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.exact_text.exact_text_v1 import (
    ExactRetriever,
)


class _Corpus:
    """Minimal stand-in for PatientChunkStore: only the members the retrievers read."""

    def __init__(self, rows: list[tuple[str, str, str]]):  # (chunk_id, source_file, text)
        self.chunk_index = pl.DataFrame(
            {"chunk_id": [r[0] for r in rows], "source_file": [r[1] for r in rows]}
        )
        self._text = {r[0]: r[2] for r in rows}

    def text_for(self, chunk_id: str) -> str:
        return self._text[chunk_id]


# "carcinoma" is deliberately a MINORITY term within each source_file (1 of 3
# chunks). BM25's IDF turns non-positive when a term appears in every — or half —
# of a tiny document set, which would drop it; keeping it rare per file keeps the
# score positive so the ranking is what the test actually exercises.
_ROWS = [
    ("note:0", "clinical_note.csv", "patient has invasive ductal carcinoma"),
    ("note:1", "clinical_note.csv", "follow up visit, no acute distress"),
    ("note:2", "clinical_note.csv", "blood pressure normal today"),
    ("path:0", "pathology_report.csv", "carcinoma confirmed on pathology, carcinoma grade 2"),
    ("path:1", "pathology_report.csv", "specimen received in formalin, margins clear"),
    ("path:2", "pathology_report.csv", "no evidence of malignancy in this sample"),
]


# ── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_ranks_matches_and_drops_zero_score_chunks():
    hits = BM25Retriever().query(_Corpus(_ROWS), text="carcinoma", k=10)

    ids = [h.chunk_id for h in hits]
    assert set(ids) == {"note:0", "path:0"}         # note:1 has no match → dropped
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)


def test_bm25_source_file_filter_restricts_to_one_file():
    hits = BM25Retriever(source_file="pathology_report.csv").query(
        _Corpus(_ROWS), text="carcinoma", k=10
    )

    assert [h.chunk_id for h in hits] == ["path:0"]


def test_bm25_empty_query_returns_nothing():
    assert BM25Retriever().query(_Corpus(_ROWS), text="   ", k=5) == []


# ── exact substring ──────────────────────────────────────────────────────────

def test_exact_orders_by_hit_count():
    hits = ExactRetriever().query(_Corpus(_ROWS), text="carcinoma", k=10)

    ids = [h.chunk_id for h in hits]
    assert ids[0] == "path:0"          # two occurrences → ranked first
    assert set(ids) == {"note:0", "path:0"}
    assert hits[0].score == 2.0


def test_exact_source_file_filter_restricts_to_one_file():
    hits = ExactRetriever(source_file="clinical_note.csv").query(
        _Corpus(_ROWS), text="carcinoma", k=10
    )

    assert [h.chunk_id for h in hits] == ["note:0"]
