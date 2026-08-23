"""Declarative newest-chart coverage for retrieve-and-prompt recipe steps."""
from __future__ import annotations

from datetime import date
from typing import Any

from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate


def _parse_document_date(value: Any) -> date | None:
    """Return the calendar date from canonical chunk metadata, if present."""
    if not isinstance(value, str) or len(value.strip()) < 10:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def append_recent_candidates(
    candidates: list[Candidate],
    corpus,
    retrieval_cfg: dict[str, Any],
) -> tuple[list[Candidate], list[str]]:
    """Union relevance hits with newest dated chunks requested by the recipe."""
    count = retrieval_cfg.get("include_recent")
    if count is None:
        return candidates, []
    chunk_index = getattr(corpus, "chunk_index", None)
    if chunk_index is None:
        # Lightweight test/adaptor corpora may intentionally expose only row_for.
        # Coverage is additive, so absence of an index leaves relevance hits unchanged.
        return candidates, []

    source_file = retrieval_cfg.get("source_file")
    dated_rows: list[tuple[date, str]] = []
    undated_chunk_ids: list[str] = []
    for row in chunk_index.iter_rows(named=True):
        if source_file and row.get("source_file") != source_file:
            continue
        document_date = _parse_document_date(row.get("document_date"))
        chunk_id = row.get("chunk_id")
        if document_date is not None and isinstance(chunk_id, str):
            dated_rows.append((document_date, chunk_id))
        elif isinstance(chunk_id, str):
            undated_chunk_ids.append(chunk_id)
    dated_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    undated_chunk_ids.sort()

    combined = list(candidates)
    seen = {candidate.chunk_id for candidate in combined}
    added: list[str] = []
    for _document_date, chunk_id in dated_rows:
        if chunk_id in seen:
            continue
        combined.append(
            Candidate(
                chunk_id=chunk_id,
                rank=len(combined) + 1,
                score=0.0,
                retriever="recent_coverage",
            )
        )
        seen.add(chunk_id)
        added.append(chunk_id)
        if len(added) >= int(count):
            break
    if len(added) < int(count):
        for chunk_id in undated_chunk_ids:
            if chunk_id in seen:
                continue
            combined.append(
                Candidate(
                    chunk_id=chunk_id,
                    rank=len(combined) + 1,
                    score=0.0,
                    retriever="recent_coverage_undated_fallback",
                )
            )
            seen.add(chunk_id)
            added.append(chunk_id)
            if len(added) >= int(count):
                break
    return combined, added
