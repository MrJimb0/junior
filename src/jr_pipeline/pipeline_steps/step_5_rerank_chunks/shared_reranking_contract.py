"""The record types and contracts every reranker shares.

Plus the two things no scorer decides for itself: ``rank_and_trim``, so every scorer
breaks ties the same way, and ``chunk_text_or_empty``, so an unresolvable chunk id reads
as empty text everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol

from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate, PatientChunkStore


@dataclass(frozen=True)
class EvidenceFilter:
    """One metadata condition a candidate chunk must satisfy to be kept.

    ``field`` and ``op`` are the vocabulary filter_candidates defines and step 7 validates
    a recipe against. ``value`` is already a plain literal — any ``{{ vars.* }}`` a recipe
    wrote was resolved upstream, so the reranker does no templating. ``keep_if_missing``
    keeps a candidate that has no value for ``field``; it defaults False, because a filter
    is a gate.
    """

    field: str
    op: str
    value: Any
    keep_if_missing: bool = False


@dataclass(frozen=True)
class RerankInput:
    """Everything a single rerank call needs: one patient, one question, and its candidate chunks."""

    candidates: list[Candidate]
    query_text: str
    # Metadata filters to apply before scoring; empty means keep every candidate.
    filters: tuple[EvidenceFilter, ...] = ()


@dataclass(frozen=True)
class RerankedCandidate:
    """A candidate chunk after reranking; ``features`` records each signal's contribution to ``score``."""

    chunk_id: str
    rank: int  # final position after reranking, starting at 1 for the best
    score: float
    prior_rank: int
    features: dict[str, float]


@dataclass(frozen=True)
class RerankerInfo:
    """A record of which reranker ran and how it was configured, written into reranked.json under 'reranker'."""

    kind: str
    version: str
    config: dict[str, Any]


@dataclass(frozen=True)
class RerankResult:
    """What one rerank call produced: the short list, plus what was dropped to get it.

    The counts ride back with the result rather than being stored on the reranker because
    the same reranker can be called twice — the fallback-to-unfiltered path does — and the
    second call would overwrite the first's.

      candidates    the chosen chunks, ranked 1..n, best first (empty is a valid answer)
      filter_stats  per filter: how many candidates it dropped, and how many of those for
                    a missing or unparseable field
      dedup_stats   duplicate groups collapsed, and which chunk each one deferred to;
                    empty when dedup was off
    """

    candidates: list[RerankedCandidate]
    filter_stats: list[dict[str, Any]]
    dedup_stats: dict[str, Any]


class CandidateScorer(Protocol):
    """What a scorer must provide. It answers one question: given this pool and this
    query, what order?

    Filtering, duplicate removal and trimming to ``top_n`` belong to the
    ``CandidateRanker`` that calls it, so a new scorer cannot quietly diverge on any of
    them.
    """

    info: RerankerInfo

    def rerank(
        self,
        corpus: PatientChunkStore,
        inp: RerankInput,
        *,
        top_n: int,
    ) -> list[RerankedCandidate]:
        """Re-score the candidate chunks and return the best top_n, ordered best-first."""


def rank_and_trim(
    scored: list[RerankedCandidate],
    top_n: int,
    *,
    rounding: int | None = None,
) -> list[RerankedCandidate]:
    """Order scored candidates best-first, keep ``top_n``, stamp final ranks 1, 2, 3 ...

    The ordering contract every scorer shares: highest score first, ties broken by the
    original search rank and then chunk_id, so the same inputs always produce the same
    list.

    ``rounding`` quantizes the score for the COMPARISON only, never the stored one — the
    cross-encoder rounds its logits so float noise past the 6th decimal cannot reorder two
    otherwise-tied chunks between runs.
    """
    def order(r: RerankedCandidate) -> tuple[float, int, str]:
        score = r.score if rounding is None else round(r.score, rounding)
        return (-score, r.prior_rank, r.chunk_id)

    return [
        replace(r, rank=i + 1)
        for i, r in enumerate(sorted(scored, key=order)[:top_n])
    ]


def chunk_text_or_empty(corpus: PatientChunkStore, chunk_id: str) -> str:
    """A chunk's text, or ``""`` when the store cannot resolve it.

    An id the store no longer knows (a stale candidate, a table row that vanished) must
    not sink the whole rerank call — it scores as a chunk with no text and ranks last on
    its own merits.
    """
    try:
        return corpus.text_for(chunk_id) or ""
    except (KeyError, IndexError, FileNotFoundError):
        return ""
