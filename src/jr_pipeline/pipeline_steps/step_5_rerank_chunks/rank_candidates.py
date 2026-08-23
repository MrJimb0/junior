"""Turn a pool of candidate chunks into the short list the model reads.

Step 4 (search) hands step 5 the pool for one question; this module decides which few of
them become the evidence, and in what order. The cross-encoder is the one part that lives
elsewhere — it plugs in as a scorer. rerank.py builds the ranker from the recipe's config;
step 7 is what calls it.

Four stages, in order:

  1. filter    Drop candidates that fail the recipe's metadata conditions: only notes
               authored by medical oncology, only documents dated after the diagnosis,
               only outpatient notes. See ``apply_filters`` in filter_candidates.

  2. rank      Score everything that survived and order it best-first. The WHOLE pool is
               ranked here; nothing is trimmed yet.

  3. dedup     Collapse chunks whose text is identical, keeping the copy that ranked
               best, then take the ``top_n``.

  4. re-sort   Optionally lay the chosen top_n back out in date order
               (``chronological_order``), so the model reads the evidence
               chronologically even though relevance is what chose it.

``rank_by`` picks the scorer stage 2 uses:

  * ``cross_encoder`` — the model scorer the shell built.
  * ``combined_score`` — the rule-based weighted sum below. No model, fully reproducible,
    and the default.
  * ``newest_documents`` / ``oldest_documents`` — order by document date, for a recipe
    that wants "the N most recent notes that pass the filters" rather than the N most
    on-topic.

Selection order and reading order are deliberately separate. ``rank_by`` decides WHICH
chunks survive; ``chronological_order`` only decides how the survivors are laid out. The
two compose: let the cross-encoder trim to the best 5, then read those oldest-first.

The combined score is a weighted sum of three signals:

  * retriever_score — 1 / (k0 + prior_rank): how high the chunk ranked in search. k0 is a
    smoothing constant; a larger k0 flattens the gap between ranks. This is the same scale
    the search step's reciprocal rank fusion uses, so the two never disagree on what
    counts as "high".
  * lexical_overlap — fraction of the question's distinct words that also appear in the
    chunk (case-insensitive). A plain keyword-match check that demotes search hits that
    are topically close but never mention the obvious term.
  * source_priority — a per-source-file weight from the recipe: "date of birth lives in
    the demographics file; trust that source".

A signal with no configured weight counts as 0, so adding a signal later never breaks an
older recipe that does not mention it. Equal scores are always broken the same way
(original search rank, then chunk_id), so a re-run on the same inputs produces the same
order.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Any, TypeVar

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.filter_candidates import (
    apply_filters,
    parse_date_interval,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    CandidateScorer,
    RerankedCandidate,
    RerankerInfo,
    RerankInput,
    RerankResult,
    chunk_text_or_empty,
    rank_and_trim,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_bytes,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate, PatientChunkStore

_log = get_logger("rerank")

_VERSION = "v1"

# What a recipe may select the top_n WITH. The cross-encoder choice needs the model
# scorer the shell builds; the other three need nothing but the candidate pool.
RANK_BY_CHOICES = ("cross_encoder", "combined_score", "newest_documents", "oldest_documents")
_DATE_RANK_BY = {"newest_documents": True, "oldest_documents": False}  # -> newest_first

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_DEFAULT_WEIGHTS = {
    "retriever_score": 0.4,
    "lexical_overlap": 0.4,
    "source_priority": 0.2,
}

# The signals the combined score knows how to compute — the vocabulary a recipe's
# ``weights:`` keys must come from. Derived from the defaults so the two cannot drift.
# Step 7 validates a recipe against this at load time, as it does rank_by: scoring reads
# the weight OF EACH SIGNAL, so a weight naming something else would silently do nothing.
COMBINED_SCORE_SIGNALS = tuple(_DEFAULT_WEIGHTS)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def document_day(
    corpus: PatientChunkStore, chunk_id: str, *, newest_first: bool
) -> date | None:
    """The single calendar day to order this chunk by, or None when it has no usable
    document date (the caller puts those last).

    Ingest canonicalizes document dates to ISO, so this normally parses a full date. It
    uses the filters' interval parse anyway, to cover a source system that exports a
    partial date verbatim, then takes the most favourable day that date could mean:
    latest when sorting newest-first, earliest when sorting oldest-first.

    Reads through ``metadata_for``, the same accessor the filters use, so a structured
    table row is ordered by its own document date rather than treated as undated.

    Selecting by date and re-sorting into reading order both resolve their day here, so
    the two cannot disagree about a chunk's date.
    """
    interval = parse_date_interval((corpus.metadata_for(chunk_id) or {}).get("document_date"))
    if interval is None:
        return None
    return interval[1] if newest_first else interval[0]



# Dedup reads only ``chunk_id`` and ``rank``, so it works on the raw search pool or on
# a ranked list. Step 5 runs it on the ranked list; the type stays open so the rule
# cannot drift into two copies if another caller ever needs it earlier.
_Ranked = TypeVar("_Ranked", Candidate, RerankedCandidate)


def _normalized_text(text: str) -> str:
    """Fold a chunk's text to the form duplicates are compared in: one space between
    words, no leading/trailing space, case-insensitive. Two chunks that differ only in
    how the source wrapped or capitalized them are the same evidence."""
    return " ".join(text.split()).casefold()


def drop_duplicate_text(
    candidates: Sequence[_Ranked], corpus: PatientChunkStore
) -> tuple[list[_Ranked], dict[str, Any]]:
    """Keep one candidate per distinct chunk text; return the survivors and a stats row.

    A chart repeats itself: copy-forward notes, one document ingested under two source
    files, a table row restating a note. Two copies of the same text are one piece of
    evidence, so without this they take two ``top_n`` slots and the repetition reads to
    the model as agreement.

    Matching is exact on the normalized text (see ``_normalized_text``), never fuzzy — a
    similarity threshold would drop evidence that was only similar.

    The copy kept is the one that ranked best (lowest ``rank``, ties by chunk_id). That
    is why this runs on the ranked list, not the raw pool: two identical texts can score
    differently when a recipe's ``source_priority`` trusts one file over another, and the
    copy that scoring preferred is the one to keep. Survivors stay in their incoming
    order. A chunk whose text the store cannot resolve reads as empty and is never
    deduplicated, or every unresolvable chunk would collapse into one.

    The stats identify each group by a hash of its text, never by the text.
    """
    texts = [_normalized_text(chunk_text_or_empty(corpus, c.chunk_id)) for c in candidates]
    keeper_of: dict[str, int] = {}
    for i, (candidate, text) in enumerate(zip(candidates, texts, strict=True)):
        if not text:
            continue
        best = keeper_of.get(text)
        if best is None or (candidate.rank, candidate.chunk_id) < (
            candidates[best].rank, candidates[best].chunk_id
        ):
            keeper_of[text] = i

    keepers = set(keeper_of.values())
    survivors = [c for i, c in enumerate(candidates) if not texts[i] or i in keepers]
    dropped_by_text: dict[str, list[str]] = {}
    for i, candidate in enumerate(candidates):
        if texts[i] and i not in keepers:
            dropped_by_text.setdefault(texts[i], []).append(candidate.chunk_id)

    stats = {
        "duplicate_groups": len(dropped_by_text),
        "dropped": sum(len(ids) for ids in dropped_by_text.values()),
        "groups": [
            {
                # sha256 of the normalized text, truncated — enough to tell two groups
                # apart in a receipt without putting chart text in one.
                "text_hash": hash_bytes(text.encode("utf-8"))[7:19],
                "kept": candidates[keeper_of[text]].chunk_id,
                "dropped": ids,
            }
            for text, ids in dropped_by_text.items()
        ],
    }
    return survivors, stats


def resort_by_document_date(
    corpus: PatientChunkStore,
    ranked: list[RerankedCandidate],
    *,
    newest_first: bool = True,
) -> list[RerankedCandidate]:
    """Re-order already-chosen candidates by document date, most recent first, so the
    evidence packet reads chronologically. Each candidate KEEPS its relevance rank.

    Chunks with no date go last. The sort is stable, so within one date the candidates
    keep the relevance order they arrived in. Only the order of the returned list
    changes — ``rank`` stays the relevance (cross-encoder) rank the reranker assigned,
    because the NO_PHI selection trace documents ``rank`` as that rank and
    relevance-label joins consume it as such; renumbering it here by date would corrupt
    the recorded rank. Scores and features are untouched.
    """
    # Only the dated ones go in `days`, so the sort key is never comparing a date
    # against a missing one — which is a TypeError, not a lower rank.
    days = {
        r.chunk_id: day for r in ranked
        if (day := document_day(corpus, r.chunk_id, newest_first=newest_first)) is not None
    }
    with_date = [r for r in ranked if r.chunk_id in days]
    without_date = [r for r in ranked if r.chunk_id not in days]
    with_date.sort(key=lambda r: days[r.chunk_id], reverse=newest_first)  # stable
    return with_date + without_date


class CandidateRanker:
    """Filter the candidate pool, drop duplicate text, select the best top_n, optionally
    re-sort them into reading order.

    ``rank_by`` chooses the scorer: the cross-encoder the shell built, the rule-based
    combined score (the default), or the document date. Filtering, dedup, trimming and
    the optional chronological re-sort happen here whichever one is chosen — a scorer
    only ever answers "what order?".

    ``rerank`` returns a ``RerankResult``: the short list plus the filter and dedup
    counts the receipt needs. Nothing about a call is kept on the instance, so the same
    reranker can be re-run (the filters-fallback path does) without one call's record
    overwriting another's.
    """

    info: RerankerInfo

    def __init__(
        self,
        *,
        cross_encoder_scorer: CandidateScorer | None = None,
        rank_by: str | None = None,
        dedup_identical_text: bool = True,
        weights: dict[str, float] | None = None,
        source_priority: dict[str, float] | None = None,
        k0: int = 60,
        resort_by_date: bool = False,
        chronological_order: str | None = None,
    ) -> None:
        self._cross_encoder_scorer = cross_encoder_scorer
        self._rank_by = self._resolve_rank_by(rank_by, cross_encoder_scorer)
        self._dedup_identical_text = bool(dedup_identical_text)
        self._weights = dict(_DEFAULT_WEIGHTS)
        if weights:
            self._weights.update({k: float(v) for k, v in weights.items()})
        self._source_priority = {
            str(k): float(v) for k, v in (source_priority or {}).items()
        }
        self._k0 = int(k0)
        if chronological_order not in (None, "newest_first", "oldest_first"):
            raise ValueError(
                "chronological_order must be 'newest_first' or 'oldest_first'"
            )
        self._chronological_order = (
            chronological_order
            if chronological_order is not None
            else ("newest_first" if resort_by_date else None)
        )
        self.info = RerankerInfo(
            kind="composed",
            version=_VERSION,
            config={
                # When the cross-encoder is on, carry its full identity (model_path,
                # max_length, batch_size) here, not just a boolean — the reranker
                # fingerprint hashes this config, so two runs using different
                # cross-encoder models must produce different fingerprints. Off is False.
                "cross_encoder": (
                    dict(cross_encoder_scorer.info.config)
                    if cross_encoder_scorer is not None
                    else False
                ),
                # Both of these change which chunks reach the model, so both belong in
                # the fingerprinted config: a run that selected by date must never be
                # mistaken for one that selected by relevance.
                "rank_by": self._rank_by,
                "dedup_identical_text": self._dedup_identical_text,
                "weights": dict(self._weights),
                "source_priority": dict(self._source_priority),
                "k0": self._k0,
                "resort_by_date": self._chronological_order is not None,
                "chronological_order": self._chronological_order,
            },
        )

    @staticmethod
    def _resolve_rank_by(rank_by: str | None, cross_encoder_scorer: CandidateScorer | None) -> str:
        """Settle which selector runs, refusing the two contradictory combinations.

        Unset means what it always meant: the cross-encoder when one was built, the
        combined score otherwise — so a recipe written before ``rank_by`` existed keeps
        its exact behavior. Asking for a scorer that is not there, or building a
        cross-encoder that nothing would call, is a recipe mistake and fails loudly
        rather than quietly ranking by something else (or loading weights for nothing).
        """
        if rank_by is None:
            return "cross_encoder" if cross_encoder_scorer is not None else "combined_score"
        if rank_by not in RANK_BY_CHOICES:
            raise ValueError(f"rank_by must be one of {list(RANK_BY_CHOICES)}, not {rank_by!r}")
        if rank_by == "cross_encoder" and cross_encoder_scorer is None:
            raise ValueError(
                "rank_by 'cross_encoder' needs the cross-encoder: set cross_encoder: true "
                "and model_id in the recipe's reranking block"
            )
        if rank_by != "cross_encoder" and cross_encoder_scorer is not None:
            raise ValueError(
                f"rank_by {rank_by!r} would leave the configured cross-encoder unused — "
                "set cross_encoder: false, or rank_by: cross_encoder to use it"
            )
        return rank_by

    def rerank(
        self,
        corpus: PatientChunkStore,
        inp: RerankInput,
        *,
        top_n: int,
    ) -> RerankResult:
        if not inp.candidates or top_n <= 0:
            return RerankResult(candidates=[], filter_stats=[], dedup_stats={})

        # 1. Filter the pool down to the candidates that match the recipe's
        #    metadata conditions, recording (and warning about) what was dropped.
        survivors, filter_stats = apply_filters(inp.candidates, corpus, inp.filters)
        for s in filter_stats:
            if s["dropped_missing"]:
                _log.warning(
                    "rerank_filter_dropped_missing_field",
                    extra_={
                        "field": s["field"],
                        "op": s["op"],
                        "dropped_for_missing_value": s["dropped_missing"],
                        "dropped_total": s["dropped"],
                    },
                )
            if s["dropped_unparseable"]:
                _log.warning(
                    "rerank_filter_dropped_unparseable_date",
                    extra_={
                        "field": s["field"],
                        "op": s["op"],
                        "dropped_for_unparseable_date": s["dropped_unparseable"],
                        "example_value": s["unparseable_example"],
                        "dropped_total": s["dropped"],
                    },
                )
        # An empty pool is a real answer, and filter_stats is exactly the record of why —
        # it rides back with it rather than being left on the object for the caller to find.
        if not survivors:
            return RerankResult(candidates=[], filter_stats=filter_stats, dedup_stats={})

        # 2. Rank everything that survived the filters — the whole pool, not top_n.
        #    Trimming waits until after dedup, so a duplicate can never consume a slot
        #    that a distinct document should have had. Ranking the full pool costs
        #    nothing extra: every scorer already scores each candidate before trimming.
        rank_all = len(survivors)
        scored_input = RerankInput(candidates=survivors, query_text=inp.query_text)
        if self._rank_by == "cross_encoder":
            # _resolve_rank_by refuses "cross_encoder" without a scorer and refuses a
            # scorer without "cross_encoder", and the constructor is the only writer of
            # either — so reaching here means the scorer is there.
            assert self._cross_encoder_scorer is not None
            ranked = self._cross_encoder_scorer.rerank(corpus, scored_input, top_n=rank_all)
        elif self._rank_by == "combined_score":
            ranked = self._combined_score(corpus, scored_input, top_n=rank_all)
        else:
            ranked = self._by_document_date(
                corpus, scored_input, top_n=rank_all,
                newest_first=_DATE_RANK_BY[self._rank_by],
            )

        # 3. Collapse repeated text, keeping the copy that ranked best, then take the
        #    top_n and renumber so the packet reads 1..n with no gaps where a duplicate
        #    was removed.
        dedup_stats: dict[str, Any] = {}
        if self._dedup_identical_text:
            ranked, dedup_stats = drop_duplicate_text(ranked, corpus)
            if dedup_stats["dropped"]:
                _log.info(
                    "rerank_dropped_duplicate_text",
                    extra_={
                        "dropped": dedup_stats["dropped"],
                        "duplicate_groups": dedup_stats["duplicate_groups"],
                        "candidates_remaining": len(ranked),
                    },
                )
        ranked = [replace(r, rank=i + 1) for i, r in enumerate(ranked[:top_n])]

        # 4. Optionally re-order the chosen chunks chronologically for the model.
        if self._chronological_order is not None:
            ranked = resort_by_document_date(
                corpus,
                ranked,
                newest_first=self._chronological_order == "newest_first",
            )
        return RerankResult(candidates=ranked, filter_stats=filter_stats, dedup_stats=dedup_stats)

    def _combined_score(
        self,
        corpus: PatientChunkStore,
        inp: RerankInput,
        *,
        top_n: int,
    ) -> list[RerankedCandidate]:
        """Rule-based scoring: weighted sum of the explainable signals, best first."""
        query_tokens = _tokenize(inp.query_text)
        scored: list[RerankedCandidate] = []
        for c in inp.candidates:
            features = self._features(c, corpus, query_tokens)
            score = sum(self._weights.get(k, 0.0) * v for k, v in features.items())
            scored.append(
                RerankedCandidate(
                    chunk_id=c.chunk_id,
                    rank=0,
                    score=score,
                    prior_rank=c.rank,
                    features=features,
                )
            )
        return rank_and_trim(scored, top_n)

    @staticmethod
    def _by_document_date(
        corpus: PatientChunkStore,
        inp: RerankInput,
        *,
        top_n: int,
        newest_first: bool,
    ) -> list[RerankedCandidate]:
        """Select by document date: the ``top_n`` most recent (or oldest) chunks in the
        pool, for a recipe whose question is "what happened last", not "what is on topic".

        Undated chunks go last, keeping their search order, so they are taken only to
        fill slots the dated ones left. Within one date the search order stands.

        The score is the candidate's POSITION in that date order (1, 1/2, 1/3, ...), never
        the date itself — deliberately. Scores and features flow into the NO_PHI selection
        trace, which carries no dates by design; a date-derived score would export the
        service dates of a patient's chart into the shareable exhaust. A positional score
        adds nothing to what ``rank`` already says, and keeps the ordering total so
        ``rank_and_trim`` reproduces exactly this order.
        """
        # Only the dated ones go in `days` — see resort_by_document_date: a sort key that
        # can return "no date" raises rather than ordering.
        days = {
            c.chunk_id: day for c in inp.candidates
            if (day := document_day(corpus, c.chunk_id, newest_first=newest_first)) is not None
        }
        dated = [c for c in inp.candidates if c.chunk_id in days]
        undated = [c for c in inp.candidates if c.chunk_id not in days]
        dated.sort(key=lambda c: days[c.chunk_id], reverse=newest_first)  # stable
        scored = [
            RerankedCandidate(
                chunk_id=c.chunk_id,
                rank=0,
                score=1.0 / (i + 1),
                prior_rank=c.rank,
                features={"document_recency": round(1.0 / (i + 1), 6)},
            )
            for i, c in enumerate(dated + undated)
        ]
        return rank_and_trim(scored, top_n)

    def _features(
        self,
        c: Candidate,
        corpus: PatientChunkStore,
        query_tokens: set[str],
    ) -> dict[str, float]:
        retriever_score = 1.0 / (self._k0 + max(c.rank, 1))

        chunk_tokens = _tokenize(chunk_text_or_empty(corpus, c.chunk_id))
        if query_tokens:
            lexical_overlap = (
                len(query_tokens & chunk_tokens) / len(query_tokens)
                if chunk_tokens
                else 0.0
            )
        else:
            lexical_overlap = 0.0

        source_file = self._source_file_for(c.chunk_id, corpus)
        source_priority = self._source_priority.get(source_file or "", 0.0)

        return {
            "retriever_score": retriever_score,
            "lexical_overlap": lexical_overlap,
            "source_priority": source_priority,
        }

    @staticmethod
    def _source_file_for(chunk_id: str, corpus: PatientChunkStore) -> str | None:
        # metadata_for carries source_file for both kinds of chunk: an ordinary text
        # chunk reads it off the chunk index, a ":TABLE" row (a whole structured-table
        # row treated as one piece of evidence) off the ingest sidecar.
        return (corpus.metadata_for(chunk_id) or {}).get("source_file")
