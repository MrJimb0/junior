"""The candidate ranker: filtering, duplicate-text removal, selection, date re-sort.

Scoring math (the rule-based combined score) is pinned in test_rerank.py; this file
pins the rest of step 5 — the filters, the dedup pass, selecting by document date,
and the optional chronological re-sort — plus the recipe-spec guards around them.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.cross_encoder import CrossEncoderReranker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.filter_candidates import apply_filters
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import (
    CandidateRanker,
    drop_duplicate_text,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import build_reranker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    EvidenceFilter,
    RerankInput,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    _parse_reranking,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate


class _FakeCorpus:
    """chunk_id -> its chart metadata (author/doc_type/document_date/...), however the
    store resolved it. A chunk_id mapped to None reads as unresolvable — the store could
    not place it, so every field is absent."""

    def __init__(self, rows: dict[str, dict | None], texts: dict[str, str] | None = None):
        self._rows = rows
        self._texts = texts or {}

    def metadata_for(self, chunk_id: str):
        return self._rows.get(chunk_id)

    def text_for(self, chunk_id: str) -> str:
        return self._texts.get(chunk_id, f"text::{chunk_id}")


def _cand(chunk_id, rank):
    return Candidate(chunk_id=chunk_id, rank=rank, score=1.0 / rank, retriever="bm25")


# ---- filters ---------------------------------------------------------------

def test_filter_author_contains_is_case_insensitive_substring():
    corpus = _FakeCorpus({
        "a": {"author": "Smith, MD - Medical Oncology"},
        "b": {"author": "Jones, MD - Radiology"},
    })
    survivors, stats = apply_filters(
        [_cand("a", 1), _cand("b", 2)],
        corpus,
        (EvidenceFilter(field="author", op="contains", value="medical oncology"),),
    )
    assert [c.chunk_id for c in survivors] == ["a"]
    assert stats[0]["dropped"] == 1 and stats[0]["dropped_missing"] == 0


def test_filter_doc_type_equality_and_in():
    corpus = _FakeCorpus({
        "a": {"doc_type": "Outpatient"},
        "b": {"doc_type": "inpatient"},
        "c": {"doc_type": "progress_note"},
    })
    eq, _ = apply_filters(
        [_cand("a", 1), _cand("b", 2)],
        corpus,
        (EvidenceFilter(field="doc_type", op="==", value="outpatient"),),
    )
    assert [c.chunk_id for c in eq] == ["a"]
    in_, _ = apply_filters(
        [_cand("a", 1), _cand("b", 2), _cand("c", 3)],
        corpus,
        (EvidenceFilter(field="doc_type", op="in", value=["outpatient", "progress_note"]),),
    )
    assert [c.chunk_id for c in in_] == ["a", "c"]


def test_filter_date_after_threshold_drops_earlier_and_undated():
    corpus = _FakeCorpus({
        "before": {"document_date": "2019-05-01"},
        "after": {"document_date": "2021-07-02"},
        "undated": {"document_date": None},
    })
    survivors, stats = apply_filters(
        [_cand("before", 1), _cand("after", 2), _cand("undated", 3)],
        corpus,
        (EvidenceFilter(field="document_date", op=">", value="2020-01-01"),),
    )
    assert [c.chunk_id for c in survivors] == ["after"]
    # one dropped for being earlier, one dropped for having no date
    assert stats[0]["dropped"] == 2 and stats[0]["dropped_missing"] == 1


def test_filter_missing_field_dropped_by_default_kept_with_keep_if_missing():
    corpus = _FakeCorpus({
        "has": {"author": "Dr Medical Oncology"},
        "missing": {"author": None},
        "table": None,  # :TABLE-style row, no chunk_index metadata at all
    })
    cands = [_cand("has", 1), _cand("missing", 2), _cand("table", 3)]

    dropped, stats = apply_filters(
        cands, corpus, (EvidenceFilter(field="author", op="contains", value="oncology"),)
    )
    assert [c.chunk_id for c in dropped] == ["has"]
    assert stats[0]["dropped"] == 2 and stats[0]["dropped_missing"] == 2

    kept, _ = apply_filters(
        cands,
        corpus,
        (EvidenceFilter(field="author", op="contains", value="oncology", keep_if_missing=True),),
    )
    assert [c.chunk_id for c in kept] == ["has", "missing", "table"]


def test_partial_target_date_keeps_candidates_that_could_satisfy():
    # The target is a partial diagnosis date (year only). Under possibly-satisfies, a
    # full-date candidate is kept whenever any day in the target year could satisfy >=,
    # so only the strictly-earlier 2019 note is dropped.
    corpus = _FakeCorpus({
        "before": {"document_date": "2019-06-01"},
        "in_year": {"document_date": "2020-06-01"},
        "after": {"document_date": "2021-06-01"},
    })
    survivors, _ = apply_filters(
        [_cand("before", 1), _cand("in_year", 2), _cand("after", 3)],
        corpus,
        (EvidenceFilter(field="document_date", op=">=", value="2020-XX-XX"),),
    )
    assert [c.chunk_id for c in survivors] == ["in_year", "after"]


def test_partial_candidate_date_kept_when_it_might_satisfy():
    # A partial CANDIDATE date is kept when it could satisfy a full-date target: the
    # document could be as late as 2020-12-31, which is >= 2020-07-01.
    corpus = _FakeCorpus({"partial": {"document_date": "2020-XX-XX"}})
    survivors, _ = apply_filters(
        [_cand("partial", 1)],
        corpus,
        (EvidenceFilter(field="document_date", op=">=", value="2020-07-01"),),
    )
    assert [c.chunk_id for c in survivors] == ["partial"]


def test_partial_dates_equality_uses_interval_overlap():
    # == on partial dates compares intervals: a 2020 month overlaps the 2020 target
    # year and is kept; a 2021 month is disjoint and dropped.
    corpus = _FakeCorpus({
        "same_year": {"document_date": "2020-03-XX"},
        "other_year": {"document_date": "2021-03-XX"},
    })
    survivors, _ = apply_filters(
        [_cand("same_year", 1), _cand("other_year", 2)],
        corpus,
        (EvidenceFilter(field="document_date", op="==", value="2020-XX-XX"),),
    )
    assert [c.chunk_id for c in survivors] == ["same_year"]


def test_partial_date_not_equal_and_strict_boundary_operators():
    # The interval branches the happy-path tests don't reach: != and the >= vs > boundary.
    corpus = _FakeCorpus({
        "exact": {"document_date": "2020-06-15"},
        "partial": {"document_date": "2020-XX-XX"},
        "jan1": {"document_date": "2020-01-01"},
    })

    def survivors(op, value, ids):
        keep, _ = apply_filters(
            [_cand(i, r) for r, i in enumerate(ids, 1)], corpus,
            (EvidenceFilter(field="document_date", op=op, value=value),),
        )
        return {c.chunk_id for c in keep}

    # != drops a candidate that can only equal the target, keeps one that might differ
    assert survivors("!=", "2020-06-15", ["exact", "partial"]) == {"partial"}
    # partial-year target: >= keeps the year's first day (candidate_latest >= target_earliest),
    # > drops it (must strictly exceed the earliest possible target day)
    assert survivors(">=", "2020-XX-XX", ["jan1"]) == {"jan1"}
    assert survivors(">", "2020-XX-XX", ["jan1"]) == set()


def test_unparseable_date_is_counted_apart_from_a_missing_field():
    # A present-but-garbage date is dropped like a missing field, but recorded in its
    # own counter so a reviewer can tell "no date" from "unparseable date".
    corpus = _FakeCorpus({"garbled": {"document_date": "not-a-date"}})
    survivors, stats = apply_filters(
        [_cand("garbled", 1)],
        corpus,
        (EvidenceFilter(field="document_date", op=">=", value="2020-01-01"),),
    )
    assert survivors == []
    assert stats[0]["dropped_unparseable"] == 1
    assert stats[0]["dropped_missing"] == 0
    assert stats[0]["unparseable_example"] == "not-a-date"


def test_a_number_field_compares_numerically_not_as_text():
    # The whole point of the field's kind: as text, "9" sorts above "18" and an age
    # cohort silently keeps the wrong patients.
    corpus = _FakeCorpus({
        "child": {"age": "9"},
        "adult": {"age": "18"},
        "older": {"age": "62.4"},
    })
    survivors, _ = apply_filters(
        [_cand("child", 1), _cand("adult", 2), _cand("older", 3)],
        corpus,
        (EvidenceFilter(field="age", op=">=", value=18),),
    )
    assert [c.chunk_id for c in survivors] == ["adult", "older"]


def test_a_number_compares_the_same_however_the_site_typed_its_column():
    # 62, "62", 62.0 and "62.0" are one age; a recipe must not depend on the export.
    for stored in ("62", "62.0", 62, 62.0):
        corpus = _FakeCorpus({"c": {"age": stored}})
        for wanted in (62, "62", 62.0):
            survivors, _ = apply_filters(
                [_cand("c", 1)], corpus,
                (EvidenceFilter(field="age", op="==", value=wanted),),
            )
            assert [x.chunk_id for x in survivors] == ["c"], (stored, wanted)


def test_a_non_numeric_age_is_unparseable_not_a_silent_keep():
    # A null numeric column reaches the filter as the text "nan". It is not an age, so it
    # is recorded as unparseable and keep_if_missing decides — never compared as text.
    corpus = _FakeCorpus({"blank": {"age": "nan"}, "words": {"age": "unknown"}})
    survivors, stats = apply_filters(
        [_cand("blank", 1), _cand("words", 2)],
        corpus,
        (EvidenceFilter(field="age", op=">=", value=18),),
    )
    assert survivors == []
    assert stats[0]["dropped_unparseable"] == 2


def test_an_identifier_that_looks_like_a_number_is_still_text():
    # encounter_id is text by kind, so it compares verbatim: 007 is not 7.
    corpus = _FakeCorpus({"a": {"encounter_id": "007"}})
    survivors, _ = apply_filters(
        [_cand("a", 1)], corpus,
        (EvidenceFilter(field="encounter_id", op="==", value="7"),),
    )
    assert survivors == []


def test_filters_are_anded():
    corpus = _FakeCorpus({
        "a": {"author": "Onc", "doc_type": "outpatient"},
        "b": {"author": "Onc", "doc_type": "inpatient"},
        "c": {"author": "Rad", "doc_type": "outpatient"},
    })
    survivors, _ = apply_filters(
        [_cand("a", 1), _cand("b", 2), _cand("c", 3)],
        corpus,
        (
            EvidenceFilter(field="author", op="contains", value="onc"),
            EvidenceFilter(field="doc_type", op="==", value="outpatient"),
        ),
    )
    assert [c.chunk_id for c in survivors] == ["a"]


# ---- dedup: identical chunk text -------------------------------------------

def test_dedup_keeps_the_best_ranked_copy_and_preserves_pool_order():
    # Same note reaching the pool three times (copy-forward / ingested twice). The copy
    # kept is the one search ranked highest; the survivors stay in pool order.
    corpus = _FakeCorpus(
        {"first": {}, "dup_low": {}, "other": {}},
        texts={"first": "PET shows no residual disease.",
               "dup_low": "PET shows no residual disease.",
               "other": "Margins negative."},
    )
    survivors, stats = drop_duplicate_text(
        [_cand("dup_low", 4), _cand("first", 2), _cand("other", 3)], corpus
    )

    assert [c.chunk_id for c in survivors] == ["first", "other"]  # pool order kept
    assert stats["dropped"] == 1 and stats["duplicate_groups"] == 1
    assert stats["groups"][0]["kept"] == "first"
    assert stats["groups"][0]["dropped"] == ["dup_low"]


def test_dedup_matches_only_after_whitespace_and_case_folding():
    # Same text re-wrapped and re-cased is the same evidence; a text that differs by a
    # real word is not, and both survive.
    corpus = _FakeCorpus(
        {"a": {}, "b": {}, "c": {}},
        texts={"a": "Stage  IIA\ninvasive ductal", "b": "stage iia invasive ductal",
               "c": "Stage IIB invasive ductal"},
    )
    survivors, stats = drop_duplicate_text([_cand("a", 1), _cand("b", 2), _cand("c", 3)], corpus)

    assert [c.chunk_id for c in survivors] == ["a", "c"]
    assert stats["dropped"] == 1


def test_dedup_never_collapses_chunks_whose_text_is_unresolvable():
    # Empty text is "we could not read it", not "they are the same" — collapsing those
    # would silently drop unrelated evidence.
    corpus = _FakeCorpus({"x": {}, "y": {}}, texts={"x": "", "y": ""})
    survivors, stats = drop_duplicate_text([_cand("x", 1), _cand("y", 2)], corpus)

    assert [c.chunk_id for c in survivors] == ["x", "y"]
    assert stats["dropped"] == 0


def test_dedup_stats_identify_a_group_by_hash_not_by_chart_text():
    corpus = _FakeCorpus({"a": {}, "b": {}}, texts={"a": "Patient tolerated chemo well.",
                                                    "b": "Patient tolerated chemo well."})
    _, stats = drop_duplicate_text([_cand("a", 1), _cand("b", 2)], corpus)

    text_hash = stats["groups"][0]["text_hash"]
    assert "chemo" not in text_hash and len(text_hash) == 12


def test_duplicates_do_not_consume_top_n_slots():
    # The point of running dedup before selection: two copies of one note must not take
    # two of the three slots and crowd out a distinct third document.
    corpus = _FakeCorpus(
        {"a": {}, "a_copy": {}, "b": {}},
        texts={"a": "alpha finding", "a_copy": "alpha finding", "b": "alpha other"},
    )
    reranker = build_reranker({
        "weights": {"retriever_score": 1.0, "lexical_overlap": 0.0, "source_priority": 0.0},
    })
    result = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("a", 1), _cand("a_copy", 2), _cand("b", 3)], query_text="alpha"),
        top_n=2,
    )

    assert [c.chunk_id for c in result.candidates] == ["a", "b"]
    assert result.dedup_stats["dropped"] == 1


def test_dedup_keeps_the_copy_THE_RANKING_preferred_not_the_best_retrieved():
    # Why dedup runs on the ranked list: the same text sits in two files, and the recipe
    # trusts one of them (source_priority). Search liked the note copy better, but the
    # recipe's own scoring prefers the demographics copy — that is the one to keep.
    corpus = _FakeCorpus(
        {"from_note": {"source_file": "clinical_note.csv"},
         "from_demographics": {"source_file": "demographics.csv"}},
        texts={"from_note": "DOB 1957-04-15", "from_demographics": "DOB 1957-04-15"},
    )
    reranker = build_reranker({
        "weights": {"retriever_score": 0.0, "lexical_overlap": 0.0, "source_priority": 1.0},
        "source_priority": {"demographics.csv": 1.0, "clinical_note.csv": 0.1},
    })
    result = reranker.rerank(
        corpus,
        RerankInput(
            # from_note retrieved FIRST; dedup on the raw pool would have kept it
            candidates=[_cand("from_note", 1), _cand("from_demographics", 2)],
            query_text="dob",
        ),
        top_n=5,
    )

    assert [c.chunk_id for c in result.candidates] == ["from_demographics"]
    assert result.dedup_stats["groups"][0]["dropped"] == ["from_note"]


def test_trimming_happens_after_dedup_so_ranks_have_no_gaps():
    # A duplicate at rank 2 must not leave a hole in the packet numbering, and must not
    # cost the third distinct document its slot.
    corpus = _FakeCorpus(
        {"a": {}, "a_copy": {}, "b": {}},
        texts={"a": "alpha", "a_copy": "alpha", "b": "beta"},
    )
    reranker = build_reranker({
        "weights": {"retriever_score": 1.0, "lexical_overlap": 0.0, "source_priority": 0.0},
    })
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("a", 1), _cand("a_copy", 2), _cand("b", 3)], query_text="q"),
        top_n=2,
    ).candidates

    assert [c.chunk_id for c in out] == ["a", "b"]
    assert [c.rank for c in out] == [1, 2]      # renumbered, not 1 and 3
    assert [c.prior_rank for c in out] == [1, 3]  # search rank is still the real one


def test_dedup_can_be_turned_off():
    corpus = _FakeCorpus({"a": {}, "a_copy": {}}, texts={"a": "same", "a_copy": "same"})
    reranker = build_reranker({"dedup_identical_text": False})
    result = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("a", 1), _cand("a_copy", 2)], query_text="same"),
        top_n=5,
    )

    assert {c.chunk_id for c in result.candidates} == {"a", "a_copy"}
    assert result.dedup_stats == {}


# ---- selecting by document date --------------------------------------------

def test_rank_by_newest_documents_selects_the_most_recent_not_the_most_relevant():
    # The whole gap this closes: date decides WHICH chunks survive, not just how the
    # chosen ones are laid out. Search liked "old" best; newest_documents overrules it.
    corpus = _FakeCorpus({
        "old": {"document_date": "2019-01-01"},
        "mid": {"document_date": "2022-06-01"},
        "new": {"document_date": "2024-03-01"},
    })
    reranker = build_reranker({"rank_by": "newest_documents"})
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("old", 1), _cand("mid", 2), _cand("new", 3)], query_text="q"),
        top_n=2,
    ).candidates

    assert [c.chunk_id for c in out] == ["new", "mid"]
    assert [c.rank for c in out] == [1, 2]


def test_rank_by_oldest_documents_selects_from_the_other_end():
    corpus = _FakeCorpus({
        "old": {"document_date": "2019-01-01"},
        "new": {"document_date": "2024-03-01"},
    })
    reranker = build_reranker({"rank_by": "oldest_documents"})
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("new", 1), _cand("old", 2)], query_text="q"),
        top_n=1,
    ).candidates

    assert [c.chunk_id for c in out] == ["old"]


def test_date_selection_puts_undated_last_and_keeps_search_order_within_a_date():
    corpus = _FakeCorpus({
        "same_a": {"document_date": "2024-01-01"},
        "same_b": {"document_date": "2024-01-01"},
        "undated": {"document_date": None},
    })
    reranker = build_reranker({"rank_by": "newest_documents"})
    out = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("same_b", 1), _cand("same_a", 2), _cand("undated", 3)],
            query_text="q",
        ),
        top_n=3,
    ).candidates

    # equal dates keep the order search gave them; the undated chunk fills the last slot
    assert [c.chunk_id for c in out] == ["same_b", "same_a", "undated"]


def test_date_selection_scores_are_positional_and_carry_no_date():
    # Scores and features flow into the NO_PHI selection trace, which carries no dates.
    # A date-derived score would export service dates; a positional one cannot.
    corpus = _FakeCorpus({
        "new": {"document_date": "2024-03-01"},
        "old": {"document_date": "1999-12-31"},
    })
    reranker = build_reranker({"rank_by": "newest_documents"})
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("old", 1), _cand("new", 2)], query_text="q"),
        top_n=2,
    ).candidates

    assert [c.score for c in out] == [1.0, 0.5]
    assert [set(c.features) for c in out] == [{"document_recency"}, {"document_recency"}]


def test_partial_document_date_is_placed_by_its_interval_not_treated_as_undated():
    # A year-only note could be as late as 2020-12-31, so newest-first ranks it above a
    # dated June note; undated chunks still go last.
    corpus = _FakeCorpus({
        "june": {"document_date": "2020-06-01"},
        "year_only": {"document_date": "2020-XX-XX"},
        "undated": {"document_date": None},
    })
    reranker = build_reranker({"rank_by": "newest_documents"})
    out = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("june", 1), _cand("year_only", 2), _cand("undated", 3)],
            query_text="q",
        ),
        top_n=3,
    ).candidates

    assert [c.chunk_id for c in out] == ["year_only", "june", "undated"]


def test_cross_encoder_trim_then_chronological_reading_order():
    # Selection and reading order are independent: keep the 2 newest documents, then lay
    # them out oldest-first for the model. Each keeps its selection rank.
    corpus = _FakeCorpus({
        "old": {"document_date": "2019-01-01"},
        "mid": {"document_date": "2022-06-01"},
        "new": {"document_date": "2024-03-01"},
    })
    reranker = build_reranker({
        "rank_by": "newest_documents", "chronological_order": "oldest_first",
    })
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("old", 1), _cand("mid", 2), _cand("new", 3)], query_text="q"),
        top_n=2,
    ).candidates

    assert [c.chunk_id for c in out] == ["mid", "new"]   # read oldest-first
    assert [c.rank for c in out] == [2, 1]               # selection rank preserved


# ---- rank_by / cross-encoder wiring guards ---------------------------------

def test_rank_by_defaults_preserve_pre_existing_behavior():
    assert CandidateRanker().info.config["rank_by"] == "combined_score"
    assert CandidateRanker(
        cross_encoder_scorer=CrossEncoderReranker(model_path="/models/rer")
    ).info.config["rank_by"] == "cross_encoder"


def test_rank_by_cross_encoder_without_a_cross_encoder_is_refused():
    with pytest.raises(ValueError, match="needs the cross-encoder"):
        build_reranker({"rank_by": "cross_encoder"})


def test_a_cross_encoder_that_nothing_would_call_is_refused():
    # Loading reranker weights that no scorer uses is a recipe mistake, not a default.
    with pytest.raises(ValueError, match="would leave the configured cross-encoder unused"):
        CandidateRanker(
            cross_encoder_scorer=CrossEncoderReranker(model_path="/models/rer"),
            rank_by="newest_documents",
        )


def test_unknown_rank_by_is_refused():
    with pytest.raises(ValueError, match="rank_by must be one of"):
        build_reranker({"rank_by": "vibes"})


# ---- end-to-end through CandidateRanker -----------------------------------

def test_compose_filters_then_scores_then_selects_top_n():
    corpus = _FakeCorpus({
        "keep1": {"doc_type": "outpatient"},
        "keep2": {"doc_type": "outpatient"},
        "drop": {"doc_type": "inpatient"},
    })
    reranker = build_reranker({})  # cross-encoder off; rule-based combined score
    result = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("keep1", 1), _cand("drop", 2), _cand("keep2", 3)],
            query_text="q",
            filters=(EvidenceFilter(field="doc_type", op="==", value="outpatient"),),
        ),
        top_n=10,
    )
    assert set(c.chunk_id for c in result.candidates) == {"keep1", "keep2"}  # inpatient filtered out
    assert result.filter_stats[0]["dropped"] == 1


def test_resort_by_date_reorders_list_but_keeps_relevance_rank():
    # The date re-sort changes the packet ORDER (newest document first, undated last)
    # but must NOT renumber rank: rank stays the relevance rank the reranker assigned,
    # which the NO_PHI selection trace records as the cross-encoder rank. Relevance
    # order here is old(1), undated(2), new(3); after the chronological re-sort the list
    # is [new, old, undated] but each keeps its relevance rank -> [3, 1, 2].
    corpus = _FakeCorpus({
        "old": {"document_date": "2019-01-01"},
        "new": {"document_date": "2024-06-01"},
        "undated": {"document_date": None},
    })
    reranker = build_reranker({"resort_by_date": True})
    out = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("old", 1), _cand("undated", 2), _cand("new", 3)],
            query_text="q",
        ),
        top_n=10,
    ).candidates
    assert [c.chunk_id for c in out] == ["new", "old", "undated"]
    assert [c.rank for c in out] == [3, 1, 2]


def test_resort_by_date_equal_dates_keep_relevance_order_undated_last():
    # The date re-sort must be STABLE: chunks with the SAME date keep the relevance
    # order they arrived in, and undated chunks go last. (Reproducibility guard,
    # previously covered by the deleted newest_document_first tiebreak test.)
    corpus = _FakeCorpus({
        "z": {"document_date": "2024-01-01"},
        "a": {"document_date": "2024-01-01"},  # same date as z
        "u": {"document_date": None},          # undated -> last
    })
    # retriever-only weights => relevance order is exactly the search rank order.
    reranker = build_reranker({
        "resort_by_date": True,
        "weights": {"retriever_score": 1.0, "lexical_overlap": 0.0, "source_priority": 0.0},
    })
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("a", 1), _cand("z", 2), _cand("u", 3)], query_text="q"),
        top_n=10,
    ).candidates
    assert [c.chunk_id for c in out] == ["a", "z", "u"]
    assert [c.rank for c in out] == [1, 2, 3]


def test_chronological_order_can_read_oldest_first():
    corpus = _FakeCorpus({
        "new": {"document_date": "2024-01-01"},
        "old": {"document_date": "2020-01-01"},
        "undated": {"document_date": None},
    })
    reranker = build_reranker({"chronological_order": "oldest_first"})
    out = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("new", 1), _cand("old", 2), _cand("undated", 3)],
            query_text="q",
        ),
        top_n=3,
    ).candidates
    assert [c.chunk_id for c in out] == ["old", "new", "undated"]


def test_retriever_only_weights_reproduce_retrieval_order():
    # The old "passthrough" reranker is now the degenerate combined score: weight
    # only retriever_score and the order is exactly the search order.
    corpus = _FakeCorpus({"c": {}, "a": {}, "b": {}})
    reranker = build_reranker(
        {"weights": {"retriever_score": 1.0, "lexical_overlap": 0.0, "source_priority": 0.0}}
    )
    out = reranker.rerank(
        corpus,
        RerankInput(candidates=[_cand("c", 1), _cand("a", 2), _cand("b", 3)], query_text="q"),
        top_n=2,
    ).candidates
    assert [c.chunk_id for c in out] == ["c", "a"]  # retrieval order, top 2


def test_empty_or_nonpositive_top_n():
    corpus = _FakeCorpus({"x": {}})
    reranker = build_reranker({})
    assert reranker.rerank(corpus, RerankInput(candidates=[], query_text="q"), top_n=5).candidates == []
    assert reranker.rerank(corpus, RerankInput(candidates=[_cand("x", 1)], query_text="q"), top_n=0).candidates == []


def test_filter_that_drops_everything_returns_empty():
    corpus = _FakeCorpus({"a": {"doc_type": "inpatient"}})
    reranker = build_reranker({})
    out = reranker.rerank(
        corpus,
        RerankInput(
            candidates=[_cand("a", 1)],
            query_text="q",
            filters=(EvidenceFilter(field="doc_type", op="==", value="outpatient"),),
        ),
        top_n=10,
    ).candidates
    assert out == []


# ---- fingerprint identity --------------------------------------------------

def test_composed_config_tracks_the_cross_encoder_model():
    # Two candidate rankers wrapping cross-encoders with DIFFERENT model paths must not
    # share a config (and so must not share a reranker fingerprint) — a silent
    # cross-encoder weight/model swap has to remain visible in the receipt + NO_PHI trace.
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        fingerprint,
    )

    a = CandidateRanker(cross_encoder_scorer=CrossEncoderReranker(model_path="/models/rer_a"))
    b = CandidateRanker(cross_encoder_scorer=CrossEncoderReranker(model_path="/models/rer_b"))
    assert a.info.config["cross_encoder"]["model_path"].endswith("rer_a")
    assert a.info.config != b.info.config
    assert fingerprint(a.info.config) != fingerprint(b.info.config)


# ---- recipe-spec guards ----------------------------------------------------

def test_parse_reranking_rejects_removed_kind_key(tmp_path):
    with pytest.raises(ValueError, match="reranking.kind was removed"):
        _parse_reranking({"kind": "cross_encoder"}, tmp_path / "r.yaml")


def test_parse_reranking_rejects_unknown_filter_field(tmp_path):
    with pytest.raises(ValueError, match="not filterable"):
        _parse_reranking(
            {"filters": [{"field": "patient_name", "op": "==", "value": "x"}]},
            tmp_path / "r.yaml",
        )


def test_parse_reranking_rejects_an_ordered_op_on_a_text_field(tmp_path):
    # An ordered op is refused for the reason it is meaningless, not because of which
    # field it names: text has no ordering a chart question would ask for.
    with pytest.raises(ValueError, match="orders its operands.*holds text"):
        _parse_reranking(
            {"filters": [{"field": "author", "op": ">", "value": "x"}]},
            tmp_path / "r.yaml",
        )


def test_parse_reranking_allows_an_ordered_op_on_every_orderable_field(tmp_path):
    # Whether a field can be ordered is the registry's answer, so a numeric field gets
    # range filtering without filter_candidates naming it.
    for field_name, value in (("document_date", "2020-01-01"), ("age", 18)):
        spec = _parse_reranking(
            {"filters": [{"field": field_name, "op": ">=", "value": value}]},
            tmp_path / "r.yaml",
        )
        assert spec.filters[0].field == field_name


def test_parse_reranking_rejects_an_unknown_rank_by(tmp_path):
    with pytest.raises(ValueError, match="reranking.rank_by must be one of"):
        _parse_reranking({"rank_by": "by_vibes"}, tmp_path / "r.yaml")


def test_parse_reranking_rejects_a_weight_for_a_signal_that_does_not_exist(tmp_path):
    # Scoring looks up the weight of each signal it computes, so a misspelled weight is a
    # no-op: the author retunes the ranking and nothing changes. It has to be refused.
    with pytest.raises(ValueError, match="names no such signal"):
        _parse_reranking({"weights": {"lexical_overlapp": 0.9}}, tmp_path / "r.yaml")


def test_parse_reranking_accepts_a_partial_set_of_real_weights(tmp_path):
    # Naming only some signals stays legal — the rest keep their default weight.
    spec = _parse_reranking({"weights": {"lexical_overlap": 1.0}}, tmp_path / "r.yaml")
    assert spec.scorer_config["weights"] == {"lexical_overlap": 1.0}


def test_parse_reranking_rejects_weights_that_are_not_a_mapping(tmp_path):
    with pytest.raises(ValueError, match="must map a signal name to a number"):
        _parse_reranking({"weights": ["lexical_overlap"]}, tmp_path / "r.yaml")


def test_parse_reranking_carries_rank_by_and_dedup_default(tmp_path):
    spec = _parse_reranking({"rank_by": "newest_documents"}, tmp_path / "r.yaml")

    assert spec.rank_by == "newest_documents"
    assert spec.dedup_identical_text is True          # on unless a recipe opts out
    assert spec.scorer_config == {}                   # both are top-level keys


def test_selection_choices_are_part_of_the_reranker_fingerprint(tmp_path):
    # Selecting by date vs by relevance changes which evidence the model saw, so two
    # such runs must not share a reranker fingerprint.
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        fingerprint,
    )

    by_score = build_reranker({}).info.config
    by_date = build_reranker({"rank_by": "newest_documents"}).info.config
    no_dedup = build_reranker({"dedup_identical_text": False}).info.config

    assert fingerprint(by_score) != fingerprint(by_date)
    assert fingerprint(by_score) != fingerprint(no_dedup)


def test_parse_reranking_splits_scorer_config_and_keeps_filters(tmp_path):
    spec = _parse_reranking(
        {
            "cross_encoder": False,
            "top_n": 7,
            "resort_by_date": True,
            "chronological_order": "oldest_first",
            "filters_fallback_to_unfiltered": True,
            "k0": 30,
            "weights": {"lexical_overlap": 1.0},
            "filters": [{"field": "doc_type", "op": "==", "value": "outpatient"}],
        },
        tmp_path / "r.yaml",
    )
    assert spec.cross_encoder is False
    assert spec.top_n == 7
    assert spec.resort_by_date is True
    assert spec.chronological_order == "oldest_first"
    assert spec.filters_fallback_to_unfiltered is True
    assert spec.scorer_config == {"k0": 30, "weights": {"lexical_overlap": 1.0}}
    assert len(spec.filters) == 1
    assert spec.filters[0].field == "doc_type"
