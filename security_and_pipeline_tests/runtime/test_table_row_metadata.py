"""A structured-table row is filterable and date-orderable, like any other evidence.

A ``:TABLE`` pseudo-chunk (one whole row of an ingested table, retrieved by the
direct_parquet retriever) has no chunk_index row — nothing embedded it. Its chart
metadata therefore has to come from the parquet row itself, through the column mapping
ingest recorded in the file's sidecar. ``PatientChunkStore.metadata_for`` is where that
happens, and step 5 reads every candidate's metadata through it.

Without that, a recipe combining table evidence with ``filters:`` drops every table row
for a missing field it never looked for, and ``rank_by: newest_documents`` treats every
table row as undated. These pin the fix, ending with the case that motivated it: the
most recent row authored by a named clinician.
"""
import json

import polars as pl

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import CandidateRanker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    EvidenceFilter,
    RerankInput,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate, PatientChunkStore

_ONCOLOGY = "Chen, MD - Medical Oncology"
_RADIOLOGY = "Patel, MD - Radiology"


def _make_store(tmp_path, table, *, metadata_columns=None, sidecar=True):
    """A patient root holding one ingested table (stem ``labs``) and an empty chunk index.

    ``metadata_columns`` is the mapping ingest would have recorded; pass None to write a
    sidecar without one (the builder-table case, where the fallback resolves aliases
    against the header instead).
    """
    root = tmp_path / "patient"
    structured = root / "structured"
    structured.mkdir(parents=True)
    pl.DataFrame({"chunk_id": [], "source_file": []}, schema={
        "chunk_id": pl.Utf8, "source_file": pl.Utf8,
    }).write_parquet(root / "chunk_index.parquet")

    parquet = structured / "labs.parquet"
    pl.DataFrame(table).write_parquet(parquet)
    if sidecar:
        payload = {
            "source_file": "labs.csv",
            "parquet_content_hash": hash_file(parquet),
        }
        if metadata_columns is not None:
            payload["metadata_columns"] = metadata_columns
        (structured / "labs.parquet.meta.json").write_text(json.dumps({"payload": payload}))
    return PatientChunkStore(patient_root=root)


def _table_candidates(count):
    return [Candidate(chunk_id=f"p:labs:{i}:TABLE", rank=i + 1, score=1.0) for i in range(count)]


# ---- the store resolves a table row's metadata ------------------------------

def test_metadata_comes_from_the_columns_ingest_recorded(tmp_path):
    # The site calls them "signed_by" and "collected"; ingest resolved the mapping once
    # and wrote it beside the data, so the filter's standard names find them.
    store = _make_store(
        tmp_path,
        {"signed_by": [_ONCOLOGY], "collected": ["2021-06-01"], "result": ["CA 15-3 42"]},
        metadata_columns={"author": "signed_by", "document_date": "collected", "doc_type": None},
    )
    metadata = store.metadata_for("p:labs:0:TABLE")
    assert metadata["author"] == _ONCOLOGY
    assert metadata["document_date"] == "2021-06-01"
    # a field this file has no column for is absent, not an error
    assert metadata["doc_type"] is None
    # source_file comes from the sidecar, so a table row and a text chunk both carry one
    assert metadata["source_file"] == "labs.csv"


def test_default_aliases_resolve_a_table_with_no_recorded_mapping(tmp_path):
    # A builder table ingest never saw has no metadata_columns in its sidecar; the
    # header's own names are matched against the default aliases instead.
    store = _make_store(
        tmp_path,
        {"author": [_ONCOLOGY], "date": ["2021-06-01"]},
        metadata_columns=None,
    )
    metadata = store.metadata_for("p:labs:0:TABLE")
    assert metadata["author"] == _ONCOLOGY
    assert metadata["document_date"] == "2021-06-01"


def test_a_typed_date_column_reads_as_the_iso_text_the_filters_parse(tmp_path):
    # A parquet date column arrives as a date object, not a string. It has to reach the
    # filters as ISO text or every date comparison silently reads it as unparseable.
    store = _make_store(
        tmp_path,
        {"author": [_ONCOLOGY], "date": [pl.Series(["2021-06-01"]).str.to_date()[0]]},
        metadata_columns={"author": "author", "document_date": "date"},
    )
    assert store.metadata_for("p:labs:0:TABLE")["document_date"] == "2021-06-01"


def test_an_unresolvable_id_reads_as_no_metadata(tmp_path):
    store = _make_store(tmp_path, {"author": [_ONCOLOGY]}, metadata_columns={"author": "author"})
    assert store.metadata_for("p:labs:99:TABLE") is None   # row past the end of the table
    assert store.metadata_for("p:missing_table:0:TABLE") is None
    assert store.metadata_for("not-a-chunk-id") is None


# ---- step 5 filters and orders table rows on those terms --------------------

def _ranker(**kwargs):
    return CandidateRanker(**kwargs)


def test_a_filter_keeps_the_table_rows_that_match_instead_of_dropping_all_of_them(tmp_path):
    store = _make_store(
        tmp_path,
        {
            "signed_by": [_ONCOLOGY, _RADIOLOGY, _ONCOLOGY],
            "collected": ["2021-06-01", "2022-01-01", "2020-01-01"],
            "result": ["CA 15-3 42", "no acute finding", "CA 15-3 30"],
        },
        metadata_columns={"author": "signed_by", "document_date": "collected"},
    )
    result = _ranker().rerank(
        store,
        RerankInput(
            candidates=_table_candidates(3),
            query_text="tumour marker",
            filters=(EvidenceFilter(field="author", op="contains", value="medical oncology"),),
        ),
        top_n=10,
    )
    assert [c.chunk_id for c in result.candidates] == ["p:labs:0:TABLE", "p:labs:2:TABLE"]
    assert result.filter_stats[0]["dropped"] == 1
    # the one that went was the radiologist's row, not a row whose author went unread
    assert result.filter_stats[0]["dropped_missing"] == 0


def test_a_date_filter_compares_the_table_rows_own_date(tmp_path):
    store = _make_store(
        tmp_path,
        {"signed_by": [_ONCOLOGY] * 2, "collected": ["2019-05-01", "2021-06-01"]},
        metadata_columns={"author": "signed_by", "document_date": "collected"},
    )
    result = _ranker().rerank(
        store,
        RerankInput(
            candidates=_table_candidates(2),
            query_text="tumour marker",
            filters=(EvidenceFilter(field="document_date", op=">=", value="2020-01-01"),),
        ),
        top_n=10,
    )
    assert [c.chunk_id for c in result.candidates] == ["p:labs:1:TABLE"]


def test_rank_by_newest_documents_orders_table_rows_by_their_date(tmp_path):
    store = _make_store(
        tmp_path,
        {"signed_by": [_ONCOLOGY] * 3, "collected": ["2020-01-01", "2022-03-15", "2021-06-01"]},
        metadata_columns={"author": "signed_by", "document_date": "collected"},
    )
    result = _ranker(rank_by="newest_documents").rerank(
        store, RerankInput(candidates=_table_candidates(3), query_text=""), top_n=3
    )
    assert [c.chunk_id for c in result.candidates] == [
        "p:labs:1:TABLE", "p:labs:2:TABLE", "p:labs:0:TABLE",
    ]


def test_the_last_row_by_a_named_author(tmp_path):
    """The case this is all for: of the rows this clinician signed, take the most recent.

    Selection has to read the author (to drop the other clinician's rows) and the date
    (to order what is left) off rows that were never embedded — and 2022-01-01 is the
    newest row in the table, so a filter that did not really apply would show up here.
    """
    store = _make_store(
        tmp_path,
        {
            "signed_by": [_ONCOLOGY, _RADIOLOGY, _ONCOLOGY, _ONCOLOGY],
            "collected": ["2020-01-01", "2022-01-01", "2021-06-01", "2019-01-01"],
            "result": ["CA 15-3 42", "no acute finding", "CA 15-3 51", "CA 15-3 30"],
        },
        metadata_columns={"author": "signed_by", "document_date": "collected"},
    )
    result = _ranker(rank_by="newest_documents").rerank(
        store,
        RerankInput(
            candidates=_table_candidates(4),
            query_text="tumour marker",
            filters=(EvidenceFilter(field="author", op="contains", value="medical oncology"),),
        ),
        top_n=1,
    )
    assert [c.chunk_id for c in result.candidates] == ["p:labs:2:TABLE"]
    assert store.text_for("p:labs:2:TABLE").endswith("result: CA 15-3 51")


def test_a_file_with_no_column_for_the_filtered_field_still_gates_on_keep_if_missing(tmp_path):
    # The guarantee that did not change: a field the file genuinely has no column for is
    # absent, and keep_if_missing is what decides whether that row survives.
    store = _make_store(
        tmp_path,
        {"result": ["CA 15-3 42"]},
        metadata_columns={"author": None, "document_date": None},
    )
    dropped = _ranker().rerank(
        store,
        RerankInput(
            candidates=_table_candidates(1),
            query_text="tumour marker",
            filters=(EvidenceFilter(field="author", op="contains", value="oncology"),),
        ),
        top_n=10,
    )
    assert dropped.candidates == []
    assert dropped.filter_stats[0]["dropped_missing"] == 1

    kept = _ranker().rerank(
        store,
        RerankInput(
            candidates=_table_candidates(1),
            query_text="tumour marker",
            filters=(
                EvidenceFilter(
                    field="author", op="contains", value="oncology", keep_if_missing=True
                ),
            ),
        ),
        top_n=10,
    )
    assert [c.chunk_id for c in kept.candidates] == ["p:labs:0:TABLE"]
