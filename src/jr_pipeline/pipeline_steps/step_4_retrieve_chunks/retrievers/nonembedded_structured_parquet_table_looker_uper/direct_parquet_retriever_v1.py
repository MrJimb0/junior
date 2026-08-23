"""Pull rows straight out of a structured table, instead of searching free text.

Use this when the answer lives in a tidy column rather than in note prose — for
example the last hemoglobin (Hgb) value, or every medication marked "Given".

These candidates do NOT appear in the chunk index; they point at structured/<stem>.parquet directly, where <stem> is the table's base filename (e.g. "diagnoses" for diagnoses.parquet).

Recommended use: run this on its own to build intermediate steps for processing a clinical question or gathering more data around a date/event
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import structured_dir
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    RetrieverInfo,
)

_VERSION = "v1"

# ── recipe config options ─────────────────────────────────────────────────────
# Set in your recipe YAML under retrieval:
#
#   kind:   direct_parquet
#   table:  parquet stem to query, e.g. "diagnoses" or "med_admin"  [required]
#   filter: row filter, e.g. "status == 'Given'"                    [optional]
#           operators: == | != | contains
#
# Example — all meds actually given (not just ordered):
#   retrieval:
#     kind: direct_parquet
#     table: med_admin
#     filter: "mar_action == 'Given'"
#
# The recipe key is `filter`; `filter_expr` below is the internal argument name it
# is passed as. A recipe that writes `filter_expr:` gets an UNFILTERED table, since
# an unrecognized retrieval key is carried in the step config and never read.

_FILTER_EXPR = re.compile(
    r"\s*(\w+)\s*(==|!=|contains)\s*(['\"])(.*?)\3\s*$"
)

# The column the original (pre-filter) row position is parked in, so a candidate's
# :TABLE id points back at the correct source row even after rows are filtered out.
_ROW_ID_COLUMN = "_orig_row_id"


def read_filtered_table_rows(parquet_path: Path, filter_expr: str | None) -> pl.DataFrame:
    """Read a structured parquet, tag each row with its ORIGINAL position
    (``_orig_row_id``) before filtering, then apply the recipe's row filter.

    The one place this 'read a table -> tag original rows -> filter' step lives, so
    the direct_parquet retriever and the retrieve_and_prompt ``also_include_tables``
    knob resolve table rows — and therefore their ``:TABLE`` ids — identically."""
    table = pl.read_parquet(parquet_path).with_row_index(_ROW_ID_COLUMN)
    return _apply_filter(table, filter_expr)


def _apply_filter(df: pl.DataFrame, expr: str | None) -> pl.DataFrame:
    """apply one ``col OP 'value'`` clause; OP in {==, !=, contains}."""
    if not expr:
        return df
    m = _FILTER_EXPR.match(expr)
    if not m:
        raise ValueError(f"Unsupported filter expression: {expr!r} — expected col OP 'value' where OP is ==, !=, or contains")
    col, op, _, val = m.group(1), m.group(2), m.group(3), m.group(4)
    if col not in df.columns:
        raise KeyError(f"Column {col!r} not in table; available: {df.columns}")
    if op == "==":
        return df.filter(pl.col(col) == val)
    if op == "!=":
        return df.filter(pl.col(col) != val)
    if op == "contains":
        # Plain substring match. literal=True means the value is matched character
        # for character, NOT as a search pattern: recipe values like "5-FU (bolus)"
        # contain punctuation that a pattern engine would treat specially, which
        # silently matched zero rows (B6).
        return df.filter(pl.col(col).cast(pl.Utf8).str.contains(val, literal=True))
    raise AssertionError("unreachable")

@dataclass
class DirectParquetRetriever:
    """Direct row lookup from a structured parquet — no embeddings needed."""

    info: RetrieverInfo

    def __init__(
        self,
        *,
        table: str,
        filter_expr: str | None = None,
    ):
        self.info = RetrieverInfo(
            kind="direct_parquet",
            version=_VERSION,
            config={
                "table": table,
                "filter_expr": filter_expr,
            },
        )
        self._table = table
        self._filter_expr = filter_expr

    @property
    def score_normalization(self) -> str:
        return "raw"

    def query(
        self,
        corpus: PatientChunkStore,
        *,
        text: str,  # noqa: ARG002 — Retriever protocol; selection is config-driven not query-driven
        k: int,
    ) -> list[Candidate]:
        """Filter the configured table by ``self._filter_expr`` and return up to ``k`` matching rows.

        ``text`` (the query string) is ignored on purpose — which rows come back is
        decided by the recipe's column filter, not by what the user typed.
        """
        stem = Path(self._table).stem
        path = structured_dir(corpus.patient_root) / f"{stem}.parquet"
        if not path.is_file():
            return []
        # Tag each row with its original position before filtering, so the candidate's
        # id points back at the correct source row and later lookups
        # (PatientChunkStore.text_for) read the right one. Shared with the
        # also_include_tables knob so both build identical :TABLE ids.
        df = read_filtered_table_rows(path, self._filter_expr)
        patient_id = corpus.chunk_index["patient_id"][0] if corpus.chunk_index.height else ""

        out: list[Candidate] = []
        for rank, row in enumerate(df.iter_rows(named=True), start=1):
            out.append(
                Candidate(
                    chunk_id=f"{patient_id}:{stem}:{row[_ROW_ID_COLUMN]}:TABLE",
                    rank=rank,
                    score=1.0,
                )
            )
            if len(out) >= k:
                break
        return out
