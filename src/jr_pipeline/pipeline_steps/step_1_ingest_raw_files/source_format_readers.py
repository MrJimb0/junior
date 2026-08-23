"""Read each supported chart file type (CSV, Excel, JSON, parquet, ...) into a
table, keeping every value as plain text.

We read everything as text on purpose: dates and codes get cleaned up in later,
deliberate steps rather than guessed at by the file reader. To support a new file
type, add one entry to ``_FORMAT_READERS`` and the matching file extension to
``SUPPORTED_SOURCE_EXTENSIONS`` (in the shared discovery module). A check that
runs when this module loads keeps those two lists in lockstep, so the set of file
types we can read can never drift from the set we go looking for."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl

from jr_pipeline.runtime_infrastructure.source_file_discovery import SUPPORTED_SOURCE_EXTENSIONS

# Strings that should be read as "no value" (empty/missing) rather than as
# literal text. The many case variants are listed explicitly because EHR exports
# are inconsistent across vendors (Epic, Cerner, in-house data pipelines).
# "Unknown" is intentionally NOT in this list — it can be a real, meaningful
# value (e.g. for race/ethnicity).
_NULL_TOKENS = [
    "", "NA", "N/A", "n/a", "na", "NaN", "nan",
    "NULL", "Null", "null", "None", "none",
]


def _read_csv(path: Path) -> pl.DataFrame:
    """Read a CSV or tab-separated file, keeping every value as text. Raises an
    error on rows with the wrong number of columns rather than quietly dropping
    the extra fields — silently cutting off trailing fields could lose a note's
    ``text`` column with no warning."""
    separator = "\t" if path.suffix.lower() == ".tsv" else ","
    return pl.read_csv(
        path,
        separator=separator,
        infer_schema_length=0,
        ignore_errors=False,
        null_values=_NULL_TOKENS,
        truncate_ragged_lines=False,
    )


def _read_xlsx(path: Path) -> pl.DataFrame:
    """Read an Excel sheet, keeping every value as text. Needs the ``fastexcel``
    package — raise a clear, actionable error if it isn't installed instead of
    the confusing import error polars would otherwise produce."""
    try:
        return pl.read_excel(path, infer_schema_length=0)
    except ImportError as e:
        raise RuntimeError(
            f"Reading {path.name} requires the 'fastexcel' package; "
            "install with `uv pip install fastexcel`."
        ) from e


def _as_text(frame: pl.DataFrame) -> pl.DataFrame:
    """Cast each scalar column to text, matching the text-first contract the CSV and
    Excel readers get from ``infer_schema_length=0``. The JSON parsers infer types from
    the document (a bare number becomes an integer, ``true`` a boolean), so a numeric
    ``note_id`` or a boolean flag would otherwise reach the pipeline as a non-text column
    and read differently than the same value out of a CSV. Nested columns (a JSON array
    or object field) are left untouched: they cannot be cast to a single string, and the
    chunk builder stringifies whatever it reads anyway. (Passing ``infer_schema_length=0``
    to ``read_json`` is not an option — polars panics on it.)"""
    scalar_columns = [name for name, dtype in frame.schema.items() if not dtype.is_nested()]
    return frame.with_columns(pl.col(scalar_columns).cast(pl.Utf8))


def _read_json(path: Path) -> pl.DataFrame:
    """Read a single JSON document (object or array of objects) as an all-text frame."""
    return _as_text(pl.read_json(path))


def _read_jsonl(path: Path) -> pl.DataFrame:
    """Read newline-delimited JSON (one record per line) as an all-text frame."""
    return _as_text(pl.read_ndjson(path))


def _read_parquet(path: Path) -> pl.DataFrame:
    """Read a parquet file. Its columns already carry types; the date-cleanup
    step may still rewrite text columns to ISO dates."""
    return pl.read_parquet(path)


_FORMAT_READERS: dict[str, Callable[[Path], pl.DataFrame]] = {
    ".csv":     _read_csv,
    ".tsv":     _read_csv,
    ".xlsx":    _read_xlsx,
    ".json":    _read_json,
    ".jsonl":   _read_jsonl,
    ".parquet": _read_parquet,
}
# The set of file types we can read must exactly match the set of file types we
# go looking for: a reader added here but not declared discoverable (or the
# reverse) would let the two drift apart. Stop loudly when this module loads if
# they ever disagree.
assert set(_FORMAT_READERS) == set(SUPPORTED_SOURCE_EXTENSIONS), (
    "ingest readers and SUPPORTED_SOURCE_EXTENSIONS disagree: "
    f"{sorted(set(_FORMAT_READERS) ^ set(SUPPORTED_SOURCE_EXTENSIONS))}"
)


def _read_source(path: Path) -> pl.DataFrame:
    """Pick the right reader for a file based on its extension and run it."""
    reader = _FORMAT_READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"Unsupported source extension: {path.suffix!r} ({path}); "
            f"supported: {list(_FORMAT_READERS)}"
        )
    return reader(path)
