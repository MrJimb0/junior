"""Rewrite date columns into one consistent format so downstream steps compare
dates reliably.

Source charts write dates many ways (2024-01-02, 1/2/2024, with or without a
time). This module rewrites any text column that looks like dates into the
standard ISO-8601 form (YYYY-MM-DD, e.g. 2024-01-02). Two patterns (regular
expressions) recognize the only two shapes ingest understands — ISO-8601-style
and American slash dates (M/D/YYYY); anything else is left exactly as written.
A column is only treated as dates if at least 60% of a sample of its values look
date-shaped, so free-text columns that happen to mention a date are left alone.
A 2-digit year (e.g. 1/2/24) raises AmbiguousYearError: ingest refuses, rather
than guess, whether '24' means 1924 or 2024. Self-contained: depends only on
polars (the table library) and the Python standard library."""
from __future__ import annotations

import re
from datetime import UTC, datetime

import polars as pl

# Matches YYYY-MM-DD, optionally followed by a time (separated by T or a space),
# seconds, fractional seconds, and a timezone offset.
_DATE_CANDIDATE = re.compile(
    r"^\s*\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?([+-]\d{2}:?\d{2}|Z)?\s*$"
)
# Matches American slash dates M/D/YY or M/D/YYYY, optionally followed by a time
# (HH:MM, with optional :SS). A 2-digit year here triggers AmbiguousYearError —
# ingest refuses to guess the century.
_SLASH_DATE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*$")

# How much of a column to sample, and how much of that sample must look like
# dates, before we treat the whole column as a date column. 100 rows is enough to
# avoid being fooled by a rare typo; the 60% threshold is high enough to skip
# free-text columns that happen to mention a date or two, but low enough to catch
# sparse columns where many rows are blank but every filled-in value is a date.
_DATE_SAMPLE_SIZE = 100
_DATE_SAMPLE_THRESHOLD = 0.6


class AmbiguousYearError(ValueError):
    """Raised when a date has a 2-digit year. Ingest will not guess centuries —
    re-export the source CSV with 4-digit years (M/D/YYYY)."""


def _try_iso(value: str) -> str | None:
    """Rewrite one string into standard ISO-8601 date form, or return None if it
    isn't date-shaped.

    A date with no time stays a plain date (``1957-04-15``) — inventing a
    midnight timestamp would add false precision (a date of birth has no time of
    day or timezone), and since the language model downstream copies whatever the
    evidence shows, that invented time would break date-only output fields. A
    value that carries a time is rewritten to a full ISO-8601 timestamp in UTC.
    Raises AmbiguousYearError on a slash date with a 2-digit year.
    """
    s = value.strip()
    if not s:
        return None
    iso_m = _DATE_CANDIDATE.match(s)
    if iso_m:
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
        if iso_m.group(1) is None:  # the source value had no time, only a date
            return dt.date().isoformat()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    m = _SLASH_DATE.match(s)
    if m:
        mo, day, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yr < 100:
            raise AmbiguousYearError(
                f"date {s!r} has a 2-digit year; re-export with M/D/YYYY"
            )
        try:
            base = datetime(yr, mo, day, tzinfo=UTC)
        except ValueError:
            return None
        time_part = m.group(4)
        if not time_part:
            return base.date().isoformat()
        # Preserve the source's time-of-day.
        parts = time_part.strip().split(":")
        hh, mi = int(parts[0]), int(parts[1])
        ss = int(parts[2]) if len(parts) > 2 else 0
        try:
            return base.replace(hour=hh, minute=mi, second=ss).isoformat()
        except ValueError:
            return None
    return None


def _sample_date_offenders(series: pl.Series) -> tuple[bool, list[str]]:
    """Sample the first values of a text column and decide:
       - is_date_column: at least the threshold fraction of sampled values look
         date-shaped
       - offenders: any 2-digit-year slash dates seen in the sample (the ones we
         refuse to interpret)

    Shared by _canonicalize_dates (which raises on offenders) and _preflight_one
    (which records them as a problem to report) so the threshold and the parsing
    rule are defined in exactly one place."""
    sample = series.drop_nulls().head(_DATE_SAMPLE_SIZE).to_list()
    if not sample:
        return False, []
    parseable = 0
    offenders: list[str] = []
    for v in sample:
        try:
            if _try_iso(v) is not None:
                parseable += 1
        except AmbiguousYearError:
            parseable += 1
            offenders.append(v)
    is_date = parseable / len(sample) >= _DATE_SAMPLE_THRESHOLD
    return is_date, offenders


# A slash date with a 2-digit year (M/D/YY, optional time) — the shape ingest
# refuses to interpret. Written so that a 4-digit (M/D/YYYY) or 3-digit year
# never matches. Used to scan an entire column at once.
_TWO_DIGIT_YEAR_SLASH_DATE = r"^\s*\d{1,2}/\d{1,2}/\d{2}(\s+\d{1,2}:\d{2}(:\d{2})?)?\s*$"


def _two_digit_year_offenders(series: pl.Series) -> list[str]:
    """The distinct values in a text column that are slash dates with a 2-digit
    year. Scans the whole column, not just the sampled head, so an offending
    value that sits past the sample window is still caught and refused."""
    mask = series.str.contains(_TWO_DIGIT_YEAR_SLASH_DATE).fill_null(False)
    return series.filter(mask).unique().to_list()


def _iso_replacement_map(series: pl.Series, col_name: str) -> dict[str, str]:
    """Build a lookup from each distinct date-shaped value in a column to its
    ISO-8601 form, skipping non-dates and values that are already in ISO form.
    Computing it over the distinct values (then applying the lookup to the whole
    column at once) gives the same result as converting cell by cell, but far
    faster. Any 2-digit-year value must have been caught earlier (see
    ``_two_digit_year_offenders``); this is a backstop that re-raises with the
    column name if one slips through."""
    mapping: dict[str, str] = {}
    for value in series.drop_nulls().unique().to_list():
        try:
            iso = _try_iso(value)
        except AmbiguousYearError as e:
            raise AmbiguousYearError(f"column {col_name!r}: {e.args[0]}") from None
        if iso is not None and iso != value:
            mapping[value] = iso
    return mapping


def _canonicalize_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Rewrite every date-like text column in the table to ISO-8601. Values that
    don't look like dates are left exactly as they were."""
    if df.is_empty():
        return df
    for col_name, dtype in df.schema.items():
        if dtype != pl.Utf8:
            continue
        series = df[col_name]
        is_date, _ = _sample_date_offenders(series)
        if not is_date:
            continue
        offenders = _two_digit_year_offenders(series)
        if offenders:
            shown = offenders[:5]
            preview = ", ".join(repr(v) for v in shown)
            raise AmbiguousYearError(
                f"column {col_name!r} contains date values with 2-digit years "
                f"(first {len(shown)} of {len(offenders)}: {preview}); "
                "re-export the source CSV with 4-digit years (M/D/YYYY)"
            )
        mapping = _iso_replacement_map(series, col_name)
        if mapping:
            df = df.with_columns(pl.col(col_name).replace(mapping).alias(col_name))
    return df
