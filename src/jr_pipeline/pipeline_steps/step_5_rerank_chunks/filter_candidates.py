"""Decide which candidate chunks a recipe's ``filters:`` block lets through.

A recipe can gate evidence on chart metadata — author, document type, title, document
date. For example: only medical oncology notes, or only documents dated after the
diagnosis. Step 5 applies these before it ranks anything.

Free text and structured-table rows are gated on the same terms.

Date filters compare ranges, not single days.

Ranking, deduplication and trimming happen after in rank_candidates.
"""
from __future__ import annotations

import math
import re
from calendar import monthrange
from datetime import date
from typing import Any

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    EvidenceFilter,
)
from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    METADATA_FIELD_KINDS,
    STANDARD_METADATA_FIELDS,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate, PatientChunkStore

# A recipe may filter on any of the standard chart metadata fields — one definition,
# shared with the step that writes those columns, so the two cannot drift.
FILTERABLE_FIELDS = STANDARD_METADATA_FIELDS
# Text comparisons, valid on any field.
TEXT_OPS = ("contains", "not_contains", "==", "!=", "in")
# Ordered comparisons. Valid on any field the registry gives an orderable kind — a date or
# a number — so which fields those are is answered in one place, not by a name checked here.
ORDERED_OPS = (">", ">=", "<", "<=")
ALL_OPS = TEXT_OPS + ORDERED_OPS


# YYYY-MM-DD, where the month and/or day may be XX, and a time may follow.
#
# Ingest canonicalizes every date column to ISO and never writes XX, so a masked date
# never arrives on a chunk. It arrives on the other side of the comparison: a filter
# whose value came from an upstream extracted variable, where the chart gave only a year.
#
# The recipe tree owns this convention (_shared_validation_rules/partial_date.py). It is
# restated here because that tree is swappable data — pointing recipes_root at another
# folder must not change what the engine treats as a date. Only the parse is shared in
# spirit: partial_date.strictly_before refuses to order two dates once precision runs
# out, while the filters here keep anything that might satisfy.
_MASKED_DATE_RE = re.compile(r"^(\d{4})-(\d{2}|XX)-(\d{2}|XX)(?:[T ].*)?$")


def parse_date_interval(value: Any) -> tuple[date, date] | None:
    """Parse a (possibly partial) date string into the closed interval of real dates it
    could denote, or None if it is not date-shaped.

    A full date is a point interval; ``2020-XX-XX`` becomes ``[2020-01-01, 2020-12-31]``
    and ``2020-03-XX`` becomes ``[2020-03-01, 2020-03-31]``. A trailing time (document
    metadata carries one) is ignored. Returns None for a value that is not date-shaped so
    the caller can fall back to its missing-value handling.
    """
    if not isinstance(value, str):
        return None
    m = _MASKED_DATE_RE.match(value.strip())
    if not m:
        return None
    year = int(m.group(1))
    month_masked = m.group(2) == "XX"
    day_masked = m.group(3) == "XX"
    earliest_month = 1 if month_masked else int(m.group(2))
    latest_month = 12 if month_masked else int(m.group(2))
    try:
        earliest_day = 1 if day_masked else int(m.group(3))
        latest_day = monthrange(year, latest_month)[1] if day_masked else int(m.group(3))
        earliest = date(year, earliest_month, earliest_day)
        latest = date(year, latest_month, latest_day)
    except (ValueError, IndexError):
        return None
    if earliest > latest:
        return None
    return (earliest, latest)


def _intervals_overlap(a: tuple[date, date], b: tuple[date, date]) -> bool:
    """True when the two closed date intervals share at least one calendar day."""
    return a[0] <= b[1] and b[0] <= a[1]


def parse_number(value: Any) -> float | None:
    """Parse a quantity, or None if it is not one.

    A chart writes an age as ``62``, ``62.0`` or ``"62"`` depending on the export, and a
    null numeric column reaches here as the text ``"nan"``. All of those have to compare
    as the same kind of thing, or a recipe's ``age >= 18`` would depend on how the site
    happened to type its column. A bool is not a quantity even though Python says it is.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _field_value(corpus: PatientChunkStore, chunk_id: str, field: str) -> str | None:
    """The candidate's value for one metadata field, or None when it is absent.

    ``metadata_for`` answers for an embedded chunk (from its chunk_index row) and for a
    ``:TABLE`` pseudo-chunk (a whole structured-table row, read from the parquet through
    the column mapping ingest recorded), so a recipe's filters gate table evidence on the
    same terms as free text. A field the source file has no column for still reads as
    absent, and ``keep_if_missing`` decides what that means.
    """
    row = corpus.metadata_for(chunk_id)
    if row is None:
        return None
    value = row.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _matches(value: str, op: str, target: Any, kind: str) -> bool | None:
    """Does ``value`` satisfy ``op`` against ``target``, compared as ``kind`` says?

    ``kind`` is the field's kind from the registry — date, number or text. It decides how
    both sides are read, so a recipe writes ``age >= 18`` and ``document_date >= ...`` the
    same way and each means what it should.

    Returns True/False, or None when the comparison cannot be made (an operand that is not
    a parseable date, or not a number, when the field is one) so the caller can treat it as
    missing. Text comparisons are case-insensitive.

    Dates are compared as intervals under a **possibly-satisfies** rule, not
    definitely-satisfies. A partial date like ``2020-XX-XX`` denotes a whole range of
    real dates, so the candidate is kept when ANY pair of points from the candidate's and
    the target's intervals satisfies the operator. This is a deliberate choice: these
    filters gate which evidence the model reads, and the failure mode being guarded
    against is dropping evidence we cannot prove should be dropped — so when precision
    runs out we keep the chunk and let reranking judge it.
    """
    low = value.lower()
    if op == "contains":
        return str(target).lower() in low
    if op == "not_contains":
        return str(target).lower() not in low

    targets = target if isinstance(target, (list, tuple)) else [target]

    if kind == "number":
        # Every comparison on a quantity is numeric, so "62.0" and 62 are one age and a
        # site that exported its column as text does not get a different answer.
        candidate = parse_number(value)
        wanted = [n for n in (parse_number(t) for t in targets) if n is not None]
        if candidate is None or not wanted or len(wanted) != len(targets):
            return None  # not a quantity -> caller treats it like a missing field
        if op == "in":
            return any(candidate == w for w in wanted)
        if op == "==":
            return candidate == wanted[0]
        if op == "!=":
            return candidate != wanted[0]
        if op == ">":
            return candidate > wanted[0]
        if op == ">=":
            return candidate >= wanted[0]
        if op == "<":
            return candidate < wanted[0]
        if op == "<=":
            return candidate <= wanted[0]
        raise ValueError(f"unknown filter op {op!r}")

    if op == "in":
        return low.strip() in {str(t).strip().lower() for t in targets}

    if kind == "date":
        # Dates compare as intervals: ``==`` is possibly-equal (they overlap) and ``!=``
        # possibly-different (not the same single day). An operand that is not a date
        # falls back to a verbatim compare for ==/!=, and is unanswerable for the rest.
        left = parse_date_interval(value)
        right = parse_date_interval(target)
        if left is not None and right is not None:
            candidate_earliest, candidate_latest = left
            target_earliest, target_latest = right
            if op == "==":
                return _intervals_overlap(left, right)
            if op == "!=":
                both_single_day = left[0] == left[1] and right[0] == right[1]
                return not (both_single_day and left[0] == right[0])
            if op == ">":
                return candidate_latest > target_earliest
            if op == ">=":
                return candidate_latest >= target_earliest
            if op == "<":
                return candidate_earliest < target_latest
            if op == "<=":
                return candidate_earliest <= target_latest
            raise ValueError(f"unknown filter op {op!r}")
        if op not in ("==", "!="):
            return None  # unparseable date -> caller treats it like a missing field

    # text, and the date fallback: compare verbatim
    if op in ("==", "!="):
        equal = low.strip() == str(target).strip().lower()
        return equal if op == "==" else not equal
    # An ordered op on a text field has nothing to order by. Recipe validation refuses
    # this combination at load, so reaching here means a caller built the filter directly.
    raise ValueError(f"filter op {op!r} needs an orderable field, not {kind!r}")


def apply_filters(
    candidates: list[Candidate],
    corpus: PatientChunkStore,
    filters: tuple[EvidenceFilter, ...],
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Keep only the candidates that satisfy every filter (filters are ANDed).

    Returns the survivors plus one stats row per filter: how many candidates it
    dropped, how many of those were dropped because the field was absent, and how many
    because the field was present but its value could not be parsed as a date (a
    distinct signal — an unparseable date is a data problem, not simply a missing one).
    A candidate is checked against the filters in order and removed by the first one it
    fails.
    """
    # Annotated because the rows mix counters with the field/op labels and the
    # unparseable example; without it the counters below read as text-or-number.
    stats: list[dict[str, Any]] = [
        {
            "field": f.field,
            "op": f.op,
            "dropped": 0,
            "dropped_missing": 0,
            "dropped_unparseable": 0,
            "unparseable_example": None,
        }
        for f in filters
    ]
    if not filters:
        return list(candidates), stats

    survivors: list[Candidate] = []
    for candidate in candidates:
        failed_index: int | None = None
        failed_on_missing = False
        failed_on_unparseable = False
        failed_value: str | None = None
        for i, f in enumerate(filters):
            value = _field_value(corpus, candidate.chunk_id, f.field)
            missing = False
            unparseable = False
            if value is None:
                missing = True
                passed = f.keep_if_missing
            else:
                result = _matches(
                    value, f.op, f.value, METADATA_FIELD_KINDS.get(f.field, "text")
                )
                if result is None:
                    # Present but a date operator could not parse it as a date. Treat it
                    # like a missing field for the keep decision, but record it apart.
                    unparseable = True
                    passed = f.keep_if_missing
                else:
                    passed = result
            if not passed:
                failed_index = i
                failed_on_missing = missing
                failed_on_unparseable = unparseable
                failed_value = value
                break
        if failed_index is None:
            survivors.append(candidate)
        else:
            stats[failed_index]["dropped"] += 1
            if failed_on_missing:
                stats[failed_index]["dropped_missing"] += 1
            if failed_on_unparseable:
                stats[failed_index]["dropped_unparseable"] += 1
                if stats[failed_index]["unparseable_example"] is None:
                    stats[failed_index]["unparseable_example"] = failed_value
    return survivors, stats
