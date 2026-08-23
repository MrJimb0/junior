"""Partial clinical-date parsing — the single source of truth.

A clinical date is ``YYYY-MM-DD`` where the month and/or day may be masked as ``XX``
when the chart only gave us a partial date (e.g. a year with no month/day). Two callers
share the same parse:

* the non-date default — ``date_key`` returns ``None`` (consistency checks look for ``is
  None`` to skip a rule that needs a real date), ``date_sort_key`` returns a key that sorts
  LAST (so sorting never silently drops an undatable row).
* ``year_from_date`` tolerates a trailing time (``YYYY-MM-DD[ T...]``) because document
  metadata carries one; the others reject it.

Loaded by ``jr_pipeline.runtime_infrastructure.recipe_shared_rules.load_shared_validation_rule``
since recipe helpers run by file path, not as an importable package. Stdlib-only, pure.
"""
from __future__ import annotations

import re
from typing import Any

# YYYY-MM-DD with MM and/or DD optionally masked as XX. No trailing time.
_DATE_RE = re.compile(r"^(\d{4})-(\d{2}|XX)-(\d{2}|XX)$")
# Same, but tolerant of a trailing time/timestamp (document metadata carries one).
_DATE_OR_DATETIME_RE = re.compile(r"^(\d{4})-(\d{2}|XX)-(\d{2}|XX)([T ].*)?$")

# Sorts after every real date — the conventional "undatable -> last" key.
UNDATABLE_SORT_KEY: tuple[int, int, int] = (9999, 99, 99)


def is_date(value: Any) -> bool:
    """True iff ``value`` is a date-shaped string ``YYYY-MM-DD`` (MM/DD may be XX).

    Does NOT strip — matches the verbatim model output a merge step guards against."""
    return isinstance(value, str) and bool(_DATE_RE.match(value))


def date_key(value: Any) -> tuple[int, int, int] | None:
    """Parse a partial date into a comparable ``(year, month, day)`` with ``0`` for a
    masked (XX) component; ``None`` if ``value`` is not a date-shaped string."""
    if not isinstance(value, str):
        return None
    m = _DATE_RE.match(value.strip())
    if not m:
        return None
    month = 0 if m.group(2) == "XX" else int(m.group(2))
    day = 0 if m.group(3) == "XX" else int(m.group(3))
    return (int(m.group(1)), month, day)


def date_sort_key(value: Any) -> tuple[int, int, int]:
    """Like :func:`date_key` but maps a non-date to :data:`UNDATABLE_SORT_KEY` so a
    chronological sort orders undatable rows last instead of dropping them."""
    return date_key(value) or UNDATABLE_SORT_KEY


def strictly_before(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    """True only if ``a`` is unambiguously before ``b`` at the precision they SHARE. A
    masked component (``0``, an unknown month or day) means we run out of precision and stop
    comparing: ``2020-XX-XX`` vs ``2020-03-XX`` yields False rather than guessing an order we
    cannot actually justify."""
    for ca, cb in zip(a, b, strict=False):
        if ca == 0 or cb == 0:
            return False   # precision exhausted; cannot assert ordering
        if ca != cb:
            return ca < cb
    return False           # equal at full known precision -> not strictly before


def month_ordinal(dk: tuple[int, int, int]) -> int | None:
    """Calendar-month index (``year*12 + month``) for closeness checks, or None if the
    month is masked."""
    year, month, _day = dk
    return None if month == 0 else year * 12 + month


def year_from_date(value: Any) -> int | None:
    """Year from a date OR a full document timestamp (``YYYY-MM-DD[ T...]``); None if not
    date-shaped. Tolerates a trailing time because document metadata carries one."""
    if not isinstance(value, str):
        return None
    m = _DATE_OR_DATETIME_RE.match(value.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None
