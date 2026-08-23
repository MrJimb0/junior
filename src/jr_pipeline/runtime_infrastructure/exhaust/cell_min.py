"""Small-cell (CELL_MIN) suppression policy for exhaust record types.

Two kinds of exhaust record:

* **row-level** training signal (one row per patient/recipe). These may ship raw if
  PHI-clean and surrogate-safe -- a single patient's outcome is not an aggregate
  disclosure, so CELL_MIN does not apply.
* **aggregate** records that directly encode counts. A small count is itself a
  re-identification risk, so every aggregate type MUST declare which count field to
  guard, the dimensions that define a cell, and the action for a cell below CELL_MIN
  (``drop`` or ``bucket``).

Every registered record type must have a policy here; a record type with no declared
policy is a **gate failure** at finalize (you cannot add an aggregate exhaust type and
forget to decide its suppression). Today all six types are row-level (each is one
patient/recipe/reviewer judgment, not an aggregate count); the mechanism is ready for
the first aggregate type without re-plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from jr_pipeline.runtime_infrastructure.exhaust.schema import RECORD_SCHEMAS

# HIPAA-style small-cell threshold: a cell counting fewer than this many subjects is
# suppressed before it can be shared/exported.
CELL_MIN = 11


class SuppressionAction(StrEnum):
    DROP = "drop"      # remove the small cell entirely
    BUCKET = "bucket"  # replace the count with a "<CELL_MIN" band


@dataclass(frozen=True)
class CellMinPolicy:
    """row-level (CELL_MIN N/A) or aggregate (guard ``count_field`` over
    ``suppression_dimensions`` with ``action``)."""

    kind: str  # "row_level" | "aggregate"
    count_field: str | None = None
    suppression_dimensions: tuple[str, ...] = ()
    action: SuppressionAction | None = None


_ROW_LEVEL = CellMinPolicy(kind="row_level")

CELL_MIN_POLICIES: dict[str, CellMinPolicy] = {
    "selection_judgment": _ROW_LEVEL,
    "clinical_invariant": _ROW_LEVEL,
    "method_provenance": _ROW_LEVEL,
    "extraction_outcome": _ROW_LEVEL,
    "relevance_label": _ROW_LEVEL,
    "retriever_miss": _ROW_LEVEL,
}


def policy_for(record_type: str) -> CellMinPolicy:
    """The CELL_MIN policy for a record type; raises if none is declared (gate failure)."""
    try:
        return CELL_MIN_POLICIES[record_type]
    except KeyError:
        raise ValueError(
            f"exhaust record type {record_type!r} has no CELL_MIN policy declared "
            "(row-level or aggregate+suppression must be decided in cell_min.py)"
        ) from None


def assert_all_record_types_have_a_policy() -> None:
    """Every registered exhaust record type must declare a CELL_MIN policy."""
    missing = [rt for rt in RECORD_SCHEMAS if rt not in CELL_MIN_POLICIES]
    if missing:
        raise ValueError(f"exhaust record types missing a CELL_MIN policy: {missing}")


def apply_cell_min(rows: list[dict[str, Any]], policy: CellMinPolicy) -> list[dict[str, Any]]:
    """Suppress small cells per ``policy``. Row-level policies pass rows through
    unchanged; aggregate policies drop or bucket any cell whose count < CELL_MIN."""
    if policy.kind == "row_level":
        return rows
    if policy.kind != "aggregate" or not policy.count_field or policy.action is None:
        raise ValueError(f"malformed aggregate CELL_MIN policy: {policy}")

    field = policy.count_field
    kept: list[dict[str, Any]] = []
    for row in rows:
        count = row.get(field)
        if isinstance(count, (int, float)) and count < CELL_MIN:
            if policy.action == SuppressionAction.DROP:
                continue
            kept.append({**row, field: None, f"{field}_band": f"<{CELL_MIN}"})
        else:
            kept.append(row)
    return kept
