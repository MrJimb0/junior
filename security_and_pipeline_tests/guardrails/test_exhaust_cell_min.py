"""Every exhaust record type has a CELL_MIN policy; aggregates suppress."""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.exhaust import schema as schema_mod
from jr_pipeline.runtime_infrastructure.exhaust.cell_min import (
    CELL_MIN,
    CellMinPolicy,
    SuppressionAction,
    apply_cell_min,
    assert_all_record_types_have_a_policy,
    policy_for,
)
from jr_pipeline.runtime_infrastructure.exhaust.schema import RECORD_SCHEMAS


def test_every_registered_record_type_has_a_policy():
    assert_all_record_types_have_a_policy()  # must not raise
    for record_type in RECORD_SCHEMAS:
        assert policy_for(record_type).kind in ("row_level", "aggregate")


def test_missing_policy_is_a_gate_failure(monkeypatch):
    monkeypatch.setitem(schema_mod.RECORD_SCHEMAS, "an_aggregate_with_no_policy", object)
    with pytest.raises(ValueError, match="missing a CELL_MIN policy"):
        assert_all_record_types_have_a_policy()


def test_policy_for_unknown_type_raises():
    with pytest.raises(ValueError, match="no CELL_MIN policy"):
        policy_for("not_a_record_type")


def test_row_level_rows_pass_through_unchanged():
    rows = [{"patient_surrogate": "surr:v1:run:patient_surrogate:" + "a" * 24}]
    assert apply_cell_min(rows, policy_for("extraction_outcome")) == rows


def test_aggregate_drop_suppresses_cells_below_cell_min():
    policy = CellMinPolicy(kind="aggregate", count_field="n",
                           suppression_dimensions=("field",), action=SuppressionAction.DROP)
    rows = [{"field": "a", "n": CELL_MIN - 1}, {"field": "b", "n": CELL_MIN}]
    kept = apply_cell_min(rows, policy)
    assert [r["field"] for r in kept] == ["b"]  # N=CELL_MIN-1 dropped, N=CELL_MIN kept


def test_aggregate_bucket_bands_small_cells_instead_of_dropping():
    policy = CellMinPolicy(kind="aggregate", count_field="n",
                           suppression_dimensions=("field",), action=SuppressionAction.BUCKET)
    rows = [{"field": "a", "n": CELL_MIN - 1}, {"field": "b", "n": CELL_MIN + 5}]
    kept = apply_cell_min(rows, policy)
    assert kept[0]["n"] is None and kept[0]["n_band"] == f"<{CELL_MIN}"
    assert kept[1]["n"] == CELL_MIN + 5  # large cell untouched
