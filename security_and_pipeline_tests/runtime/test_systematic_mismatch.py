"""Systematic-vs-isolated aggregation of value mismatches."""
from __future__ import annotations

from jr_pipeline.evaluating_pipeline_performance.eval_error_analysis import (
    aggregate_mismatch_patterns,
)


def _mismatch(patient_id, field, extracted, expected):
    return {
        "patient_id": patient_id,
        "error_type": "value_mismatch",
        "mismatches": {field: {"extracted": extracted, "expected": expected}},
    }


def test_same_mistake_two_patients_is_systematic():
    errors = [
        _mismatch("p1", "stage", "IIA", "IIB"),
        _mismatch("p2", "stage", "IIA", "IIB"),
    ]
    patterns = aggregate_mismatch_patterns(errors)
    assert len(patterns) == 1
    assert patterns[0]["pattern"] == "systematic"
    assert patterns[0]["n_patients"] == 2


def test_single_patient_is_isolated():
    patterns = aggregate_mismatch_patterns([_mismatch("p1", "stage", "IIA", "IIB")])
    assert patterns[0]["pattern"] == "isolated"
    assert patterns[0]["n_patients"] == 1


def test_distinct_mistakes_do_not_merge():
    errors = [
        _mismatch("p1", "stage", "IIA", "IIB"),
        _mismatch("p2", "stage", "IIIA", "IIB"),  # different extracted -> different bucket
    ]
    patterns = aggregate_mismatch_patterns(errors)
    assert len(patterns) == 2
    assert all(p["pattern"] == "isolated" for p in patterns)


def test_same_patient_does_not_inflate_n_patients():
    # one patient flagged twice for the same mistake is still one patient
    errors = [
        _mismatch("p1", "stage", "IIA", "IIB"),
        _mismatch("p1", "stage", "IIA", "IIB"),
    ]
    patterns = aggregate_mismatch_patterns(errors)
    assert patterns[0]["n_patients"] == 1
    assert patterns[0]["pattern"] == "isolated"


def test_list_and_dict_values_group_via_stable_key():
    errors = [
        _mismatch("p1", "lines", ["a", "b"], ["a", "c"]),
        _mismatch("p2", "lines", ["a", "b"], ["a", "c"]),
    ]
    patterns = aggregate_mismatch_patterns(errors)
    assert len(patterns) == 1 and patterns[0]["pattern"] == "systematic"


def test_non_value_mismatch_errors_are_ignored():
    errors = [
        {"patient_id": "p1", "error_type": "missing_extraction", "expected": {}},
        _mismatch("p2", "stage", "IIA", "IIB"),
    ]
    patterns = aggregate_mismatch_patterns(errors)
    assert len(patterns) == 1 and patterns[0]["field"] == "stage"
