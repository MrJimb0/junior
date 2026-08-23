"""retrieval_miss / extraction_miss / both decomposition of a value_mismatch."""
from __future__ import annotations

import json
from pathlib import Path

from jr_pipeline.evaluating_pipeline_performance.eval_error_analysis import (
    chunks_fed_to_llm,
    decompose_value_mismatch,
)
from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import (
    evaluate_against_ground_truth,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
    phi_patient_run_dir,
)

RUN = "20260101_000000_aa"
PID = "Test_Patient"
VAR = "date_of_diagnosis"


def _write_step_receipt(dr: Path, step_id: str, included: list[str]) -> None:
    steps = phi_patient_run_dir(RUN, PID, dr) / "extract" / VAR / "steps" / step_id
    steps.mkdir(parents=True, exist_ok=True)
    (steps / "receipt.json").write_text(
        json.dumps({"payload": {"evidence_packet": {"included_chunk_ids": included}}}),
        encoding="utf-8",
    )


def _gold(min_sets, relevant):
    return {"patient_id": PID, "variable": VAR, "minimum_sufficient_set": min_sets, "relevant_chunk_ids": relevant}


def test_chunks_fed_unions_across_steps(tmp_path):
    _write_step_receipt(tmp_path, "s1", ["A", "B"])
    _write_step_receipt(tmp_path, "s2", ["B", "C"])
    assert chunks_fed_to_llm(RUN, PID, VAR, tmp_path) == {"A", "B", "C"}


def test_extraction_miss_when_a_sufficient_set_was_fed(tmp_path):
    _write_step_receipt(tmp_path, "s1", ["A", "X"])  # A covers the sufficient set
    out = decompose_value_mismatch(
        run_id=RUN, patient_id=PID, variable=VAR,
        gold_entry=_gold([["A"]], ["A"]), dr=tmp_path,
    )
    assert out["decomposition"] == "extraction_miss"
    assert out["bundle_sufficient"] is True


def test_retrieval_miss_when_no_relevant_chunk_reached_the_model(tmp_path):
    _write_step_receipt(tmp_path, "s1", ["X", "Y"])  # nothing relevant
    out = decompose_value_mismatch(
        run_id=RUN, patient_id=PID, variable=VAR,
        gold_entry=_gold([["A"]], ["A"]), dr=tmp_path,
    )
    assert out["decomposition"] == "retrieval_miss"
    assert out["n_relevant_fed"] == 0


def test_both_when_partial_relevant_but_insufficient(tmp_path):
    _write_step_receipt(tmp_path, "s1", ["A"])  # A is relevant but [A,B] needed
    out = decompose_value_mismatch(
        run_id=RUN, patient_id=PID, variable=VAR,
        gold_entry=_gold([["A", "B"]], ["A", "B"]), dr=tmp_path,
    )
    assert out["decomposition"] == "both"
    assert out["bundle_sufficient"] is False
    assert out["n_relevant_fed"] == 1


def test_no_gold_is_unclassifiable(tmp_path):
    out = decompose_value_mismatch(
        run_id=RUN, patient_id=PID, variable=VAR, gold_entry=None, dr=tmp_path,
    )
    assert out["decomposition"] == "no_retrieval_gold"


def test_scorer_annotates_value_mismatch_with_decomposition(tmp_path):
    # a clinician correction that disagrees with the extraction -> value_mismatch
    fb = clinician_feedback_dir(RUN, tmp_path)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "f.json").write_text(json.dumps({
        "variable": VAR, "patient_id": PID,
        "feedback": [{"type": "extraction_correction", "field": "date_of_diagnosis", "correct_value": "2020-01-01"}],
    }), encoding="utf-8")
    extract = phi_patient_run_dir(RUN, PID, tmp_path) / "extract" / VAR
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "result.json").write_text(
        json.dumps({"payload": {"data": {"date_of_diagnosis": "2099-12-31"}}}), encoding="utf-8")
    _write_step_receipt(tmp_path, "pass1", ["X"])  # retrieval missed -> retrieval_miss

    gold = [_gold([["A"]], ["A"])]
    report = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path, retrieval_gold=gold)
    mismatch = next(e for e in report["errors"] if e["error_type"] == "value_mismatch")
    assert mismatch["decomposition"]["decomposition"] == "retrieval_miss"


def test_characterized_accuracy_report_has_all_condition4_sections(tmp_path):
    from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import (
        characterized_accuracy_report,
    )
    fb = clinician_feedback_dir(RUN, tmp_path)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "f.json").write_text(json.dumps({
        "variable": VAR, "patient_id": PID,
        "feedback": [{"type": "extraction_correction", "field": "date_of_diagnosis", "correct_value": "2020-01-01"}],
    }), encoding="utf-8")
    extract = phi_patient_run_dir(RUN, PID, tmp_path) / "extract" / VAR
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "result.json").write_text(
        json.dumps({"payload": {"data": {"date_of_diagnosis": "2099-12-31"}}}), encoding="utf-8")
    _write_step_receipt(tmp_path, "pass1", ["A"])  # a sufficient set WAS fed -> extraction_miss

    report = characterized_accuracy_report(
        run_id=RUN, variables=[VAR], retrieval_gold=[_gold([["A"]], ["A"])], dr=tmp_path)

    assert "systematic_patterns" in report
    var_report = report["per_variable"][0]
    assert "per_field" in var_report and "power" in var_report  # per-field + char/underpowered
    mismatch = next(e for e in var_report["errors"] if e["error_type"] == "value_mismatch")
    assert mismatch["decomposition"]["decomposition"] == "extraction_miss"
    assert report["systematic_patterns"][0]["pattern"] == "isolated"  # single patient


def test_scorer_omits_decomposition_when_no_gold_passed(tmp_path):
    # callers that pass no retrieval_gold get no decomposition key
    fb = clinician_feedback_dir(RUN, tmp_path)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "f.json").write_text(json.dumps({
        "variable": VAR, "patient_id": PID,
        "feedback": [{"type": "extraction_correction", "field": "date_of_diagnosis", "correct_value": "2020-01-01"}],
    }), encoding="utf-8")
    extract = phi_patient_run_dir(RUN, PID, tmp_path) / "extract" / VAR
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "result.json").write_text(
        json.dumps({"payload": {"data": {"date_of_diagnosis": "2099-12-31"}}}), encoding="utf-8")

    report = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    mismatch = next(e for e in report["errors"] if e["error_type"] == "value_mismatch")
    assert "decomposition" not in mismatch
