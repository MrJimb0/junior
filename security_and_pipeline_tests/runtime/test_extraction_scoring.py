"""Extraction-value scoring reads the extraction tree and is date-tolerant.

Guards that the scorer reads ``extract/<var>/result.json``, unwraps the result
envelope to ``payload.data`` before comparing (not the whole envelope), and matches
dates tolerantly so a less-precise gold date still matches a precise extraction.
"""
import json
from pathlib import Path

from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import (
    _dates_match,
    _values_match,
    evaluate_against_ground_truth,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
    phi_patient_run_dir,
)

RUN = "20990101_000000"
LATER_RUN = "20990202_000000"
VAR = "date_of_diagnosis"


def _write_result(dr: Path, patient_id: str, data: dict, run_id: str = RUN) -> None:
    """Write a live variable-result envelope (payload.data is what the scorer reads)."""
    out = phi_patient_run_dir(run_id, patient_id, dr) / "extract" / VAR / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"artifact_type": "variable_result", "payload": {"variable": VAR, "data": data}}
    out.write_text(json.dumps(envelope), encoding="utf-8")


def _write_feedback(dr: Path, patient_id: str, field: str, correct_value, run_id: str = RUN) -> None:
    fb_dir = clinician_feedback_dir(run_id, dr)
    fb_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "patient_id": patient_id,
        "variable": VAR,
        "run_id": run_id,
        "annotator": "test",
        "feedback": [{"type": "extraction_correction", "field": field, "correct_value": correct_value}],
    }
    (fb_dir / f"{patient_id}__{VAR}.json").write_text(json.dumps(record), encoding="utf-8")


def _write_confirmation(dr: Path, patient_id: str, reviewed_value, run_id: str = RUN) -> None:
    """Record that a reviewer read this run's answer for the patient and agreed."""
    fb_dir = clinician_feedback_dir(run_id, dr)
    fb_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "patient_id": patient_id,
        "variable": VAR,
        "run_id": run_id,
        "annotator": "test",
        "feedback": [{
            "type": "extraction_confirmation",
            "field": VAR,
            "reviewed_value": reviewed_value,
            "agreed": True,
        }],
    }
    (fb_dir / f"{patient_id}__{VAR}__confirm.json").write_text(json.dumps(record), encoding="utf-8")


def test_dates_match_one_directional():
    # a GOLD wildcard tolerates a precise extraction (annotator knew less)
    assert _dates_match("2019-03-01", "2019-03-XX")
    assert _dates_match("2019-08-15", "2019-XX-XX")
    # but a VAGUE extraction against a PRECISE gold must score wrong (no upward bias)
    assert not _dates_match("2019-03-XX", "2019-03-01")
    assert not _dates_match("2019-XX-XX", "2019-08-15")
    # zero-pad insensitive; real disagreements fail
    assert _dates_match("2019-03-01", "2019-03-1")
    assert not _dates_match("2019-03-01", "2019-04-01")
    assert not _dates_match("2019-03-01", "2020-03-XX")


def test_values_match_non_dates():
    assert _values_match("Mastectomy", "mastectomy ")
    assert _values_match({"a": "X", "b": "2019-03-01"}, {"a": "x", "b": "2019-03-XX"})
    assert not _values_match("lumpectomy", "mastectomy")


def test_live_tree_date_tolerant_match(tmp_path):
    # Extraction is precise; the gold annotation only knew the month.
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2019-03-01", "noise": "x"})
    _write_feedback(tmp_path, "p1", "date_of_diagnosis", "2019-03-XX")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_evaluated"] == 1
    assert res["n_correct"] == 1
    assert res["accuracy"] == 1.0
    assert res["errors"] == []


def test_live_tree_value_mismatch(tmp_path):
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2018-01-01"})
    _write_feedback(tmp_path, "p1", "date_of_diagnosis", "2019-03-XX")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_correct"] == 0
    assert res["errors"][0]["error_type"] == "value_mismatch"
    assert res["errors"][0]["mismatches"]["date_of_diagnosis"]["extracted"] == "2018-01-01"


def test_missing_extraction(tmp_path):
    # Feedback exists but no result.json was written for this patient.
    _write_feedback(tmp_path, "p1", "date_of_diagnosis", "2019-03-XX")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_correct"] == 0
    assert res["errors"][0]["error_type"] == "missing_extraction"


def test_field_defaults_to_variable_name(tmp_path):
    # A correction with no explicit "field" targets the variable's canonical field.
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2019-03-01"})
    fb_dir = clinician_feedback_dir(RUN, tmp_path)
    fb_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "patient_id": "p1", "variable": VAR, "run_id": RUN, "annotator": "test",
        "feedback": [{"type": "extraction_correction", "correct_value": "2019-03-01"}],
    }
    (fb_dir / "p1__x.json").write_text(json.dumps(record), encoding="utf-8")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_correct"] == 1


def test_lists_match_order_independent_and_tolerant():
    from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import _lists_match
    # order-independent + per-element date tolerance
    assert _lists_match(
        [{"date": "2019-03-01", "type": "chemo"}, {"date": "2020-01-15", "type": "surgery"}],
        [{"type": "surgery", "date": "2020-01-XX"}, {"type": "chemo", "date": "2019-03-XX"}],
    )
    # a missing element scores wrong (the relational/temporal core)
    assert not _lists_match([{"type": "chemo"}], [{"type": "chemo"}, {"type": "surgery"}])
    # a spurious extra element scores wrong
    assert not _lists_match([{"type": "chemo"}, {"type": "surgery"}], [{"type": "chemo"}])


def test_wilson_ci_and_n_guard():
    from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import wilson_ci
    low, high = wilson_ci(8, 10)
    assert 0.0 <= low < 0.8 < high <= 1.0      # interval brackets the point estimate
    assert wilson_ci(0, 0) == (0.0, 0.0)       # no data -> degenerate
    lo1, hi1 = wilson_ci(1, 1)                  # tiny n -> very wide, not (1,1)
    assert lo1 < 1.0


def test_confirmations_supply_the_denominator(tmp_path):
    # A confirmed (agreed) case is a correct evaluated case — it supplies the
    # denominator the errors-only channel lacks. Here 1 confirmed + 1 corrected-correct = 2/2.
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2019-03-01"})
    _write_feedback(tmp_path, "p1", "date_of_diagnosis", "2019-03-XX")  # correction, matches
    fb_dir = clinician_feedback_dir(RUN, tmp_path)
    (fb_dir / "p2__date_of_diagnosis.json").write_text(json.dumps({
        "patient_id": "p2", "variable": VAR, "run_id": RUN, "annotator": "t",
        "feedback": [{"type": "extraction_confirmation", "agreed": True, "field": VAR}],
    }), encoding="utf-8")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_evaluated"] == 2 and res["n_correct"] == 2
    assert res["n_confirmed"] == 1
    assert res["accuracy"] == 1.0
    assert "accuracy_ci_low" in res and "accuracy_ci_high" in res
    assert res["min_n_met"] is False  # n=2 < MIN_N_FOR_CI


# --- scoring one run against another run's review ---------------------------
# A reviewed sample is expensive, so it has to be reusable: the same clinician review
# should be able to characterize a new extraction model, a retrained reranker, or a
# draft recipe. That only works if a confirmation is re-scored against the run being
# measured instead of being taken on trust, which is what these guard.

def test_confirmation_is_credited_within_its_own_run(tmp_path):
    # Same-run scoring is unchanged: the reviewer read THIS run's answer and agreed, so
    # it is a correct case without re-reading the extraction at all.
    _write_confirmation(tmp_path, "p1", "2019-03-01")

    res = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert res["n_evaluated"] == 1 and res["n_correct"] == 1
    assert res["n_confirmations_unscorable"] == 0


def test_confirmation_from_another_run_is_rescored_and_can_fail(tmp_path):
    # The defect this fixes: a later run that answers a confirmed patient DIFFERENTLY
    # must lose the point, not inherit it.
    _write_confirmation(tmp_path, "p1", "2019-03-01", run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2021-07-04"}, run_id=LATER_RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_evaluated"] == 1
    assert res["n_correct"] == 0
    assert res["accuracy"] == 0.0
    assert res["errors"][0]["error_type"] == "value_mismatch"
    assert res["errors"][0]["mismatches"][VAR]["extracted"] == "2021-07-04"
    assert res["errors"][0]["mismatches"][VAR]["expected"] == "2019-03-01"
    # The same run's own review still credits it, so the two runs score differently
    # off one reviewed sample — which is the point.
    same_run = evaluate_against_ground_truth(run_id=RUN, variable=VAR, dr=tmp_path)
    assert same_run["n_correct"] == 1


def test_confirmation_from_another_run_that_agrees_is_correct(tmp_path):
    _write_confirmation(tmp_path, "p1", "2019-03-01", run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2019-03-01"}, run_id=LATER_RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_evaluated"] == 1 and res["n_correct"] == 1
    assert res["accuracy"] == 1.0
    assert res["errors"] == []


def test_corrections_are_read_from_the_gold_run(tmp_path):
    # The correction lives on the reviewed run; the prediction being scored lives on
    # the later one. Without gold_run_id there is no reviewed sample to score against.
    _write_feedback(tmp_path, "p1", VAR, "2019-03-XX", run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2019-03-01"}, run_id=LATER_RUN)

    unscored = evaluate_against_ground_truth(run_id=LATER_RUN, variable=VAR, dr=tmp_path)
    assert unscored["n_evaluated"] == 0

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_evaluated"] == 1 and res["n_correct"] == 1


def test_confirmation_without_a_recorded_value_is_unscorable_not_correct(tmp_path):
    # An agreed value that was never recorded cannot be compared to another run's
    # answer. It leaves the denominator rather than handing out a free point.
    _write_confirmation(tmp_path, "p1", None, run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2021-07-04"}, run_id=LATER_RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_confirmations_unscorable"] == 1
    assert res["n_evaluated"] == 0
    assert res["n_correct"] == 0
    assert res["accuracy"] == 0.0
    assert res["errors"] == []


def test_shortened_and_structured_confirmations_are_unscorable(tmp_path):
    # The review app displays a cut-short value with a marker, and renders a
    # multi-field result as a summary of the whole structure. Neither can be matched
    # back to a stored value, so neither is scored.
    _write_confirmation(tmp_path, "p1", "a very long treatment narrative…", run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2021-07-04"}, run_id=LATER_RUN)
    _write_confirmation(tmp_path, "p2", "2019-03-01", run_id=RUN)
    _write_result(tmp_path, "p2", {"date_of_diagnosis": {"value": "2019-03-01"}}, run_id=LATER_RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_confirmations_unscorable"] == 2
    assert res["n_evaluated"] == 0


def test_missing_extraction_for_a_confirmed_patient_scores_wrong(tmp_path):
    # A run that produced nothing for a confirmed patient is wrong, not unscorable —
    # the comparison is possible and it fails.
    _write_confirmation(tmp_path, "p1", "2019-03-01", run_id=RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_evaluated"] == 1 and res["n_correct"] == 0
    assert res["n_confirmations_unscorable"] == 0
    assert res["errors"][0]["mismatches"][VAR]["extracted"] is None


def test_per_field_accuracy_does_not_credit_a_failed_confirmation(tmp_path):
    # The per-field denominators add confirmations as correct for every field. Across
    # runs that would re-introduce the same free point one level down.
    _write_confirmation(tmp_path, "p1", "2019-03-01", run_id=RUN)
    _write_result(tmp_path, "p1", {"date_of_diagnosis": "2021-07-04"}, run_id=LATER_RUN)

    res = evaluate_against_ground_truth(
        run_id=LATER_RUN, variable=VAR, dr=tmp_path, gold_run_id=RUN,
    )
    assert res["n_confirmed"] == 0
    assert res["per_field"][VAR]["n_evaluated"] == 1
    assert res["per_field"][VAR]["n_correct"] == 0
