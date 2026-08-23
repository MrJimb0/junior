"""Reporting honesty — no silent averaging of failed/unannotated retrieval
queries, exported denominators, per-field accuracy, characterized/underpowered labels."""
from __future__ import annotations

import json
from pathlib import Path

import jr_pipeline.evaluating_pipeline_performance.run as run_mod
from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import (
    evaluate_against_ground_truth,
    power_label,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
    phi_patient_run_dir,
)


def test_power_label_thresholds():
    assert power_label(3, 0.0, 1.0) == "underpowered"        # n < 5
    assert power_label(10, 0.2, 0.95) == "underpowered"      # CI too wide (0.75 > 0.35)
    assert power_label(20, 0.70, 0.92) == "characterized"    # n>=5 and CI width 0.22


def test_run_eval_does_not_silently_average_failed_or_unannotated(tmp_path, monkeypatch):
    gold = tmp_path / "gold.jsonl"
    gold.write_text("\n".join(json.dumps(x) for x in [
        {"query_id": "q_ok", "patient_id": "P", "query_text": "t", "variable": "v",
         "relevant_chunk_ids": ["A"], "minimum_sufficient_set": [["A"]]},
        {"query_id": "q_fail", "patient_id": "P", "query_text": "t", "variable": "v",
         "relevant_chunk_ids": ["A"], "minimum_sufficient_set": [["A"]]},
        {"query_id": "q_unannotated", "patient_id": "P", "query_text": "t", "variable": "v",
         "relevant_chunk_ids": [], "minimum_sufficient_set": []},
    ]), encoding="utf-8")

    def fake_retrieve_one(*, cfg, patient_id, text, query_id, variable, code_lock_hash):
        if query_id == "q_fail":
            raise RuntimeError("retriever boom")
        return {"payload": {"candidates": [{"chunk_id": "A", "rank": 1, "score": 1.0}],
                            "selected_chunk_ids": ["A"]}}

    written: list[dict] = []
    monkeypatch.setattr(run_mod, "retrieve_one", fake_retrieve_one)
    monkeypatch.setattr(run_mod, "write_artifact", lambda env, **k: written.append(env))

    out = run_mod.run_eval(cfg={"run_id": "r", "k": 10}, gold_path=gold)
    cand = out["candidate"]
    assert cand["n_queries"] == 3
    assert cand["n_ok"] == 2 and cand["n_failed"] == 1 and cand["n_unannotated"] == 1
    assert cand["n_scored"] == 1                       # only q_ok (ok AND annotated)
    assert cand["mean_recall_at_k"][10] == 1.0         # the failed 0.0 did NOT drag the mean
    # recall denominator is exported per query
    assert {r["query_id"]: r["n_relevant"] for r in cand["per_query"]} == {
        "q_ok": 1, "q_fail": 1, "q_unannotated": 0}
    bun = out["bundle"]
    assert bun["n_bundle_scored"] == 1                 # q_fail + q_unannotated excluded
    assert bun["bundle_sufficient_rate"] == 1.0

    # the failed query records only the exception CLASS, never the raw message
    # (which could carry PHI); the metrics artifact is labeled PHI-side ("high").
    failed = next(r for r in cand["per_query"] if r["query_id"] == "q_fail")
    assert failed["error_kind"] == "RuntimeError" and "boom" not in str(failed)
    assert written and all(env["sensitivity"] == "high" for env in written)


def _feedback(dirpath: Path, pid: str, entries: list[dict]):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / f"{pid}.json").write_text(
        json.dumps({"variable": "date_of_diagnosis", "patient_id": pid, "feedback": entries}),
        encoding="utf-8")


def test_scorer_reports_per_field_denominators_and_power(tmp_path):
    run_id = "20260101_000000_aa"
    fb = clinician_feedback_dir(run_id, tmp_path)
    # p1 corrected (wrong), p2 + p3 confirmed (agreed)
    _feedback(fb, "p1", [{"type": "extraction_correction", "field": "date_of_diagnosis",
                          "correct_value": "2020-01-01"}])
    _feedback(fb, "p2", [{"type": "extraction_confirmation", "agreed": True}])
    _feedback(fb, "p3", [{"type": "extraction_confirmation", "agreed": True}])
    extract = phi_patient_run_dir(run_id, "p1", tmp_path) / "extract" / "date_of_diagnosis"
    extract.mkdir(parents=True, exist_ok=True)
    (extract / "result.json").write_text(
        json.dumps({"payload": {"data": {"date_of_diagnosis": "2099-12-31"}}}), encoding="utf-8")

    report = evaluate_against_ground_truth(run_id=run_id, variable="date_of_diagnosis", dr=tmp_path)
    assert report["n_evaluated"] == 3 and report["n_correct"] == 2  # 2 confirmed correct, p1 wrong
    assert report["power"] == "underpowered"  # n=3 < 5
    pf = report["per_field"]["date_of_diagnosis"]
    assert pf["n_evaluated"] == 3      # 1 corrected naming it + 2 confirmations
    assert pf["n_correct"] == 2        # 2 confirmations correct, p1's correction wrong
    assert pf["power"] == "underpowered"
