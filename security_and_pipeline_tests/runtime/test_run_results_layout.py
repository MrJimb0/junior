"""The review app resolves run paths through the layout module, not hand-built literals.

It honors JR_SENSITIVE_DATA_LABEL (via pipeline_run_receipts_root), validates any
env-supplied patient id before joining it into a path, and shares the run-id pattern with
its single owner in cohort_runner rather than duplicating the regex.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_RUN_RESULTS = REPO / "apps_and_interfaces/shiny_review_app/run_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_results_layout_under_test", _RUN_RESULTS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_run_id_pattern_owned_by_cohort_runner_matches_and_excludes_scratch():
    from jr_pipeline.runtime_infrastructure.cohort_runner import RUN_ID_PATTERN, _new_run_id

    run_id = _new_run_id()
    assert RUN_ID_PATTERN.fullmatch(run_id)
    # model-comparison scratch dirs must never be selected as "the latest run"
    assert not RUN_ID_PATTERN.fullmatch(run_id + "__compare__gpt5")


def test_resolve_run_rejects_unsafe_review_patient_id(monkeypatch):
    # monkeypatch, not os.environ directly, so the override is torn down after this test
    # instead of leaking JR_REVIEW_* into every later test that calls resolve_run().
    run_results = _load_module()
    monkeypatch.setenv("JR_REVIEW_RUN_ID", "20250101_120000_abcd")
    monkeypatch.setenv("JR_REVIEW_PATIENT_ID", "../escape")
    with pytest.raises(ValueError):
        run_results.resolve_run()


def test_find_latest_run_id_honors_the_sensitive_label(tmp_path, monkeypatch):
    import jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes as layout

    monkeypatch.setattr(layout, "SENSITIVE_LABEL", "SENSITIVE_TEST")
    run_results = _load_module()
    run_results.DATA_ROOT = tmp_path  # point the loader at a fresh data tree
    run_id = "20250101_120000_abcd"
    extract_dir = (
        tmp_path / "SENSITIVE_TEST" / "pipeline_run_receipts" / run_id
        / "patients" / "P1" / "extract" / "v1"
    )
    extract_dir.mkdir(parents=True)
    (extract_dir / "result.json").write_text("{}")
    assert run_results.find_latest_run_id() == run_id


def test_find_latest_run_id_short_circuits_to_the_newest_run(tmp_path, monkeypatch):
    # Two runs each with extract output: the newest is checked first and returned, so the
    # older run's patient tree is never scanned (the old code scanned every run's tree).
    import jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes as layout

    monkeypatch.setattr(layout, "SENSITIVE_LABEL", "SENSITIVE_TEST")
    run_results = _load_module()
    run_results.DATA_ROOT = tmp_path
    receipts = tmp_path / "SENSITIVE_TEST" / "pipeline_run_receipts"
    older, newer = "20250101_120000_aaaa", "20260101_120000_bbbb"
    for run_id in (older, newer):
        extract_dir = receipts / run_id / "patients" / "P1" / "extract" / "v1"
        extract_dir.mkdir(parents=True)
        (extract_dir / "result.json").write_text("{}")

    probed_runs: set[str] = set()
    real_extract_output_dir = run_results.extract_output_dir

    def spy(patient_dir):
        probed_runs.add(patient_dir.parent.parent.name)  # .../<run_id>/patients/<pid>
        return real_extract_output_dir(patient_dir)

    monkeypatch.setattr(run_results, "extract_output_dir", spy)

    assert run_results.find_latest_run_id() == newer
    assert older not in probed_runs  # short-circuited before touching the older run
