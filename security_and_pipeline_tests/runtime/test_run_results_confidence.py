"""Confidence rendering in the review app: real-run integers vs demo fractions.

Recipe output schemas emit an integer 0..100 confidence, scaled to a 0..1 fraction for
display; demo rows already carry a 0..1 fraction and pass through unchanged. Scaling is by
known source, not by sniffing magnitude — so an integer 1 (= 1%) renders as 0.01, not 1.00.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_RUN_RESULTS = REPO / "apps_and_interfaces/shiny_review_app/run_results.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_results_under_test", _RUN_RESULTS)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's dataclasses can resolve their own module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_real_run_integer_confidence_scaled_by_hundred():
    run_results = _load_module()
    _value, confidence, _ids, _rationale = run_results._summarize_data(
        {"date_of_birth": "1970-01-01", "dob_certainty": 1}, "date_of_birth"
    )
    assert confidence == 0.01  # 1% renders as 0.01, not 1.00


def test_real_run_full_confidence_is_one():
    run_results = _load_module()
    _value, confidence, _ids, _rationale = run_results._summarize_data(
        {"date_of_birth": "1970-01-01", "dob_certainty": 100}, "date_of_birth"
    )
    assert confidence == 1.0

