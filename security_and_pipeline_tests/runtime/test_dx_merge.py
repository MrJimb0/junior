"""Pin the date_of_diagnosis merge semantics.

A merge that can't tell a MISSING key (the model emitted only changed fields)
from an EXPLICIT null (reclassify) silently destroys a date an earlier pass found.
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Collection-agnostic: the recipe may sit under any collection dir.
_HELPER = next((REPO / "var_extraction_recipes").rglob("date_of_diagnosis_v1_python_helper.py"))


def _load():
    spec = importlib.util.spec_from_file_location("dod_helper_under_test", _HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_merge_missing_key_vs_explicit_null():
    h = _load()
    best = h._empty_result()

    # Pass A finds original + metastatic.
    h._merge_one(best, {
        "date_original_diagnosis": "2019-03-01",
        "date_metastatic_diagnosis": "2021-06-01",
    })
    assert best["date_original_diagnosis"] == "2019-03-01"
    assert best["date_metastatic_diagnosis"] == "2021-06-01"

    # Pass B OMITS metastatic -> must NOT null it (no opinion), even though
    # metastatic is reclassifiable.
    h._merge_one(
        best,
        {"date_locoregional_recurrence_diagnosis": "2020-01-01"},
        allow_null_dates=h._RECLASSIFIABLE_DATES,
    )
    assert best["date_metastatic_diagnosis"] == "2021-06-01", "missing key clobbered a found date"
    assert best["date_locoregional_recurrence_diagnosis"] == "2020-01-01"

    # Pass C EXPLICITLY nulls metastatic (reclassification) -> nulled.
    h._merge_one(best, {"date_metastatic_diagnosis": None}, allow_null_dates=h._RECLASSIFIABLE_DATES)
    assert best["date_metastatic_diagnosis"] is None

    # Malformed values (0/False/"") never win as a date; original is protected
    # (not reclassifiable); an explicit non-date on a reclassifiable key nulls it.
    h._merge_one(
        best,
        {"date_original_diagnosis": 0, "date_locoregional_recurrence_diagnosis": False},
        allow_null_dates=h._RECLASSIFIABLE_DATES,
    )
    assert best["date_original_diagnosis"] == "2019-03-01", "malformed value won as a date"
    assert best["date_locoregional_recurrence_diagnosis"] is None


def test_is_date():
    h = _load()
    assert h._is_date("2019-03-01")
    assert h._is_date("2019-03-XX")
    assert h._is_date("2019-XX-XX")
    assert not h._is_date("")
    assert not h._is_date(0)
    assert not h._is_date(None)
    assert not h._is_date("March 2019")
