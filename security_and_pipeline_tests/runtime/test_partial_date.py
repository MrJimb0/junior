"""Pin the partial-date parsing semantics.

Six recipe helpers + the clinical invariants share one partial_date module. These
tests pin its exact behaviors — including the two deliberately different non-date
defaults (None vs sort-last) and the time-tolerant year extraction — so the shared
module cannot silently change any of them.
"""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

REPO = Path(__file__).resolve().parents[2]
pd = load_shared_validation_rule("partial_date", REPO / "var_extraction_recipes")


def test_is_date_strict_no_strip():
    assert pd.is_date("2024-02-15")
    assert pd.is_date("2019-03-XX")
    assert pd.is_date("2019-XX-XX")
    assert not pd.is_date("2024-2-15")          # single-digit month
    assert not pd.is_date("2024-02-15T10:00")   # no trailing time
    assert not pd.is_date(" 2024-02-15")        # is_date does NOT strip
    assert not pd.is_date("not a date")
    assert not pd.is_date(None)
    assert not pd.is_date(5)


def test_date_key_returns_none_for_non_dates_and_zeros_masks():
    assert pd.date_key("2024-02-15") == (2024, 2, 15)
    assert pd.date_key("2020-XX-XX") == (2020, 0, 0)
    assert pd.date_key(" 2020-03-XX ") == (2020, 3, 0)   # date_key strips
    assert pd.date_key("nope") is None
    assert pd.date_key(None) is None
    assert pd.date_key(5) is None


def test_date_sort_key_sorts_undatable_last():
    assert pd.date_sort_key("2024-02-15") == (2024, 2, 15)
    assert pd.date_sort_key("nope") == (9999, 99, 99)
    assert pd.date_sort_key(None) == (9999, 99, 99)
    # the two defaults are the only difference between the variants
    for v in ("2024-02-15", "2020-XX-XX", "nope", None, 5):
        assert pd.date_sort_key(v) == (pd.date_key(v) or (9999, 99, 99))


def test_strictly_before_is_precision_aware():
    assert pd.strictly_before((2020, 1, 1), (2024, 2, 15))
    assert not pd.strictly_before((2024, 2, 15), (2020, 1, 1))
    assert not pd.strictly_before((2020, 0, 0), (2020, 3, 0))   # masked -> cannot assert
    assert not pd.strictly_before((2020, 1, 1), (2020, 1, 1))   # equal


def test_month_ordinal():
    assert pd.month_ordinal((2020, 3, 0)) == 2020 * 12 + 3
    assert pd.month_ordinal((2020, 0, 0)) is None


def test_year_from_date_tolerates_a_trailing_time():
    assert pd.year_from_date("2024-02-15") == 2024
    assert pd.year_from_date("2024-02-15T10:30:00") == 2024
    assert pd.year_from_date("2024-02-15 09:30:00") == 2024
    assert pd.year_from_date("2024-XX-XX") == 2024
    assert pd.year_from_date("nope") is None
    assert pd.year_from_date(None) is None
