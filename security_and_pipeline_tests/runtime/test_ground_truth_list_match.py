"""List match credits a correct extraction even when a gold wildcard date precedes a
more specific one.

A greedy first-match bijection would let a tolerant gold (2024-xx-xx) consume an extracted
value a later, more specific gold (2024-03-15) needs, scoring a correct extraction wrong. The
matcher searches for any perfect matching instead (hits event_sequence recipes).
"""
from __future__ import annotations

from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import _lists_match


def test_gold_wildcard_before_specific_still_matches():
    # greedy pairs 2024-xx-xx with 2024-03-15 then fails the specific gold; the valid
    # bijection is 2024-xx-xx<->2024-05-20, 2024-03-15<->2024-03-15 -> must score True.
    gold = ["2024-xx-xx", "2024-03-15"]
    extracted = ["2024-03-15", "2024-05-20"]
    assert _lists_match(extracted, gold) is True


def test_order_independent_match():
    assert _lists_match(["b", "a"], ["a", "b"]) is True


def test_missing_element_scores_wrong():
    assert _lists_match(["2024-03-15"], ["2024-03-15", "2024-05-20"]) is False


def test_spurious_element_scores_wrong():
    assert _lists_match(["a", "b", "c"], ["a", "b"]) is False


def test_no_perfect_matching_scores_wrong():
    # one extracted specific cannot satisfy two distinct specific golds
    assert _lists_match(["2024-03-15", "2024-03-15"], ["2024-03-15", "2024-05-20"]) is False
