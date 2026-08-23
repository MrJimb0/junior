"""Score a pipeline run's extractions against clinician-corrected ground truth.

Returns per-variable accuracy plus a list of mismatches — the reward signal
for agentic pipeline optimization.

Reads the extraction tree (``pipeline_run_receipts/<run>/patients/<pid>/
extract/<var>/result.json``). The on-disk artifact is an envelope — the extracted
values live at ``payload.data`` — so we unwrap before comparing.

A clinician correction names the wrong ``field`` (defaulting to the variable's
canonical field) and its ``correct_value``; a patient is correct when every
corrected field matches. Date comparison is tolerance-aware: a less-precise gold
annotation (``2019-03-XX``) matches a more-precise extraction (``2019-03-01``).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from jr_pipeline.evaluating_pipeline_performance.eval_error_analysis import (
    aggregate_mismatch_patterns,
    decompose_value_mismatch,
    gold_index,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
    phi_patient_run_dir,
)

MIN_N_FOR_CI = 5  # below this, the accuracy CI is too wide to report a point estimate
CI_WIDTH_UNDERPOWERED = 0.35  # a Wilson CI wider than this is too imprecise to characterize
# The review app cuts a long displayed value short and marks it with this character. A
# confirmation carrying a marked value records agreement with text that was never shown
# in full, so it cannot be compared to another run's answer. The app owns the display
# rules and this module must not import from it, so the marker is restated here.
SHORTENED_VALUE_MARKER = "…"


def power_label(n_evaluated: int, ci_low: float, ci_high: float) -> str:
    """``characterized`` vs ``underpowered``: a result is underpowered when the
    sample is small (n < MIN_N_FOR_CI) or the Wilson CI is too wide to be informative.
    A label, never a gate — wide CIs do not fail acceptance, they are flagged honestly."""
    if n_evaluated < MIN_N_FOR_CI or (ci_high - ci_low) > CI_WIDTH_UNDERPOWERED:
        return "underpowered"
    return "characterized"


def evaluate_against_ground_truth(
    *,
    run_id: str,
    variable: str,
    dr: Path | None = None,
    retrieval_gold: list[dict] | None = None,
    gold_run_id: str | None = None,
) -> dict[str, Any]:
    """Score a run's extractions for one variable against the reviewed sample.

    The denominator is the randomly-drawn *reviewed* set — corrections (flagged wrong)
    AND confirmations (reviewed + agreed). Confirmations are essential: an errors-only
    channel is an error-enriched convenience sample with no honest denominator.
    Accuracy ships a Wilson CI and an n-guard so a tiny sample can't masquerade as a
    point estimate.

    ``gold_run_id`` names the run whose clinician review supplies the ground truth.
    It defaults to ``run_id`` — review and predictions from the same run, the ordinary
    case. Pointing it at an earlier run lets one reviewed sample score several runs: a
    different extraction model, a retrained reranker, a draft recipe.

    Scoring across runs changes what a confirmation means, and the difference is the
    whole reason this parameter exists. Within one run a confirmation IS a correct
    case: the reviewer read that run's own answer and agreed with it. Another run
    produced a different answer that nobody has reviewed, so crediting its
    confirmations would hand it a free point per confirmed patient. Across runs each
    agreed value is therefore re-scored against what this run actually extracted, and
    a confirmation that cannot be compared is left out of the denominator and counted
    in ``n_confirmations_unscorable`` rather than assumed correct.
    """
    gold_run_id = gold_run_id or run_id
    scoring_across_runs = gold_run_id != run_id
    feedback_dir = clinician_feedback_dir(gold_run_id, dr)
    corrections = _load_corrections(feedback_dir, variable)
    confirmations = _load_confirmations(feedback_dir, variable)

    reviewed = set(corrections) | set(confirmations)
    if not reviewed:
        return {
            "variable": variable,
            "n_evaluated": 0,
            "n_correct": 0,
            "n_confirmations_unscorable": 0,
            "accuracy": 0.0,
            "accuracy_ci_low": 0.0,
            "accuracy_ci_high": 0.0,
            "min_n_met": False,
            "errors": [],
            "warning": "No clinician corrections or confirmations found for this run + variable.",
        }

    n_correct = 0
    n_confirmations_unscorable = 0
    errors = []
    gold_idx = gold_index(retrieval_gold) if retrieval_gold is not None else {}
    field_eval: dict[str, int] = {}     # per-field denominator over corrected cases
    field_correct: dict[str, int] = {}

    for patient_id in sorted(reviewed):
        if patient_id not in corrections:
            if not scoring_across_runs:
                # The reviewer read this run's own answer and agreed with it, so it is
                # a correct case without re-reading the extraction.
                n_correct += 1
                continue
            # A different run answered this patient. The agreed value is an
            # expectation on the variable's canonical field like any correction, and
            # is scored against what this run actually extracted.
            agreed_value = confirmations.get(patient_id)
            rescored = _rescore_confirmation(
                agreed_value=agreed_value,
                result_file=_result_file(run_id, patient_id, variable, dr),
                variable=variable,
            )
            if rescored is None:
                n_confirmations_unscorable += 1
                continue
            field_eval[variable] = field_eval.get(variable, 0) + 1
            if rescored["agrees"]:
                field_correct[variable] = field_correct.get(variable, 0) + 1
                n_correct += 1
            else:
                mismatch_error: dict[str, Any] = {
                    "patient_id": patient_id,
                    "mismatches": {
                        variable: {
                            "extracted": rescored["extracted"],
                            "expected": agreed_value,
                        }
                    },
                    "error_type": "value_mismatch",
                }
                if retrieval_gold is not None:
                    mismatch_error["decomposition"] = decompose_value_mismatch(
                        run_id=run_id, patient_id=patient_id, variable=variable,
                        gold_entry=gold_idx.get((patient_id, variable)), dr=dr,
                    )
                errors.append(mismatch_error)
            continue

        field_expectations = corrections[patient_id]
        result_file = _result_file(run_id, patient_id, variable, dr)
        if not result_file.exists():
            for field in field_expectations:
                field_eval[field] = field_eval.get(field, 0) + 1  # missing -> field wrong
            errors.append({
                "patient_id": patient_id, "extracted": None,
                "expected": field_expectations, "error_type": "missing_extraction",
            })
            continue

        data = _extracted_data(result_file)
        mismatches = {}
        for field, expected_val in field_expectations.items():
            field_eval[field] = field_eval.get(field, 0) + 1
            actual = data.get(field) if isinstance(data, dict) else None
            if _values_match(actual, expected_val):
                field_correct[field] = field_correct.get(field, 0) + 1
            else:
                mismatches[field] = {"extracted": actual, "expected": expected_val}

        if not mismatches:
            n_correct += 1
        else:
            error: dict[str, Any] = {
                "patient_id": patient_id, "mismatches": mismatches,
                "error_type": "value_mismatch",
            }
            if retrieval_gold is not None:
                error["decomposition"] = decompose_value_mismatch(
                    run_id=run_id, patient_id=patient_id, variable=variable,
                    gold_entry=gold_idx.get((patient_id, variable)), dr=dr,
                )
            errors.append(error)

    # A confirmation whose agreed value cannot be compared to this run's answer is
    # evidence in neither direction, so it stays outside the denominator.
    n_evaluated = len(reviewed) - n_confirmations_unscorable
    # Across runs each confirmation has already been counted in the per-field
    # bookkeeping above, so none are carried here as blanket correct cases.
    n_confirmed = 0 if scoring_across_runs else len(set(confirmations) - set(corrections))
    ci_low, ci_high = wilson_ci(n_correct, n_evaluated)
    # Per-field accuracy + denominators: a confirmation agrees the whole
    # extraction, so it is a correct case for every reviewed field; a correction names
    # only the field(s) it changed. Each field's denominator is therefore the corrected
    # cases that named it plus all confirmations. Scoring across runs leaves
    # ``n_confirmed`` at zero because each agreed value has already been counted
    # against the variable's canonical field, the same way a correction is.
    per_field: dict[str, dict[str, Any]] = {}
    for field, n_eval_f in field_eval.items():
        denom = n_eval_f + n_confirmed
        corr = field_correct.get(field, 0) + n_confirmed
        f_low, f_high = wilson_ci(corr, denom)
        per_field[field] = {
            "n_evaluated": denom,
            "n_correct": corr,
            "accuracy": round(corr / denom, 4) if denom else 0.0,
            "accuracy_ci_low": f_low,
            "accuracy_ci_high": f_high,
            "power": power_label(denom, f_low, f_high),
        }
    return {
        "variable": variable,
        "n_evaluated": n_evaluated,
        "n_correct": n_correct,
        "n_confirmed": n_confirmed,
        "n_confirmations_unscorable": n_confirmations_unscorable,
        "accuracy": round(n_correct / n_evaluated, 4) if n_evaluated > 0 else 0.0,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "min_n_met": n_evaluated >= MIN_N_FOR_CI,
        "power": power_label(n_evaluated, ci_low, ci_high),
        "per_field": per_field,
        "errors": errors,
    }


def _extracted_data(result_file: Path) -> Any:
    """Unwrap the variable-result envelope and return its ``payload.data``."""
    obj = json.loads(result_file.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        payload = obj.get("payload", obj)
        if isinstance(payload, dict):
            return payload.get("data")
    return None


def _result_file(run_id: str, patient_id: str, variable: str, dr: Path | None) -> Path:
    """Where one run stores its extracted value for one patient and variable."""
    return (
        phi_patient_run_dir(run_id, patient_id, dr)
        / "extract" / variable / "result.json"
    )


def _rescore_confirmation(
    *, agreed_value: Any, result_file: Path, variable: str
) -> dict[str, Any] | None:
    """Score a run's extraction against a value a reviewer agreed with in another run,
    or return None when the two cannot honestly be compared.

    A confirmation records agreement with the value the review app *displayed*, which
    is a rendering of the stored payload rather than the payload itself: blank results
    render as "unknown", long ones are cut short, and a multi-field result renders as a
    summary of the whole structure. The comparison is therefore only sound when that
    rendering was a plain scalar. A missing, blank, or shortened agreed value, or an
    extraction whose canonical field holds a dict or list, cannot be matched back to
    what the reviewer saw and is reported as unscorable rather than guessed at — the
    denominator loses a case, which is honest, instead of the run gaining a free
    correct one.
    """
    if not isinstance(agreed_value, str) or not agreed_value.strip():
        return None
    if agreed_value.endswith(SHORTENED_VALUE_MARKER):
        return None
    if not result_file.exists():
        return {"agrees": False, "extracted": None}
    data = _extracted_data(result_file)
    extracted = data.get(variable) if isinstance(data, dict) else None
    if isinstance(extracted, (dict, list)):
        return None
    return {"agrees": _values_match(extracted, agreed_value), "extracted": extracted}


def _load_corrections(feedback_dir: Path, variable: str) -> dict[str, dict[str, Any]]:
    """patient_id -> {field: correct_value} for this variable's corrections."""
    if not feedback_dir.exists():
        return {}

    corrections: dict[str, dict[str, Any]] = {}
    for f in sorted(feedback_dir.rglob("*.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        if record.get("variable") != variable:
            continue
        patient_id = record.get("patient_id")
        if not patient_id:
            continue
        for entry in record.get("feedback", []):
            if entry.get("type") == "extraction_correction" and "correct_value" in entry:
                field = entry.get("field") or variable
                corrections.setdefault(patient_id, {})[field] = entry["correct_value"]
    return corrections


def _load_confirmations(feedback_dir: Path, variable: str) -> dict[str, Any]:
    """patient_id -> the value the reviewer agreed with, for every patient reviewed and
    AGREED on this variable (the denominator's correct cases — see
    ``feedback_capture.write_confirmation``).

    Keeping the agreed value, rather than just the patient id, is what lets one
    reviewed sample score a later run: without it a confirmation can only be taken on
    trust. The value may be absent on a record written before it was captured, which
    ``_rescore_confirmation`` treats as unscorable."""
    if not feedback_dir.exists():
        return {}
    confirmed: dict[str, Any] = {}
    for f in sorted(feedback_dir.rglob("*.json")):
        record = json.loads(f.read_text(encoding="utf-8"))
        if record.get("variable") != variable:
            continue
        patient_id = record.get("patient_id")
        if not patient_id:
            continue
        for entry in record.get("feedback", []):
            if entry.get("type") == "extraction_confirmation" and entry.get("agreed"):
                confirmed[patient_id] = entry.get("reviewed_value")
    return confirmed


# --- value matching -------------------------------------------------------

_DATE_LIKE = re.compile(r"^\d{4}(-[0-9Xx]{1,2}){0,2}$")


def _is_date_like(s: Any) -> bool:
    return isinstance(s, str) and bool(_DATE_LIKE.match(s.strip()))


def _dates_match(extracted: str, gold: str) -> bool:
    """One-directional tolerant date match.

    A wildcard (``XX``) is honoured ONLY on the gold side: a gold annotator who knew
    only the month (``2019-03-XX``) tolerates a precise extraction (``2019-03-01``).
    The reverse is a defect, not a match — a vague extraction (``2019-03-XX``) against
    a precise gold (``2019-03-01``) is under-specified and must score wrong, else the
    matcher systematically credits the model for being vaguer than the truth.
    """
    ext = (extracted.strip().split("-") + ["xx", "xx", "xx"])[:3]
    gld = (gold.strip().split("-") + ["xx", "xx", "xx"])[:3]
    for ext_c, gold_c in zip(ext, gld, strict=False):
        if not gold_c.isdigit():
            continue  # gold is unspecified at this component -> tolerate anything
        if not ext_c.isdigit() or int(ext_c) != int(gold_c):
            return False  # gold is specific; extraction must match it exactly
    return True


def _lists_match(extracted: Any, expected: Any) -> bool:
    """Order-independent tolerant list match.

    For the list-shaped recipes (event_sequence + enumeration), each expected element
    must pair with a DISTINCT extracted element and counts must match — so
    [eventA, eventB] == [eventB, eventA] with per-element date tolerance. This searches
    for ANY perfect matching by backtracking (n is tiny here), so a tolerant expected
    (e.g. a gold wildcard date ``2024-xx-xx``) does not greedily consume an extracted
    value that a later, more-specific expected needs.
    """
    if not isinstance(extracted, list) or len(extracted) != len(expected):
        return False
    used = [False] * len(extracted)

    def _assign(want_idx: int) -> bool:
        if want_idx == len(expected):
            return True
        for i, have in enumerate(extracted):
            if not used[i] and _values_match(have, expected[want_idx]):
                used[i] = True
                if _assign(want_idx + 1):
                    return True
                used[i] = False
        return False

    return _assign(0)


def _values_match(extracted: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return _lists_match(extracted, expected)

    if isinstance(expected, dict) and isinstance(extracted, dict):
        for key, expected_val in expected.items():
            if key not in extracted:
                return False
            if not _values_match(extracted[key], expected_val):
                return False
        return True

    if isinstance(expected, str) and isinstance(extracted, str):
        if _is_date_like(expected) and _is_date_like(extracted):
            return _dates_match(extracted, expected)
        return expected.strip().lower() == extracted.strip().lower()

    return expected == extracted


def characterized_accuracy_report(
    *,
    run_id: str,
    variables: list[str],
    retrieval_gold: list[dict] | None = None,
    dr: Path | None = None,
    gold_run_id: str | None = None,
) -> dict[str, Any]:
    """The Condition-4 characterized-accuracy deliverable: for every variable, the
    per-variable accuracy with per-field denominators, Wilson CIs, and
    characterized/underpowered labels, plus — when ``retrieval_gold`` is supplied — the
    retrieval-vs-extraction-miss decomposition on each value mismatch; and across all
    variables, the systematic-vs-isolated mismatch patterns. A runs-and-reports
    artifact: wide CIs are characterized honestly, not failed.

    ``gold_run_id`` scores this run against another run's clinician review; see
    ``evaluate_against_ground_truth`` for what that changes about confirmations."""
    per_variable = [
        evaluate_against_ground_truth(
            run_id=run_id, variable=v, dr=dr, retrieval_gold=retrieval_gold,
            gold_run_id=gold_run_id,
        )
        for v in variables
    ]
    all_errors = [e for r in per_variable for e in r["errors"]]
    return {
        "run_id": run_id,
        "gold_run_id": gold_run_id or run_id,
        "per_variable": per_variable,
        "systematic_patterns": aggregate_mismatch_patterns(all_errors),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. Robust at small n
    and near 0/1 where the normal approximation fails. z=1.96 → 95%."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))
