"""Finalize step for date_of_diagnosis.

Each of the four chart-reading passes returns the same three dates (the original
diagnosis, the locoregional recurrence, and the metastatic diagnosis). This step
combines those four answers into one:

  * for each event, a filled-in date wins (a later pass overwrites an earlier one
    only when it actually supplies a date for that event);
  * pass 4 (repair_loco_met) may blank out or reclassify the locoregional/metastatic
    dates, but the ORIGINAL diagnosis date is protected (removed from pass 4's answer
    before combining, so pass 4 can never touch it);
  * sets ``date_of_diagnosis`` = the original date — the canonical value other recipes
    read.

No model call. See date_of_diagnosis_v1_recipe.yaml for the step that invokes it.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

# A date wins only if it actually looks like a date (YYYY-MM-DD, or with unknown parts
# as YYYY-MM-XX / YYYY-XX-XX), so a junk 0 / False / "" from a small model can't be
# mistaken for a real date. The date-checking helper is shared across recipes;
# test_dx_merge imports _is_date from this module, so keep the name.
_pd = load_shared_validation_rule("partial_date", __file__)
_is_date = _pd.is_date

# event -> (date_key, certainty_key, evidence_chunk_id_key).
# Evidence is stored as a POINTER (the id of the cited chart snippet/chunk), not as a
# pasted-in quote. The finalize only ever emits these keys, so even if the model dumps a
# whole report into a free-text field, that text cannot reach the output. A reviewer
# opens the cited snippet via the pointer to see the supporting text.
_TRIPLETS = {
    "original": (
        "date_original_diagnosis", "original_certainty", "original_evidence_chunk_id",
    ),
    "locoregional": (
        "date_locoregional_recurrence_diagnosis", "locoregional_recurrence_certainty",
        "locoregional_recurrence_evidence_chunk_id",
    ),
    "metastatic": (
        "date_metastatic_diagnosis", "metastatic_certainty", "metastatic_evidence_chunk_id",
    ),
}

_RECLASSIFIABLE_DATES = frozenset({
    "date_locoregional_recurrence_diagnosis",
    "date_metastatic_diagnosis",
})


def _empty_result() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for date_k, cert_k, cid_k in _TRIPLETS.values():
        out[date_k] = None
        out[cert_k] = 0
        out[cid_k] = None
    out["rationale"] = ""
    return out


def _merge_one(best: dict[str, Any], cand: Any, *, allow_null_dates: frozenset[str] = frozenset()) -> None:
    """Merge one pass's output into the accumulator."""
    if not isinstance(cand, dict):
        return
    updated = False
    for date_k, cert_k, cid_k in _TRIPLETS.values():
        if _is_date(cand.get(date_k)):  # a valid date wins for its event
            best[date_k] = cand[date_k]
            # Only overwrite the certainty / source pointer when this pass actually
            # provides them: a brief re-confirmation that omits them (small models often
            # return only the fields they changed) must NOT wipe out the source pointer an
            # earlier pass already set.
            if cand.get(cert_k) is not None:
                best[cert_k] = cand.get(cert_k)
            if cand.get(cid_k):
                best[cid_k] = cand.get(cid_k)
            updated = True
        elif date_k in cand and date_k in allow_null_dates:
            # the pass EXPLICITLY returned a non-date (blank/"") for an event it is allowed
            # to reclassify -> blank it out. A MISSING key instead means "no opinion" (small
            # models return only the fields they changed) and must NOT erase a date an
            # earlier pass found.
            best[date_k] = None
            best[cert_k] = cand.get(cert_k, 0)
            best[cid_k] = cand.get(cid_k)
            updated = True
    if updated and cand.get("rationale"):
        # rationale is the one remaining free-text field; cap its length to the schema
        # limit so a wordy model can't exceed maxLength (defensive, like storing evidence
        # as a pointer rather than a quote).
        best["rationale"] = str(cand["rationale"])[:600]


def merge_passes(ctx: StepContext) -> StepResult:
    steps = ctx.step_outputs or {}

    def data_of(step_id: str) -> Any:
        return (steps.get(step_id) or {}).get("data")

    best = _empty_result()
    _merge_one(best, data_of("pathology_snippets"))
    _merge_one(best, data_of("pathology_full"))
    _merge_one(best, data_of("refine_clinical"))

    # Pass 4 repairs locoregional/metastatic only — remove the original-event keys from
    # its answer so it can never overwrite the original diagnosis date, and allow it to
    # blank out the locoregional/metastatic dates when reclassifying them.
    p4 = data_of("repair_loco_met")
    if isinstance(p4, dict):
        p4 = {
            k: v for k, v in p4.items()
            if not k.startswith("original") and k != "date_original_diagnosis"
        }
    _merge_one(best, p4, allow_null_dates=_RECLASSIFIABLE_DATES)

    # A date no pass cited keeps its null pointer. This used to attach read[0] — the
    # first chunk of the first pass, whichever pass actually supplied the date — on the
    # reasoning that a found date should never ship without provenance. It ships with
    # provenance either way; the question is whether the provenance is true. A borrowed
    # pointer names a passage that was genuinely read, so it satisfies the grounding
    # gate by construction, and the answer table then shows a real quote next to a value
    # it does not support. The step re-asks for a missing citation now
    # (evidence_grounding.ask_until_grounded); what is still uncited after that is
    # reported as an unsupported claim, which is what it is.

    # The canonical diagnosis date other recipes (e.g. treatment_lines) read.
    best["date_of_diagnosis"] = best.get("date_original_diagnosis")

    receipt = {
        "python_merge": {
            "passes_seen": [
                sid for sid in
                ("pathology_snippets", "pathology_full", "refine_clinical", "repair_loco_met")
                if isinstance(data_of(sid), dict)
            ],
            "date_of_diagnosis": best["date_of_diagnosis"],
            "date_locoregional_recurrence_diagnosis": best["date_locoregional_recurrence_diagnosis"],
            "date_metastatic_diagnosis": best["date_metastatic_diagnosis"],
        }
    }
    return StepResult(data=best, receipt_payload=receipt)
