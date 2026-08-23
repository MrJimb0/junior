"""Last-resort estimate for date of birth (DOB).

When earlier steps found no stated DOB anywhere in the chart, estimate the birth YEAR by
subtracting the patient's age (``age_years``) from the year of a document
(``doc_date`` -- a note's date). Produces month/day as unknown (XX). Writes an output in
the exact shape the recipe expects so its validation passes.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

# Partial-date parsing lives in one shared place; year_from_date tolerates the trailing
# time that a document date (doc_date) legitimately carries (e.g. "2019-04-01 09:30").
_pd = load_shared_validation_rule("partial_date", __file__)
_year_from_date = _pd.year_from_date
# Shared helper that attaches a citation (a source chunk id) to an answer. The age-based
# estimate creates new value-bearing fields, so it must carry a citation -- otherwise the
# answer would ship with no source to point back to.
_grounding_chunk = load_shared_validation_rule("evidence_grounding", __file__).grounding_chunk


def estimate_from_age(ctx: StepContext) -> StepResult:
    """Estimate the birth year from the dedicated age/date fallback when available."""
    # Walk the earlier steps' outputs in order, keeping the most recent one that has data.
    data: dict[str, Any] = {}
    for _sid, payload in ctx.step_outputs.items():
        inner = (payload or {}).get("data") or {}
        if inner:
            data = dict(inner)

    age = data.get("age_years")
    doc_date = data.get("doc_date")
    estimated: str | None = None
    method: str | None = None

    if isinstance(age, int) and 0 < age < 121:
        year = _year_from_date(doc_date)
        if year is not None:
            est_year = year - age
            if 1900 <= est_year <= year:
                estimated = f"{est_year:04d}-XX-XX"
                method = "doc_date_year_minus_age_years"

    data.setdefault("date_of_birth", None)
    data.setdefault("dob_certainty", 0)
    data.setdefault("dob_evidence", "unclear")
    data.setdefault("rationale", "No DOB extracted; estimated from age + document date." if estimated else "No DOB available.")
    data["estimated_date_of_birth"] = estimated
    data["estimated_dob_method"] = method

    # When we land on the age-based estimate, the value-bearing fields
    # (estimated_date_of_birth / age_years / doc_date) must carry a citation. Cite the chart
    # region the keyword-search step looked at (or, failing that, the demographics step):
    # the model's own dob_evidence_chunk_id if that step actually saw it, and nothing
    # that step read.
    read = (
        ((ctx.step_outputs.get("age_date") or {}).get("evidence_chunk_ids") or [])
        or ((ctx.step_outputs.get("bm25_explicit") or {}).get("evidence_chunk_ids") or [])
        or ((ctx.step_outputs.get("demographics_grab") or {}).get("evidence_chunk_ids") or [])
    )
    has_value = estimated is not None or data.get("age_years") is not None or data.get("doc_date") is not None
    if has_value and read:
        data["dob_evidence_chunk_id"] = _grounding_chunk(data, read, "dob_evidence_chunk_id")

    receipt = {
        "estimate": {"age_years": age, "doc_date": doc_date, "estimated_date_of_birth": estimated, "method": method},
    }
    return StepResult(data=data, receipt_payload=receipt)
