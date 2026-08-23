"""Every step that asks a model must check what the model cites -- not just the one.

The grounding check began life inside retrieve_and_prompt, which 51 of the recipe
steps use. That left llm_only and map_table_rows_and_prompt able to accept a citation
to a passage they never showed, so whether a fabricated citation was caught depended on
which step kind a recipe author happened to pick. A guard that holds for most recipes
is not a guard; it is a coin flip with good odds.

This is a source-level check on purpose. A behavioural test can only cover the step
kinds that exist today, and the failure being prevented is a step kind added tomorrow
that forgets. Wiring it wrong then fails here, with this docstring as the reason.
"""
from __future__ import annotations

from pathlib import Path

STEPS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/jr_pipeline/pipeline_steps/step_7_extract_variables/recipe_steps"
)


def _steps_that_call_a_model() -> list[Path]:
    return sorted(
        path for path in STEPS_DIR.glob("*.py")
        if "provider.chat(" in path.read_text(encoding="utf-8")
    )


def test_there_is_more_than_one_prompt_step_to_get_wrong():
    """If this ever finds one file, the premise above changed and the rest is noise."""
    assert len(_steps_that_call_a_model()) >= 3


def test_every_step_that_asks_a_model_routes_through_the_grounding_check():
    missing = [
        path.name for path in _steps_that_call_a_model()
        if "ask_until_grounded" not in path.read_text(encoding="utf-8")
    ]
    assert not missing, (
        f"these steps call a model without checking what it cites: {missing}. "
        "Route the call through ask_until_grounded (evidence_grounding.py), supplying "
        "the ids that step actually showed."
    )


def test_no_step_reimplements_the_grounding_rule():
    """One definition of 'grounded', or the copies drift and the seal means less each
    time. A step supplies its own shown-id set; it does not decide what shown means."""
    offenders = [
        path.name for path in _steps_that_call_a_model()
        if "parse_structured_evidence_pointer" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"these steps reach for the grounding primitives directly: {offenders}. "
        "The rule lives in evidence_grounding.py so the step and extract cannot disagree."
    )
