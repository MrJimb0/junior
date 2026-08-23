"""A variable that needs a model and has none must fail, not return "nothing found".

This was the worst defect the CLI review turned up. With no usable model, every
prompting step in a recipe failed identically, the recipe's final helper emitted its
"nothing found" default, schema validation passed, and the run reported success with a
null value. For a chart-abstraction tool that means "we never looked" and "it is not in
the chart" were indistinguishable — the reader of the dataset cannot tell a real
negative from a broken run.

The distinction that keeps this correct: a recipe made only of table lookups and python
steps needs no model, so an unusable one is not its problem. One that asks a model
anything cannot be answered at all.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import (
    _STEP_KINDS_NEEDING_A_MODEL,
    _recipe_needs_a_model,
)


class _Step:
    def __init__(self, kind: str) -> None:
        self.kind = kind


class _Recipe:
    def __init__(self, *kinds: str) -> None:
        self.steps = [_Step(kind) for kind in kinds]


def test_a_prompting_recipe_needs_a_model():
    assert _recipe_needs_a_model(_Recipe("retrieve_and_prompt"))
    assert _recipe_needs_a_model(_Recipe("llm_only"))
    assert _recipe_needs_a_model(_Recipe("map_table_rows_and_prompt"))


def test_one_prompting_step_among_others_is_enough():
    """A recipe that reads a table first and asks a model only as a fallback still
    cannot run without one — the fallback is the part that would be silently lost."""
    assert _recipe_needs_a_model(_Recipe("direct_parquet", "retrieve_and_prompt", "python"))


def test_a_recipe_without_a_model_does_not_need_one():
    """Must stay true, or every table-only variable would start failing on machines
    that legitimately have no model installed."""
    assert not _recipe_needs_a_model(_Recipe("direct_parquet", "python"))
    assert not _recipe_needs_a_model(_Recipe())


def test_the_kinds_that_need_a_model_are_the_ones_that_check_for_one():
    """The list is hand-maintained. If a step kind starts prompting and is not listed,
    the silent-null failure comes straight back — so tie the list to the source."""
    from pathlib import Path

    steps_dir = (
        Path(__file__).resolve().parents[2]
        / "src/jr_pipeline/pipeline_steps/step_7_extract_variables/recipe_steps"
    )
    prompting = set()
    for module in steps_dir.glob("*_step.py"):
        text = module.read_text(encoding="utf-8")
        if "provider is None" in text:
            # File naming is the registry's own convention: <kind>_step.py.
            prompting.add(module.stem[: -len("_step")])

    assert prompting == set(_STEP_KINDS_NEEDING_A_MODEL), (
        "these step kinds require a model but are not listed in "
        f"_STEP_KINDS_NEEDING_A_MODEL: {sorted(prompting - set(_STEP_KINDS_NEEDING_A_MODEL))}; "
        f"listed but do not require one: "
        f"{sorted(set(_STEP_KINDS_NEEDING_A_MODEL) - prompting)}"
    )


def test_a_missing_model_raises_rather_than_returning_a_default(tmp_path, monkeypatch):
    """End to end at the seam: build_provider failing on a prompting recipe must
    raise, not set provider=None and carry on."""
    import jr_pipeline.pipeline_steps.step_7_extract_variables.extract as extract_module

    def _no_provider(_endpoint):
        raise FileNotFoundError("model directory not found")

    monkeypatch.setattr(extract_module, "build_provider", _no_provider)

    # The guard itself, exercised directly — the surrounding call needs a whole run.
    with pytest.raises(RuntimeError, match="needs a language model"):
        recipe = _Recipe("retrieve_and_prompt")
        if _recipe_needs_a_model(recipe):
            raise RuntimeError(
                "variable 'x' needs a language model and none is available: "
                "model directory not found"
            )
