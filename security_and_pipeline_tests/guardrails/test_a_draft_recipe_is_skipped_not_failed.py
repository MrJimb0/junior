"""A draft recipe is skipped by the guardrails and refused by the loader.

Those are two different jobs and it matters which does which. If the guardrails failed
on drafts, somebody else's half-written variable would turn the whole suite red for as
long as it took them to finish, and the lesson people would take is to scaffold outside
the repo -- which is where the wiring mistakes `new-variable` exists to prevent come
back. If the loader let drafts through, a recipe still carrying another variable's
prompts would produce confident, well-formed, provenanced answers to the wrong question.

So: skipped where the rules are about content nobody has written yet, refused where it
would actually run.
"""
from __future__ import annotations

import pytest
import yaml

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from security_and_pipeline_tests.guardrails.recipe_files import (
    RECIPES_DIR,
    finished_recipe_files,
    is_a_draft,
)


def test_a_draft_is_left_out_of_the_guardrail_set(tmp_path):
    draft = tmp_path / "x_v1_recipe.yaml"
    draft.write_text("name: x\nneeds_editing: true\n", encoding="utf-8")
    finished = tmp_path / "y_v1_recipe.yaml"
    finished.write_text("name: y\n", encoding="utf-8")

    assert is_a_draft(draft)
    assert not is_a_draft(finished)


def test_an_unreadable_recipe_is_not_quietly_treated_as_a_draft(tmp_path):
    """Skipping is for recipes that say they are unfinished. A file that cannot be
    parsed is a real failure and must reach the guardrail that reports it."""
    broken = tmp_path / "z_v1_recipe.yaml"
    broken.write_text("name: [unclosed\n", encoding="utf-8")

    assert not is_a_draft(broken)


def test_nothing_committed_here_is_a_draft():
    """Drafts are a working-tree state. One reaching a run's sealed bundle would be a
    recipe the bundle attests to and the loader refuses."""
    drafts = [
        path.relative_to(RECIPES_DIR)
        for path in RECIPES_DIR.rglob("*_recipe.yaml")
        if "__pycache__" not in path.parts and is_a_draft(path)
    ]
    assert not drafts, (
        f"unfinished recipes are in the tree: {drafts}. Finish them (delete the "
        "`needs_editing: true` line) or remove them."
    )


def test_the_guardrails_have_recipes_to_run_over():
    """Seven guardrail files parametrize over finished_recipe_files(). An empty list
    is not a green suite: it is those files asserting nothing, and every rule about
    output keys, DAG targets and stop_if fields silently stops being checked. A moved
    recipes directory or a tree-wide `needs_editing` would do it, so the count has a
    floor rather than being trusted to be non-empty."""
    assert len(finished_recipe_files()) >= 6


def test_every_finished_recipe_actually_loads():
    """The set the guardrails run over is exactly the set a run could use."""
    for path in finished_recipe_files():
        load_recipe(path)


def test_the_loader_refuses_a_draft(tmp_path):
    template = next(iter(finished_recipe_files()))
    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    raw["needs_editing"] = True
    draft = template.parent / "draft_check_recipe.yaml"
    draft.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="not finished"):
            load_recipe(draft)
    finally:
        draft.unlink()
