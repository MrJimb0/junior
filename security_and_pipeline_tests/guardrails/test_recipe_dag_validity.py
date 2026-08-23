"""Every recipe `depends_on` edge must resolve to a real recipe on disk.

A dangling depends_on target (one naming a recipe that does not exist) is silently
dropped from the topological DAG (`recipe_execution_order._topo_sort` only keeps edges
whose target is loaded), so the dependent recipe's upstream var is `None` forever and
nothing complains. This guard catches that class of typo/stale-edge at authoring time.
"""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_execution_order import plan
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files

REPO = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO / "var_extraction_recipes"


def _all_recipe_names() -> list[str]:
    return sorted({load_recipe(p).name for p in finished_recipe_files()})


def test_every_depends_on_target_exists() -> None:
    """Requesting every recipe walks all depends_on edges transitively; any edge
    whose target has no recipe file lands in `plan.missing`. It must be empty."""
    result = plan(recipes_root=RECIPES_DIR, requested=_all_recipe_names())
    assert result.missing == [], (
        f"depends_on edges with no recipe on disk: {result.missing} — either the "
        "recipe was renamed/never built (drop the edge) or it is a typo."
    )


def test_no_cycles_in_recipe_dag() -> None:
    """The real recipe set is acyclic; plan() raises RecipeDAGError on a cycle."""
    plan(recipes_root=RECIPES_DIR, requested=_all_recipe_names())  # must not raise


def test_dangling_target_is_detected() -> None:
    """Non-tautology guard: a name with no recipe is reported as missing, proving
    test_every_depends_on_target_exists would actually fire on a dangling edge."""
    result = plan(recipes_root=RECIPES_DIR, requested=["nonexistent_recipe_xyz"])
    assert "nonexistent_recipe_xyz" in result.missing
