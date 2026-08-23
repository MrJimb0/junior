"""A variable name must live under exactly one collection, so name->recipe
resolution is unambiguous (and the deterministic tie-break never has to fire)."""
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_execution_order import (
    _variable_dir,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "var_extraction_recipes"


def test_no_variable_in_two_locations():
    # A variable name must resolve to exactly one directory, at whatever depth the
    # collection/sub-collection tree puts it. Key by the variable dir's relative path.
    # Drafts count: an unfinished recipe still participates in name resolution, so a
    # draft shadowing a finished variable elsewhere is a real ambiguity, not WIP noise.
    seen: dict[str, set[str]] = {}
    for recipe in RECIPES.rglob("*_recipe.yaml"):
        variable_dir = recipe.parent.parent          # <...>/<variable>
        variable = variable_dir.name
        seen.setdefault(variable, set()).add(str(variable_dir.relative_to(RECIPES)))
    dups = {v: sorted(paths) for v, paths in seen.items() if len(paths) > 1}
    assert not dups, f"variables resolvable from multiple locations (ambiguous): {dups}"


def test_resolution_is_deterministic_and_found():
    p1 = _variable_dir(RECIPES, "date_of_diagnosis")
    p2 = _variable_dir(RECIPES, "date_of_diagnosis")
    assert p1 is not None and p1 == p2
    assert p1.name == "date_of_diagnosis" and "oncology" in p1.parts  # under oncology, any depth
