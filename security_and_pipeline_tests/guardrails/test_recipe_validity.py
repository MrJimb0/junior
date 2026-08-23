"""Guardrail: every recipe is structurally valid before it can ship.

Catches the failure classes that otherwise bite only at runtime, without importing
helper modules or needing torch:
  * undefined prompt context names (the `stages` vs `steps` class),
  * prompt references to steps that don't exist or run later,
  * prompt references to vars not declared in depends_on,
  * missing prompt / helper files, undefined helper functions,
  * invalid output schemas,
  * duplicate step ids, recipe name not matching its folder.
"""
import ast as pyast
import json
from pathlib import Path

import jinja2
import pytest
from jinja2 import meta as jinja_meta

from jr_pipeline.pipeline_steps.step_7_extract_variables.prompt_template_rendering import (
    load_prompt,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files

REPO = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO / "var_extraction_recipes"

# Names the step handlers put into the prompt render context.
_ALLOWED_CONTEXT = {"patient_id", "evidence_json", "evidence_text", "OUTPUT_SCHEMA", "vars", "steps"}


def _recipe_paths():
    # Depth-agnostic: recipes may sit flat (<variable>/v*/) or nested under a
    # collection (<collection>/<variable>/v*/).
    return finished_recipe_files() if RECIPES_DIR.is_dir() else []


def _attr_refs(source: str, base: str) -> set[str]:
    """The first-level attrs in <base>.<attr> references within a jinja source."""
    tree = jinja2.Environment().parse(source)
    return {
        node.attr
        for node in tree.find_all(jinja2.nodes.Getattr)
        if isinstance(node.node, jinja2.nodes.Name) and node.node.name == base
    }


@pytest.mark.parametrize("recipe_path", _recipe_paths(), ids=lambda p: p.parent.parent.name)
def test_recipe_is_valid(recipe_path: Path):
    spec = load_recipe(recipe_path)

    folder_var = recipe_path.parent.parent.name
    assert spec.name == folder_var, f"recipe name {spec.name!r} does not match folder {folder_var!r}"

    # A recipe nested under a collection must declare a `collection:` field matching
    # the TOP-level collection dir, so the on-disk grouping and the recipe's
    # self-description never drift. The path may be <collection>/<variable>/<version>/
    # or have sub-collection levels (<collection>/<subcollection>/<variable>/<version>/);
    # the top-level collection is always rel_parts[0] regardless of depth.
    rel_parts = recipe_path.relative_to(RECIPES_DIR).parts
    if len(rel_parts) >= 4:
        collection_dir = rel_parts[0]
        declared = spec.front_matter.get("collection")
        assert declared == collection_dir, (
            f"{folder_var}: collection field {declared!r} does not match top collection {collection_dir!r}"
        )

    step_ids = [s.id for s in spec.steps]
    assert len(step_ids) == len(set(step_ids)), f"{folder_var}: duplicate step ids {step_ids}"

    json.loads(Path(spec.output_schema_path).read_text(encoding="utf-8"))  # valid JSON schema doc

    depends = set(spec.depends_on)
    for i, step in enumerate(spec.steps):
        prior = set(step_ids[:i])

        if step.kind == "python":
            assert step.module and "." in step.module, f"{folder_var}/{step.id}: bad module ref"
            file_stem, func = step.module.rsplit(".", 1)
            helper = recipe_path.parent / f"{file_stem}.py"
            assert helper.is_file(), f"{folder_var}/{step.id}: missing helper {helper.name}"
            tree = pyast.parse(helper.read_text(encoding="utf-8"))
            funcs = {n.name for n in pyast.walk(tree)
                     if isinstance(n, (pyast.FunctionDef, pyast.AsyncFunctionDef))}
            assert func in funcs, f"{folder_var}/{step.id}: {func}() not defined in {helper.name}"

        if step.prompt:
            assert Path(step.prompt).is_file(), f"{folder_var}/{step.id}: missing prompt {step.prompt}"
            t = load_prompt(step.prompt)
            source = t.system + "\n" + t.user
            undeclared = jinja_meta.find_undeclared_variables(jinja2.Environment().parse(source))
            bad = undeclared - _ALLOWED_CONTEXT
            assert not bad, f"{folder_var}/{step.id}: prompt uses unknown context names {sorted(bad)}"
            for sid in _attr_refs(source, "steps"):
                assert sid in prior, f"{folder_var}/{step.id}: prompt references steps.{sid} (not a prior step)"
            for vname in _attr_refs(source, "vars"):
                assert vname in depends, f"{folder_var}/{step.id}: prompt references vars.{vname} (not in depends_on)"
