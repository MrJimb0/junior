"""Machine-checkable conformance to the clinician-readable recipe template
(var_extraction_recipes/RECIPE_TEMPLATE.md), section 3 STEPS rule.

The full ``{{ OUTPUT_SCHEMA }}`` may be injected ONLY by a recipe's authoritative final
merge -- a final LLM pass that assembles the whole output. A partial pass (any earlier
pass, or any pass at all when the recipe's last step is a deterministic ``python``
finalize) must declare only the keys it owns and must NOT inject the full schema, or it
gets shown the entire output contract and invents fields it does not own.

This guard asserts the strict rule directly: NO recipe injects the full schema into a
partial pass.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files

_SCHEMA_RE = re.compile(r"\{\{\s*OUTPUT_SCHEMA\s*\}\}")
_PROMPT_KINDS = {
    "retrieve_and_prompt",
    "map_table_rows_and_prompt",
    "llm_only",
}


def _recipe_yamls() -> list[Path]:
    # Finished recipes only: a draft's prompts still describe its template, so a
    # content rule about what they inject is not yet its to pass.
    return finished_recipe_files()


def _injects_schema_into_partial_pass(recipe_yaml: Path) -> bool:
    spec = yaml.safe_load(recipe_yaml.read_text(encoding="utf-8"))
    steps = spec.get("steps") or []
    if not steps:
        return False
    last_is_prompt = steps[-1].get("kind") in _PROMPT_KINDS
    rdir = recipe_yaml.parent
    for i, step in enumerate(steps):
        prompt = step.get("prompt")
        if not prompt:
            continue
        ptext = (rdir / prompt).read_text(encoding="utf-8")
        if not _SCHEMA_RE.search(ptext):
            continue
        is_final_authoritative = last_is_prompt and i == len(steps) - 1
        if not is_final_authoritative:
            return True  # a partial pass injected the full output schema
    return False


def _live_violations() -> set[str]:
    out: set[str] = set()
    for ry in _recipe_yamls():
        if _injects_schema_into_partial_pass(ry):
            out.add(yaml.safe_load(ry.read_text(encoding="utf-8"))["name"])
    return out


def test_no_recipe_injects_full_schema_into_a_partial_pass():
    violations = _live_violations()
    assert violations == set(), (
        "recipe(s) inject the full {{ OUTPUT_SCHEMA }} into a partial pass -- a partial "
        "pass must declare only the keys it owns (RECIPE_TEMPLATE.md section 3): "
        f"{sorted(violations)}"
    )
