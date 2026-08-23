"""A step's stop_if must only gate on keys the recipe declares.

stop_if is evaluated against ``data`` = the step's output. A stop_if that reads a
key the recipe's output schema never declares is a typo that silently evaluates
False forever. This guard parses the ``data.get('K')`` / ``data['K']`` keys out of
every stop_if and asserts each is a declared output-schema property of its recipe.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files

REPO = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO / "var_extraction_recipes"

_DATA_GET = re.compile(r"data\.get\(\s*['\"]([^'\"]+)['\"]")
_DATA_IDX = re.compile(r"data\[\s*['\"]([^'\"]+)['\"]\s*\]")


def _stop_if_data_keys(expr: str) -> set[str]:
    return set(_DATA_GET.findall(expr)) | set(_DATA_IDX.findall(expr))


def _recipe_paths() -> list[Path]:
    return finished_recipe_files()


def test_every_stop_if_key_is_a_declared_output_field() -> None:
    problems: list[str] = []
    for path in _recipe_paths():
        spec = load_recipe(path)
        schema = json.loads(spec.output_schema_path.read_text(encoding="utf-8"))
        props = set((schema.get("properties") or {}).keys())
        for step in spec.steps:
            if not step.stop_if:
                continue
            for key in _stop_if_data_keys(step.stop_if):
                if key not in props:
                    problems.append(
                        f"{spec.name}/{step.id}: stop_if gates on data[{key!r}] "
                        f"which is not a declared output field {sorted(props)}"
                    )
    assert not problems, "stop_if keys with no matching output field:\n" + "\n".join(problems)


def test_key_extractor_is_not_vacuous() -> None:
    """Non-tautology guard: the regexes actually pull keys out of the live syntax."""
    expr = "data is not none and data.get('date_of_birth') is not none"
    assert _stop_if_data_keys(expr) == {"date_of_birth"}
    assert _stop_if_data_keys("data['x'] == 'y'") == {"x"}
