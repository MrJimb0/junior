"""Run a small Python helper that ships alongside the recipe.

Some chart logic can't be expressed as a prompt to a model — for example
parsing a date string into a real date, removing duplicate treatment lines, or
checking a value against a fixed list of allowed terms. Rather than build that
logic into the pipeline itself, the recipe points at its own helper function
with ``module: file.func``, and this step loads and calls it on demand from the
recipe's own folder.

The helper receives a StepContext (everything it needs about the patient and the
run) and returns a StepResult (its output plus a receipt). We record a
fingerprint of the helper file in the receipt, so the audit trail covers the
exact Python code that ran, not just the recipe's YAML configuration.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_type_lookup import (
    register_step,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)


@register_step("python")
class PythonStepHandler:
    """Load and call the Python helper function that sits next to the recipe's recipe.yaml."""

    kind = "python"

    def execute(self, ctx: StepContext) -> StepResult:
        ref = ctx.step.module
        if not ref or "." not in ref:
            raise ValueError(
                f"step {ctx.step.id}: python step requires 'module: <file>.<func>'"
            )
        file_stem, func_name = ref.rsplit(".", 1)
        helper_file = (ctx.recipe.path.parent / f"{file_stem}.py").resolve()
        if not helper_file.is_file():
            raise FileNotFoundError(f"Python helper file not found: {helper_file}")

        module = _load_module(helper_file)
        if not hasattr(module, func_name):
            raise AttributeError(f"Python helper {file_stem}.{func_name} not defined in {helper_file}")
        func = getattr(module, func_name)

        result = func(ctx)
        if not isinstance(result, StepResult):
            raise TypeError(
                f"Python helper {ref} must return a StepResult; got {type(result).__name__}"
            )
        result.receipt_payload.setdefault("python_helper", {})
        result.receipt_payload["python_helper"].update({
            "file": helper_file.name,
            "function": func_name,
            "file_hash": hash_file(helper_file),
        })
        return result


def _load_module(path: Path):
    """Import a helper .py file by its path, reusing it if already loaded.

    The module is registered under a prefixed, unique name so a recipe helper
    can't collide with or shadow a real installed package.
    """
    mod_name = f"jr_pipeline_helpers__{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
