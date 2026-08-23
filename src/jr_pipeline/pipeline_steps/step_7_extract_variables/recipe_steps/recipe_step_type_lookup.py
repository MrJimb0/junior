"""Lookup table from a step's "kind" name to the code that runs it.

A recipe lists steps, each tagged with a kind (e.g. "llm_only"). This module
keeps a simple dictionary so the recipe runner can find the right handler for a
kind by name. New step types register themselves here via the @register_step
decorator. (ADR 0012 is the design note that chose this dictionary approach.)
"""
from __future__ import annotations

from collections.abc import Callable

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepHandler,
)

_HANDLERS: dict[str, Callable[[], StepHandler]] = {}


def register_step(kind: str) -> Callable[[type], type]:
    """Decorator that registers a handler class under a step-kind name."""

    def _dec(cls: type) -> type:
        # Store a builder function (not the class itself) so build_step() makes a
        # fresh handler instance on every call rather than reusing one shared object.
        _HANDLERS[kind] = lambda: cls()   # type: ignore[abstract]
        return cls

    return _dec


def build_step(kind: str) -> StepHandler:
    if kind not in _HANDLERS:
        raise KeyError(f"Unknown step kind: {kind!r}. Available: {sorted(_HANDLERS)}")
    return _HANDLERS[kind]()
