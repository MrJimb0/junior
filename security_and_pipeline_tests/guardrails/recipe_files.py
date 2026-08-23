"""Which recipes the guardrails hold to the full contract.

A recipe scaffolded by `junior new-variable` carries `needs_editing: true` until its
author has rewritten the prompts and schema it was copied from. Until then it is a
draft: its prompts describe another variable and its schema names that variable's
fields, so every rule about declared output keys, DAG targets and stop_if fields is
being asked of something that does not claim to be finished.

Drafts are skipped here rather than failed. Authoring a variable takes longer than one
sitting, and a shared checkout where somebody else's half-written recipe turns the whole
suite red teaches people to scaffold outside the repo, which is where the wiring
mistakes come back. The safety property does not live here anyway: load_recipe refuses
a draft outright, so an unfinished recipe cannot run, cannot be sealed into a bundle,
and cannot produce a value. It just is not held to rules about content it has not
written yet.
"""
from __future__ import annotations

from pathlib import Path

import yaml

RECIPES_DIR = Path(__file__).resolve().parents[2] / "var_extraction_recipes"


def is_a_draft(recipe_file: Path) -> bool:
    """True while a scaffolded recipe still describes the variable it was copied from."""
    try:
        raw = yaml.safe_load(recipe_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False  # unreadable is a real failure; let the guardrail report it
    return bool(isinstance(raw, dict) and raw.get("needs_editing"))


def finished_recipe_files() -> list[Path]:
    """Every recipe claiming to be finished, which is every recipe a run could use."""
    return sorted(
        path for path in RECIPES_DIR.rglob("*_recipe.yaml")
        if "__pycache__" not in path.parts and not is_a_draft(path)
    )
