"""Load a module from the recipe tree's ``_shared_validation_rules/`` folder.

Recipe python helpers and ``clinical_invariants.py`` are loaded by file path, not as an
importable package, so they cannot ``import _shared_validation_rules.partial_date``. This
finds ``_shared_validation_rules/<stem>.py`` by walking up from a caller file and loads it
once (cached), giving every helper one shared copy of portable, recipe-side validation
logic (e.g. partial-date parsing) without copying it into each helper.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SHARED_DIR = "_shared_validation_rules"


def load_shared_validation_rule(module_stem: str, anchor: str | Path) -> ModuleType:
    """Load ``<recipe-tree>/_shared_validation_rules/<module_stem>.py``, found by walking
    up from ``anchor`` (pass a caller's ``__file__``). Cached under a namespaced module
    name so all callers share one instance. Raises FileNotFoundError if not found."""
    anchor_path = Path(anchor).resolve()
    for parent in (anchor_path, *anchor_path.parents):
        candidate = parent / _SHARED_DIR / f"{module_stem}.py"
        if candidate.is_file():
            cache_name = f"jr_shared_validation_rules__{module_stem}"
            cached = sys.modules.get(cache_name)
            if cached is not None:
                return cached
            spec = importlib.util.spec_from_file_location(cache_name, candidate)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load shared validation rule: {candidate}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[cache_name] = module  # register before exec (dataclass machinery)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(
        f"{_SHARED_DIR}/{module_stem}.py not found above {anchor_path}"
    )
