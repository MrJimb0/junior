"""Work out what order to run recipes in when some recipes depend on others.

A recipe can declare that it depends on other recipes (e.g. a staging recipe needs
the tumor and node findings first). This file loads the requested recipes, follows
those ``depends_on`` links to pull in every recipe they need (and the recipes those
need, and so on), and then sorts them so each recipe runs only after the ones it
relies on. If the dependencies form a loop (A needs B which needs A), that is an
error and it refuses to run rather than guess an order.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    RecipeSpec,
    load_recipe,
)


class RecipeDAGError(RuntimeError):
    """Raised when recipe dependencies form a loop (or otherwise cannot be ordered)."""


@dataclass(frozen=True)
class RecipePlan:
    """The recipes in a safe run order, plus the names of any depended-on recipes whose yaml file was not found on disk."""

    recipes: list[RecipeSpec]
    missing: list[str]


def _variable_dir(recipes_root: Path, name: str) -> Path | None:
    """Locate <name>'s folder under recipes_root at ANY depth — flat (``<name>/``),
    one collection level (``<collection>/<name>/``), or nested sub-collections
    (``<collection>/<subcollection>/<name>/``). A variable dir is one named <name>
    that holds version (``v<N>``) subdirs, so an incidental same-named dir (e.g. a
    ``prompts/`` child) is never mistaken for it.

    Sort so resolution is deterministic if a variable somehow exists in two
    places (lexicographically-first wins); a guardrail test forbids that duplicate."""
    recipes_root = Path(recipes_root)

    def _is_variable_dir(d: Path) -> bool:
        return d.is_dir() and d.name == name and any(
            c.is_dir() and c.name[:1] == "v" and c.name[1:].isdigit() for c in d.iterdir()
        )

    flat = recipes_root / name
    if _is_variable_dir(flat):
        return flat
    matches = sorted(d for d in recipes_root.glob(f"**/{name}") if _is_variable_dir(d))
    return matches[0] if matches else None


def _recipe_path(recipes_root: Path, name: str) -> Path:
    """Highest-version recipe yaml for <name>, found whether the recipe sits flat
    (``<recipes_root>/<name>/v<N>/``) or nested under a collection
    (``<recipes_root>/<collection>/<name>/v<N>/``). Nonexistent path if not found
    (caller checks is_file())."""
    variable_dir = _variable_dir(recipes_root, name)
    if variable_dir is None:
        return Path(recipes_root) / name / "v1" / f"{name}_v1_recipe.yaml"
    versions = sorted(
        (p for p in variable_dir.iterdir() if p.is_dir() and p.name.startswith("v")),
        key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1,
    )
    if not versions:
        return variable_dir / "v1" / f"{name}_v1_recipe.yaml"
    latest = versions[-1]
    return latest / f"{name}_{latest.name}_recipe.yaml"


def _collect_transitive(
    recipes_root: Path, requested: Iterable[str]
) -> tuple[dict[str, RecipeSpec], list[str]]:
    loaded: dict[str, RecipeSpec] = {}
    missing: list[str] = []
    queue = list(requested)
    while queue:
        name = queue.pop()
        if name in loaded:
            continue
        path = _recipe_path(recipes_root, name)
        if not path.is_file():
            if name not in missing:
                missing.append(name)
            continue
        spec = load_recipe(path)
        loaded[spec.name] = spec
        for dep in spec.depends_on:
            if dep not in loaded:
                queue.append(dep)
    return loaded, missing


def _topo_sort(specs: dict[str, RecipeSpec]) -> list[str]:
    """Return recipe names in dependency order (each after the recipes it depends on).

    Uses the standard Kahn ordering: repeatedly take any recipe whose dependencies
    are all already placed. Recipes that become available at the same time are taken
    in alphabetical order, so the result is always the same for the same inputs."""
    indeg: dict[str, int] = {n: 0 for n in specs}
    adj: dict[str, list[str]] = {n: [] for n in specs}
    for name, spec in specs.items():
        for dep in spec.depends_on:
            if dep in specs:
                adj[dep].append(name)
                indeg[name] += 1

    ready = sorted([n for n, d in indeg.items() if d == 0])
    order: list[str] = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for nxt in sorted(adj[n]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    if len(order) != len(specs):
        leftover = sorted(n for n, d in indeg.items() if d > 0)
        raise RecipeDAGError(f"Cyclic recipe dependencies among: {leftover}")
    return order


def plan(*, recipes_root: Path, requested: Iterable[str]) -> RecipePlan:
    """Build the run plan: the requested recipes plus every recipe they depend on (directly or indirectly), put in an order where each recipe runs after the ones it needs. Names that could not be found on disk are returned in `missing` instead of raising an error (the runner fills those in as null upstream values)."""
    specs, missing = _collect_transitive(Path(recipes_root), requested)
    order = _topo_sort(specs)
    return RecipePlan(
        recipes=[specs[n] for n in order],
        missing=missing,
    )
