"""Import-graph checker — the dead-module gate.

Builds a PRODUCTION reachability graph over ``src/jr_pipeline`` rooted at the real
entry points, and flags any in-package module that is unreachable from production.
Such a module is "production-dead": reached only by tests (or by nothing). That is a
failure unless the module is explicitly allowlisted with a one-line reason.

Production roots:
  * the package ``jr_pipeline`` and ``jr_pipeline.__main__``; and
  * every ``jr_pipeline.*`` module imported by an out-of-package entry point —
    ``apps_and_interfaces/`` (home of the CLI — the ``[project.scripts]`` entry —
    the interactive prompt, and the review app), ``deployment/``,
    ``var_extraction_recipes/`` (recipe hooks loaded dynamically by path), and
    ``quickstart.py``.

The CLI ``STAGES`` dispatch and the recipe-step / provider registries are ordinary
in-package imports, so the static graph already covers them; the only dynamic
loading (``spec_from_file_location``) targets external recipe files, never an
in-package module, so nothing in ``src`` is reachable only dynamically.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
PKG = "jr_pipeline"

ENTRYPOINT_DIRS = ["apps_and_interfaces", "deployment", "var_extraction_recipes"]
ENTRYPOINT_FILES = ["quickstart.py"]
TEST_DIR = REPO_ROOT / "security_and_pipeline_tests"

# Production-dead modules that are legitimately so. Every entry needs a one-line
# reason. These are NOT dead-to-delete: they are real features whose production entry
# points are not wired yet. Each MUST be removed from this allowlist once its entry
# point is wired (the stale-entry test enforces that).
ALLOWLIST: dict[str, str] = {
}


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def all_src_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if not _module_name(path).startswith(PKG):
            continue  # ignore stray top-level scripts under src (e.g. egg-info)
        modules[_module_name(path)] = path
    return modules


def _jr_imports(path: Path) -> set[str]:
    """jr_pipeline.* modules imported by ``path`` (absolute + resolved relative)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    is_pkg = path.name == "__init__.py"
    module = _module_name(path) if path.is_relative_to(SRC) else ""
    current_pkg = module if is_pkg else (module.rsplit(".", 1)[0] if "." in module else module)

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PKG or alias.name.startswith(PKG + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — resolve against the current package
                base = current_pkg.split(".") if current_pkg else []
                base = base[: len(base) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target == PKG or target.startswith(PKG + "."):
                found.add(target)
                for alias in node.names:  # `from pkg.x import y` — y may be a submodule
                    found.add(f"{target}.{alias.name}")
    return found


def _build_graph(modules: dict[str, Path]) -> dict[str, set[str]]:
    keys = set(modules)
    return {
        name: {imp for imp in _jr_imports(path) if imp in keys}
        for name, path in modules.items()
    }


def _production_roots(modules: dict[str, Path]) -> set[str]:
    roots = {PKG, f"{PKG}.__main__"}
    entry_paths: list[Path] = []
    for directory in ENTRYPOINT_DIRS:
        entry_paths.extend((REPO_ROOT / directory).rglob("*.py"))
    entry_paths.extend(REPO_ROOT / f for f in ENTRYPOINT_FILES)
    for path in entry_paths:
        if path.is_file() and "__pycache__" not in path.parts:
            roots |= _jr_imports(path)
    return {r for r in roots if r in modules}


def production_reachable(modules: dict[str, Path] | None = None) -> set[str]:
    modules = modules or all_src_modules()
    graph = _build_graph(modules)
    seen: set[str] = set()
    stack = list(_production_roots(modules))
    while stack:
        node = stack.pop()
        if node in seen or node not in graph:
            continue
        seen.add(node)
        stack.extend(graph[node])
    # importing a submodule runs its ancestor packages' __init__ files
    for name in list(seen):
        parts = name.split(".")
        for i in range(1, len(parts)):
            ancestor = ".".join(parts[:i])
            if ancestor in modules:
                seen.add(ancestor)
    return seen


def _test_roots(modules: dict[str, Path]) -> set[str]:
    roots: set[str] = set()
    for path in TEST_DIR.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        roots |= _jr_imports(path)
    return {r for r in roots if r in modules}


def reachable_from_tests(modules: dict[str, Path] | None = None) -> set[str]:
    """Modules reachable from the TEST graph (rooted at the test files)."""
    modules = modules or all_src_modules()
    graph = _build_graph(modules)
    seen: set[str] = set()
    stack = list(_test_roots(modules))
    while stack:
        node = stack.pop()
        if node in seen or node not in graph:
            continue
        seen.add(node)
        stack.extend(graph[node])
    for name in list(seen):
        parts = name.split(".")
        for i in range(1, len(parts)):
            ancestor = ".".join(parts[:i])
            if ancestor in modules:
                seen.add(ancestor)
    return seen


def classify_dead(module: str) -> str:
    """Why a production-dead module is dead: 'only-test-imported' or 'zero-importer'."""
    return "only-test-imported" if module in reachable_from_tests() else "zero-importer"


def production_dead_modules() -> list[str]:
    """In-package modules unreachable from production roots and not allowlisted."""
    modules = all_src_modules()
    reachable = production_reachable(modules)
    return sorted(m for m in modules if m not in reachable and m not in ALLOWLIST)


def stale_allowlist_entries() -> list[str]:
    """Allowlisted modules that are actually production-reachable (should be removed)."""
    reachable = production_reachable()
    return sorted(m for m in ALLOWLIST if m in reachable)
