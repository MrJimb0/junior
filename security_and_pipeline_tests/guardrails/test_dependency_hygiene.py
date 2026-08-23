"""Guardrail: the PHI environment installs only what pyproject declares.

`pyproject.toml` is the source of truth for the locked PHI environment. This
guard:

  * confirms the remote LLM providers' hard dependency ``requests`` stays
    declared (a clean install dies without it), and
  * parses package *names* out of ``uv.lock`` to ensure deliberately-rejected
    libraries never re-enter the locked environment. ``litellm`` and its
    transitive ``openai``/``aiohttp``/``tiktoken`` are excluded, as are
    ``sentence-transformers``/``diskcache``/``bm25s``/``duckdb``/``json-repair``
    (see the skip ADR).

It deliberately does NOT assert ``uv.lock``'s internal metadata shape — only the
set of package names — so it stays robust to lockfile-schema churn.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"

# Must never land in the locked PHI environment.
FORBIDDEN_PACKAGES = {
    "litellm",
    "openai",
    "aiohttp",
    "tiktoken",
    "sentence-transformers",
    "diskcache",
    "bm25s",
    "duckdb",
    "json-repair",
}


def _declared_names() -> set[str]:
    """Normalised package names declared in pyproject (deps + all extras)."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        # strip version markers/extras: "rank-bm25>=0.2.2" -> "rank-bm25"
        name = spec.split(";")[0].strip()
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            name = name.split(sep)[0]
        names.add(name.strip().lower())
    return names


def test_requests_is_declared_direct_dependency():
    assert "requests" in _declared_names(), (
        "remote LLM providers import requests; it must stay a declared dependency"
    )


def test_pyproject_declares_no_forbidden_packages():
    leaked = sorted(_declared_names() & FORBIDDEN_PACKAGES)
    assert not leaked, f"deliberately-rejected packages declared in pyproject: {leaked}"


def test_uv_lock_has_no_forbidden_packages():
    if not UV_LOCK.exists():
        return  # lock is optional; pyproject remains the source of truth
    locked = tomllib.loads(UV_LOCK.read_text())
    names = {pkg["name"].lower() for pkg in locked.get("package", [])}
    leaked = sorted(names & FORBIDDEN_PACKAGES)
    assert not leaked, f"deliberately-rejected packages re-entered uv.lock: {leaked}"
