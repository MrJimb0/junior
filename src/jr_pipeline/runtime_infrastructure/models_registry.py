"""Model registry: resolve a registry ``model_id`` to its entry + absolute path.

``models/models_registry.yaml`` is the single source of truth for model references —
recipes (and the reranker) name models by registry key, not by loose path. The file is
located via ``JR_MODELS_REGISTRY`` (default ``models/models_registry.yaml`` relative to
the process cwd); each entry's repo-root-relative ``path`` is resolved to absolute
against the registry file's location (``<root>/models/models_registry.yaml`` →
``<root>``).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_REGISTRY = "models/models_registry.yaml"


def models_registry_path() -> Path:
    """The registry file path (``JR_MODELS_REGISTRY`` override, else the repo default)."""
    return Path(os.environ.get("JR_MODELS_REGISTRY") or _DEFAULT_REGISTRY)


@lru_cache(maxsize=8)
def _load_models(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"models registry not found at {path}; set JR_MODELS_REGISTRY to override."
        )
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return doc.get("models") or {}


def resolve_model(model_id: str, *, registry_path: str | Path | None = None) -> dict:
    """Return the registry entry for ``model_id``. When the entry declares a ``path``
    (repo-root-relative), the returned dict carries ``resolved_path`` (absolute). Raises
    ``KeyError`` if the id is not registered."""
    reg = Path(registry_path) if registry_path else models_registry_path()
    models = _load_models(str(reg))
    entry = models.get(model_id)
    if entry is None:
        raise KeyError(
            f"model_id {model_id!r} is not registered in {reg}; known ids: {sorted(models)}"
        )
    out = dict(entry)
    rel = entry.get("path")
    if rel:
        out["resolved_path"] = reg.resolve().parent.parent / rel
    return out
