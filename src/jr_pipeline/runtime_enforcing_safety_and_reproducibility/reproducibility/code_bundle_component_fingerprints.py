"""Scoped sub-hash computation for the code bundle.

Each scope is hashed canonically (see ``hashes.schema.json``) so two runs with
identical substance produce identical sub-hashes regardless of formatting noise.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_directory,
    hash_file,
    hash_json,
)

# Excluded from code_lock_hash: these files embed the hash, so including them
# would make the hash reference itself.
_BUNDLE_METADATA_FILES = frozenset({"hashes.json", "code.lock.json"})


def code_lock_hash(code_dir: Path) -> str:
    """Hash the substantive contents of a sealed code/ directory."""
    return hash_directory(code_dir, exclude=_BUNDLE_METADATA_FILES)


def config_subset_hash(cfg: dict, keys: list[str]) -> str | None:
    """Hash a subset of a config tree; None if no keys present."""
    subset = {k: cfg[k] for k in keys if k in cfg}
    if not subset:
        return None
    return hash_json(subset)


# Real config keys per scope. These MUST match the resolved config so a scoped
# sub-hash is non-None when the stage is configured — an empty subset would be an
# inert seal-drift gate.
_RETRIEVAL_CONFIG_KEYS = ["encoder", "index", "chunker", "retrieval", "reranking", "retrievers"]
_PROVIDER_CONFIG_KEYS = ["allowed_endpoints", "allowed_models", "model_override", "llm", "providers"]


def provider_config_hash(cfg: dict) -> str | None:
    """Hash the LLM-provider-related config subset (never URLs or keys)."""
    return config_subset_hash(cfg, _PROVIDER_CONFIG_KEYS)


def retrieval_config_hash(cfg: dict) -> str | None:
    """Hash the retrieval/embedding-stage config subset (encoder/index/chunker/...)."""
    return config_subset_hash(cfg, _RETRIEVAL_CONFIG_KEYS)


def assert_scoped_hashes_grounded(cfg: dict) -> None:
    """Fail loud if a stage IS configured but its scoped sub-hash is None — that means the
    subset keys drifted from the real config and the seal-drift gate has gone inert."""
    if any(k in cfg for k in _RETRIEVAL_CONFIG_KEYS) and retrieval_config_hash(cfg) is None:
        raise ValueError(
            "seal: retrieval_config_hash is None though the config defines a retrieval-stage "
            "key — the scoped-hash subset is mis-grounded"
        )
    if any(k in cfg for k in _PROVIDER_CONFIG_KEYS) and provider_config_hash(cfg) is None:
        raise ValueError(
            "seal: provider_config_hash is None though the config defines a provider key — "
            "the scoped-hash subset is mis-grounded"
        )


def recipe_hash(recipe_yaml_path: Path) -> str:
    """Content hash of a single recipe YAML file."""
    return hash_file(recipe_yaml_path)


def schema_hash(schema_json_path: Path) -> str:
    """Canonically hash a schema JSON file so reformatting is a no-op."""
    import json as _json

    with schema_json_path.open("r", encoding="utf-8") as f:
        return hash_json(_json.load(f))


def prompt_hash(prompt_md_path: Path) -> str:
    """Content hash of a single prompt markdown file."""
    return hash_file(prompt_md_path)


def python_helpers_hash(hooks_py_paths: list[Path]) -> str | None:
    """Combined hash of python-helper .py files, sorted by path; None if empty."""
    if not hooks_py_paths:
        return None
    h = hashlib.sha256()
    for p in sorted(hooks_py_paths):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            h.update(f.read())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


def per_recipe_hashes(recipes_root: Path) -> dict[str, dict[str, Any]]:
    """Scoped hashes for every (variable, version) recipe; keyed ``"<variable>/<version>"``.

    Recipes may sit flat (``<variable>/<version>/``) or nested under a collection
    (``<collection>/<variable>/<version>/``), so the canonically-named recipe yamls
    are found at any depth and the (variable, version) is read from the path."""
    if not recipes_root.exists():
        return {}

    out: dict[str, dict[str, Any]] = {}
    for recipe_yaml in sorted(recipes_root.rglob("*_recipe.yaml")):
        version_dir = recipe_yaml.parent
        variable_dir = version_dir.parent
        if not version_dir.name.startswith("v") or variable_dir.name.startswith((".", "_")):
            continue
        prefix = f"{variable_dir.name}_{version_dir.name}"
        if recipe_yaml.name != f"{prefix}_recipe.yaml":
            continue  # non-canonical name -> not a variable recipe
        schema_file = version_dir / f"{prefix}_output_schema.json"
        helper_file = version_dir / f"{prefix}_python_helper.py"
        prompts_dir = version_dir / "prompts"
        prompt_files = (
            sorted(prompts_dir.glob("*.md")) if prompts_dir.is_dir() else []
        )
        prompt_hashes = {p.stem: hash_file(p) for p in prompt_files}

        key = f"{variable_dir.name}/{version_dir.name}"
        out[key] = {
            "recipe_hash": hash_file(recipe_yaml),
            "schema_hash": schema_hash(schema_file) if schema_file.is_file() else None,
            "prompt_hashes": prompt_hashes,
            "python_helpers_hash": (
                python_helpers_hash([helper_file]) if helper_file.is_file() else None
            ),
        }
    return out
