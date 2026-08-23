"""Whether a sealed run may continue under the code and settings now in hand.

A run's sealed bundle records the exact code, recipes and settings that produced its
artifacts. Three different doors re-enter a sealed run — the per-stage commands,
`junior run` (and the app's Run button, which spawns it), and `junior seal` — and every
one of them must ask the same questions before doing work under that run id:

  * did any run-invariant setting change (including one being added or removed)?
  * did the content of any recipe the bundle sealed change on disk?
  * does the run's plan name a variable whose recipe the sealed bundle does not hold?

The answers are plain-word sentences, so each door can put them in front of the
operator in its own voice. The one thing no door may do is re-seal an existing
bundle: overwriting code/ and hashes.json rewrites the provenance every receipt
already claims, and (because resume compares against the sealed hashes) silently
turns finished work back into pending work.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (  # noqa: E501
    per_recipe_hashes,
    provider_config_hash,
    retrieval_config_hash,
)

# Cfg keys that MUST match across every entry into a single run. Stage configs vary
# (ingest has no encoder, embed has no retrievers), so the full config_hash can't be
# required to match — these tie the run together. max_chunks_per_prompt and
# max_provenance_retries are here because both change the answers: a cohort extracted
# under two values of either is two datasets silently merged.
RUN_INVARIANT_CFG_KEYS = (
    "run_id",
    "output_root",
    "allowlist_path",
    "recipes_root",
    "max_chunks_per_prompt",
    "max_provenance_retries",
)

# The internal names, said in the operator's terms. Internal keys mean nothing to
# someone who just edited a YAML file and re-ran, which is how people reach these.
IN_PLAIN_WORDS = {
    "retrieval_config_hash": "the embedding or index settings",
    "provider_config_hash": "the language-model settings",
    "cfg.run_id": "the run name",
    "cfg.output_root": "where output is written",
    "cfg.allowlist_path": "the allow-list file",
    "cfg.recipes_root": "where the variable recipes live",
    "cfg.max_chunks_per_prompt": "how many chunks of chart text reach one prompt",
    "cfg.max_provenance_retries": "how many times a step re-asks after a bad citation",
    # recipe/<variable> entries are already plain enough to print.
}


def sealed_config_of(run_root: Path) -> dict:
    """The config this run was sealed with — what actually produced its artifacts.
    Empty when the run has no sealed config (an unsealed or damaged bundle)."""
    sealed_cfg_path = Path(run_root) / "code" / "config_resolved.yaml"
    if not sealed_cfg_path.is_file():
        return {}
    try:
        return yaml.safe_load(sealed_cfg_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def sealed_code_lock_hash(run_root: Path) -> str | None:
    """The bundle's code_lock_hash, read from code.lock.json; None when unreadable."""
    lock_path = Path(run_root) / "code" / "code.lock.json"
    if not lock_path.is_file():
        return None
    try:
        envelope = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (envelope.get("payload") or {}).get("code_lock_hash")


def run_invariant_settings_that_changed(cfg: dict, sealed_cfg: dict) -> list[str]:
    """Run-invariant settings that differ from the sealed config — including a setting
    that was ADDED or REMOVED since sealing. `key in both sides` was the old test, and
    it let `max_chunks_per_prompt: 12` added after sealing walk straight past the gate:
    absent-then-set changes the prompts exactly as much as set-then-changed does."""
    changed = []
    for key in RUN_INVARIANT_CFG_KEYS:
        if key not in cfg and key not in sealed_cfg:
            continue
        if cfg.get(key) != sealed_cfg.get(key):
            changed.append(f"cfg.{key}")
    return changed


def recipes_that_changed_since_sealing(recipes_root: Path | str | None,
                                       sealed_recipes_dir: Path) -> list[str]:
    """Recipes whose live content differs from the copy this run was sealed with.

    Extraction reads recipes LIVE while stamping every receipt with the hashes recorded
    at sealing, so an edited recipe does not merely go unnoticed: the receipt asserts
    the hash of a recipe that did not run. Compared over the WHOLE sealed tree, not
    cfg["recipes"]: the seal snapshots and hashes all of it, so the run already claims
    all of it. A recipe ADDED to the checkout after sealing is not drift — the run never
    used it (running it is refused separately, see
    variables_missing_from_the_sealed_bundle).

    A hashing failure RAISES rather than returning "no drift": one malformed schema
    file used to disable this whole gate silently, which inverted its meaning — the
    least-checkable tree became the most trusted one."""
    if not recipes_root or not Path(sealed_recipes_dir).is_dir():
        return []
    live_root = Path(recipes_root)
    if not live_root.is_dir():
        return []  # a missing recipes_root is reported by the stage, with its own message

    try:
        live = per_recipe_hashes(live_root)
        sealed = per_recipe_hashes(Path(sealed_recipes_dir))
    except Exception as unhashable:
        raise RuntimeError(
            f"The recipe tree could not be checked against this run's sealed copy: "
            f"{unhashable}\nFix that file (or move it out of the recipe tree) — until "
            "it hashes, no one can tell whether the recipes this run would use are the "
            "ones it sealed."
        ) from unhashable

    drifted = []
    for key, sealed_scopes in sorted(sealed.items()):
        variable = key.split("/", 1)[0]
        if key in live and live[key] != sealed_scopes:
            changed = sorted(
                scope for scope, value in live[key].items()
                if sealed_scopes.get(scope) != value
            )
            drifted.append(f"the {variable} recipe ({', '.join(changed)})")
    return drifted


def variables_missing_from_the_sealed_bundle(cfg: dict, sealed_recipes_dir: Path) -> list[str]:
    """Planned variables whose recipe the sealed bundle does not contain.

    A variable added to ``recipes:`` after sealing would run a recipe the bundle never
    snapshotted: its receipts would carry no sealed per-recipe hashes and the run's
    provenance would claim less than it ran. Executing it under this run id is refused;
    it belongs in a fresh run (which will seal it)."""
    sealed_recipes = Path(sealed_recipes_dir)
    if not sealed_recipes.is_dir():
        return []
    planned = cfg.get("recipes") or []
    if not isinstance(planned, (list, tuple)):
        return []
    missing = []
    for variable in planned:
        if not any(sealed_recipes.rglob(f"{variable}/v*/{variable}_v*_recipe.yaml")):
            missing.append(str(variable))
    return missing


def why_a_sealed_run_cannot_continue(cfg: dict, run_root: Path) -> list[str]:
    """Every reason the settings/code in hand do not match this run's sealed bundle,
    in plain words. Empty means the run may continue (bundle-tamper verification is
    the caller's remaining check — it needs the bundle walk, not the config)."""
    run_root = Path(run_root)
    code_dir = run_root / "code"
    drifted: list[str] = []

    hashes_path = code_dir / "hashes.json"
    if hashes_path.is_file():
        sealed_hashes = (json.loads(hashes_path.read_text(encoding="utf-8")) or {}).get(
            "payload", {},
        )
        for hash_name, computer in (
            ("provider_config_hash", provider_config_hash),
            ("retrieval_config_hash", retrieval_config_hash),
        ):
            current_value = computer(cfg)
            sealed_value = sealed_hashes.get(hash_name)
            if current_value is None or sealed_value is None:
                continue
            if current_value != sealed_value:
                drifted.append(hash_name)

    sealed_cfg = sealed_config_of(run_root)
    if sealed_cfg:
        drifted.extend(run_invariant_settings_that_changed(cfg, sealed_cfg))

    drifted.extend(recipes_that_changed_since_sealing(cfg.get("recipes_root"), code_dir / "recipes"))
    drifted.extend(
        f"the {variable} recipe is not in this run's sealed bundle (added after sealing?)"
        for variable in variables_missing_from_the_sealed_bundle(cfg, code_dir / "recipes")
    )
    return sorted({IN_PLAIN_WORDS.get(name, name) for name in drifted})
