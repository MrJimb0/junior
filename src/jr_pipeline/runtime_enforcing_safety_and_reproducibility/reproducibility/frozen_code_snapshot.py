"""Build and seal the code-provenance bundle (``out/<run_id>/code/``).

Called once at the start of every run by ``cli.seal``, before any patient
work begins. The bundle captures everything needed to reproduce this run
WITHOUT the original git checkout:

  * ``config_resolved.yaml`` — the merged, env-interpolated cfg dict.
  * ``entry_point.json``    — argv (with ``--patient`` redacted) + run id.
  * ``git.json``            — sha, branch, dirty flag, remote.
  * ``env.json``            — Python version + key dependency versions.
  * ``dependencies.lock``   — full pip freeze of the active interpreter.
  * ``recipes/`` — snapshot copy of the recipe directory.
  * ``src/jr_pipeline/``    — snapshot copy of the package source.
  * ``allowlist_names.json`` — endpoint NAMES + attestations (no URLs).
  * ``hashes.json``         — code_lock_hash + scoped sub-hashes.
  * ``code.lock.json``      — the cross-link anchor that data-side
                               artifacts reference via produced_by.

The scrubbing invariant is enforced here — seal fails if any of the
config / entry-point bytes trips the PHI scrubber.

The returned ``code_lock_hash`` is the SHA-256 of the bundle's
substantive contents (excluding ``hashes.json`` and ``code.lock.json``,
which describe the bundle itself). It's the anchor every data-side
artifact references.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from jr_pipeline import __version__ as _pkg_version
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
    hash_file,
    hash_json,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
    ScrubReport,
    check_scrub,
    check_scrub_file,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (
    assert_scoped_hashes_grounded,
    code_lock_hash,
    per_recipe_hashes,
    provider_config_hash,
    retrieval_config_hash,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)


def resolve_repo_root() -> Path:
    """The checkout this package is running from — the tree ``build_and_seal``
    snapshots into the bundle.

    Shared by every caller that seals so a local cohort run and a cluster run
    snapshot the same tree; a caller that guessed its own root could seal one
    checkout's recipes while running another's code."""
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    # Installed as a wheel with no checkout above it: fall back to the package's
    # own parent so seal still produces a bundle rather than raising.
    return here.parent.parent


def _git_info(repo_root: Path) -> dict[str, Any]:
    """Return git sha / dirty flag / branch / tag / remote, if available."""

    def _run(args: list[str]) -> str | None:
        try:
            r = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=False)
            return r.stdout.strip() or None
        except FileNotFoundError:
            return None

    if not (repo_root / ".git").exists():
        return {"sha": None, "branch": None, "tag": None, "dirty": False, "remote": None, "available": False}
    status = _run(["git", "status", "--porcelain"]) or ""
    return {
        "sha": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "tag": _run(["git", "describe", "--tags", "--exact-match"]),
        "dirty": bool(status.strip()),
        "remote": _run(["git", "config", "--get", "remote.origin.url"]),
        "available": True,
    }


def _env_info() -> dict[str, Any]:
    """Return Python version, platform, and versions of key dependencies."""
    py = sys.version_info

    def _maybe_version(pkg: str) -> str | None:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _version
        try:
            return _version(pkg)
        except (PackageNotFoundError, Exception):
            return None

    # Names here are DISTRIBUTION names (what importlib.metadata knows), not import
    # names: PyYAML installs the ``yaml`` module but its distribution is "PyYAML", so
    # querying "yaml" silently records null instead of the installed version.
    pkgs = ["numpy", "polars", "pyarrow", "hnswlib", "torch", "transformers", "tokenizers", "jsonschema", "jinja2", "PyYAML"]
    return {
        "python": f"{py.major}.{py.minor}.{py.micro}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "jr_pipeline": _pkg_version,
        "packages": {pkg: _maybe_version(pkg) for pkg in pkgs},
    }


def _dependencies_lock() -> str:
    """Return `pip freeze` output for the active interpreter, or "" if pip is unavailable."""
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        return r.stdout
    except Exception:
        return ""


def _allowlist_names_only(allowlist_path: Path | None) -> list[dict[str, Any]]:
    """Extract endpoint name/attestation/provider from the allowlist; drop URLs and keys (HIPAA boundary)."""
    if allowlist_path is None or not Path(allowlist_path).is_file():
        return []
    with Path(allowlist_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [
        {"name": e.get("name"), "attestation": e.get("attestation"), "provider": e.get("provider")}
        for e in (data.get("allowed_endpoints") or [])
    ]


_TREE_SCAN_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml")
# Skip the scrub rule file itself — it legitimately contains the regex patterns it defines.
_TREE_SCAN_SKIP_RELPATHS = frozenset({"runtime_enforcing_safety_and_reproducibility/phi/phi_leak_prevention_checks.py"})
_TREE_COPY_IGNORE = ("__pycache__", "*.pyc", "*.pyo", "py.typed", ".DS_Store", "*.swp", "*.bak")


def _redact_patient_argv(entry_point: dict) -> dict:
    """Redact raw patient selectors from the sealed argv. A ``--patient <id>`` value
    is a raw patient id and must never enter the (shareable-post-scrub) code bundle."""
    cleaned = dict(entry_point)
    argv = cleaned.get("argv")
    if isinstance(argv, list):
        redacted = list(argv)
        for i, tok in enumerate(redacted):
            if tok in ("--patient", "--patient-id", "--patients") and i + 1 < len(redacted):
                redacted[i + 1] = "<redacted-patient>"
            elif isinstance(tok, str) and tok.startswith(("--patient=", "--patient-id=")):
                redacted[i] = tok.split("=", 1)[0] + "=<redacted-patient>"
        cleaned["argv"] = redacted
    return cleaned


def _scrub_tree(label: str, root: Path) -> None:
    """Scrub every text-shaped file under ``root``; aggregate all violations into one report before raising."""
    if not root.is_dir():
        return
    violations: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _TREE_SCAN_SUFFIXES:
            continue
        rel = p.relative_to(root).as_posix()
        if rel in _TREE_SCAN_SKIP_RELPATHS:
            continue
        report = check_scrub_file(f"{label}/{rel}", p)
        if not report.ok:
            violations.extend(report.violations)
    ScrubReport(target_name=label, ok=not violations, violations=violations).raise_if_bad()


def build_and_seal(
    *,
    run_id: str,
    run_root: Path,
    repo_root: Path,
    cfg: dict,
    entry_point: dict,
    recipes_root: Path | None = None,
    allowlist_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble and seal the code bundle at ``run_root/code/``; raise ``ScrubViolation`` on PHI."""
    run_root = Path(run_root)
    code_dir = run_root / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    config_resolved = dict(cfg)
    check_scrub("config_resolved.yaml", config_resolved).raise_if_bad()
    cfg_path = code_dir / "config_resolved.yaml"
    cfg_path.write_text(
        yaml.safe_dump(config_resolved, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    # Re-scan serialized text (ADR 0026): in-memory scrub walks values only;
    # the file scan catches anything introduced by YAML serialization.
    check_scrub_file("config_resolved.yaml", cfg_path).raise_if_bad()

    ep_clean = _redact_patient_argv(entry_point)
    check_scrub("entry_point.json", ep_clean).raise_if_bad()
    ep_path = code_dir / "entry_point.json"
    ep_path.write_text(json.dumps(ep_clean, indent=2, sort_keys=True), encoding="utf-8")
    check_scrub_file("entry_point.json", ep_path).raise_if_bad()

    git = _git_info(repo_root)
    (code_dir / "git.json").write_text(json.dumps(git, indent=2, sort_keys=True), encoding="utf-8")

    env = _env_info()
    (code_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")

    deps_path = code_dir / "dependencies.lock"
    deps_path.write_text(_dependencies_lock(), encoding="utf-8")
    check_scrub_file("dependencies.lock", deps_path).raise_if_bad()

    # Snapshot recipes + src into the bundle so a sealed run reproduces without
    # the external git checkout (ADR 0029). Scrub after copy so a stray
    # PHI-shaped string fails seal instead of landing in the bundle.
    recipes_src = recipes_root or (repo_root / "recipes")
    # A recipes_root that does not exist must fail the seal, not skip the copy: the
    # recipe tree IS part of the method, and silently omitting it yields a bundle
    # whose code_lock_hash claims to describe a run it does not cover.
    if not recipes_src.is_dir():
        raise FileNotFoundError(
            f"recipes_root {recipes_src} does not exist — refusing to seal a code "
            "bundle that omits the recipe tree."
        )
    dst = code_dir / "recipes"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(recipes_src, dst, ignore=shutil.ignore_patterns(*_TREE_COPY_IGNORE))
    _scrub_tree("recipes", dst)

    src_dst = code_dir / "src" / "jr_pipeline"
    if src_dst.exists():
        shutil.rmtree(src_dst)
    src_src = repo_root / "src" / "jr_pipeline"
    if src_src.is_dir():
        shutil.copytree(src_src, src_dst, ignore=shutil.ignore_patterns(*_TREE_COPY_IGNORE))
        _scrub_tree("src/jr_pipeline", src_dst)

    (code_dir / "allowlist_names.json").write_text(
        json.dumps({"allowed_endpoints": _allowlist_names_only(allowlist_path)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Compute hash BEFORE writing hashes.json / code.lock.json: those two are
    # bundle-metadata and are explicitly excluded from the hash itself (see
    # provenance.hashes._BUNDLE_METADATA_FILES) to avoid self-reference.
    clh = code_lock_hash(code_dir)

    bundled_recipes = code_dir / "recipes"
    recipes_hashes = per_recipe_hashes(bundled_recipes) if bundled_recipes.exists() else {}
    hashes_payload = {
        "code_lock_hash": clh,
        "config_hash": hash_json(config_resolved),
        "provider_config_hash": provider_config_hash(config_resolved),
        "retrieval_config_hash": retrieval_config_hash(config_resolved),
        "env_hash": hash_json(env),
        "dependencies_hash": hash_file(code_dir / "dependencies.lock"),
        "per_recipe": recipes_hashes,
    }
    # A configured stage must produce a non-None scoped hash — fail the seal if the
    # subset keys ever drift out of the real config (the gate going inert).
    assert_scoped_hashes_grounded(config_resolved)
    h_env = envelope_for(
        artifact_type="hashes",
        sensitivity="low",
        stream="code",
        run_id=run_id,
        step="code_seal",
        payload=hashes_payload,
        code_lock_hash=clh,
    )
    h_env["content_hash"] = hash_artifact_payload(h_env)
    validate_artifact(h_env, "hashes")
    (code_dir / "hashes.json").write_text(json.dumps(h_env, indent=2, sort_keys=True), encoding="utf-8")

    lock_payload = {"run_id": run_id, "code_lock_hash": clh, "sealed_at": datetime.now(UTC).isoformat()}
    lock_envelope = envelope_for(
        artifact_type="code_lock",
        sensitivity="low",
        stream="code",
        run_id=run_id,
        step="code_seal",
        payload=lock_payload,
        code_lock_hash=clh,
    )
    lock_envelope["content_hash"] = hash_artifact_payload(lock_envelope)
    validate_artifact(lock_envelope, "code_lock")
    (code_dir / "code.lock.json").write_text(
        json.dumps(lock_envelope, indent=2, sort_keys=True), encoding="utf-8"
    )

    return {"code_dir": code_dir, "code_lock_hash": clh, "hashes": h_env}
