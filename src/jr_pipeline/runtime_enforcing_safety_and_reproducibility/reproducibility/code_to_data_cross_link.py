"""Resolve ``code_lock_hash`` references.

When a data-side artifact carries a ``code_lock_hash`` (every receipt,
every result envelope), this module looks up the corresponding code
bundle and returns its contents. Used by:

  * ``cli._require_sealed_bundle`` — verify a step cmd's run has a seal.
  * ``inspect`` / ``diff-with-code-context`` — answer "what code produced
    this artifact?" by reaching from a data artifact to the recipes,
    prompts, and src that ran.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (
    code_lock_hash,
)


@dataclass(frozen=True)
class CodeBundleRef:
    run_id: str
    code_dir: Path
    code_lock_hash: str


def code_bundle_for(run_root: Path) -> CodeBundleRef:
    """Return the code bundle reference for one run root."""
    code_dir = Path(run_root) / "code"
    if not code_dir.is_dir():
        raise FileNotFoundError(f"No code bundle at {code_dir}")
    lock_path = code_dir / "code.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Missing code.lock.json in {code_dir}")
    with lock_path.open("r", encoding="utf-8") as f:
        lock = json.load(f)
    return CodeBundleRef(
        run_id=lock["payload"]["run_id"],
        code_dir=code_dir,
        code_lock_hash=lock["payload"]["code_lock_hash"],
    )


def verify_code_bundle(run_root: Path) -> tuple[bool, str]:
    """Recompute the directory hash and compare with code.lock.json."""
    ref = code_bundle_for(run_root)
    recomputed = code_lock_hash(ref.code_dir)
    ok = recomputed == ref.code_lock_hash
    msg = (
        f"match: {ref.code_lock_hash}"
        if ok
        else f"MISMATCH: stored={ref.code_lock_hash} recomputed={recomputed}"
    )
    return ok, msg
