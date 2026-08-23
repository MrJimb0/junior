"""An unresolved recipes_root must fail loudly, never silently seal a bundle whose
code_lock_hash claims to describe a run the recipe tree was dropped from."""
from pathlib import Path

import pytest

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    build_and_seal,
    resolve_repo_root,
)

REPO = Path(__file__).resolve().parents[2]


def _seal(tmp_path: Path, recipes_root: Path) -> dict:
    return build_and_seal(
        run_id="20260101_000000_aa",
        run_root=tmp_path / "run",
        repo_root=REPO,
        cfg={"run_id": "20260101_000000_aa"},
        entry_point={"argv_program": "pytest", "argv": [], "step": "seal"},
        recipes_root=recipes_root,
    )


def test_seal_raises_on_missing_recipes_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        _seal(tmp_path, tmp_path / "nope")


def test_seal_hashes_real_recipes_root(tmp_path):
    sealed = _seal(tmp_path, REPO / "var_extraction_recipes")
    h = sealed["code_lock_hash"]
    assert h.startswith("sha256:") and len(h) == len("sha256:") + 64


def test_resolve_repo_root_finds_the_checkout():
    assert (resolve_repo_root() / "pyproject.toml").is_file()
