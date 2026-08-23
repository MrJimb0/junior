"""A step refuses to run when the sealed code bundle's contents drifted since sealing.
_require_sealed_bundle recomputes the bundle hash (verify_code_bundle) rather than
trusting presence + config-drift alone, so code edited after sealing can't silently run
under the stale code_lock_hash the receipts will claim."""
from __future__ import annotations

import json

import click
import pytest

from apps_and_interfaces.command_line_interface import _require_sealed_bundle
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (
    code_lock_hash,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_to_data_cross_link import (
    verify_code_bundle,
)


def _seal(run_root, body: str = "sealed source"):
    code = run_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "src_snapshot.txt").write_text(body)
    lock = code_lock_hash(code)  # excludes code.lock.json/hashes.json
    (code / "code.lock.json").write_text(
        json.dumps({"payload": {"run_id": "R", "code_lock_hash": lock}})
    )
    (code / "hashes.json").write_text(json.dumps({"payload": {}}))
    return code


def test_intact_bundle_passes(tmp_path):
    _seal(tmp_path)
    assert _require_sealed_bundle(cfg={"run_id": "R"}, run_root=tmp_path) is not None


def test_drifted_bundle_refuses_to_run(tmp_path):
    """Tampering with the SNAPSHOT — a bundle altered in transit or on a share."""
    code = _seal(tmp_path)
    (code / "src_snapshot.txt").write_text("tampered after sealing")

    ok, _ = verify_code_bundle(tmp_path)
    assert ok is False
    with pytest.raises(click.ClickException) as raised:
        _require_sealed_bundle(cfg={"run_id": "R"}, run_root=tmp_path)
    # Matched on behaviour, not phrasing: the gate must refuse and must say how to
    # proceed. "drifted" was internal vocabulary that a researcher never sees now.
    message = str(raised.value)
    assert "changed" in message
    # Starting over must never send someone to invent a run id — the fresh-run
    # path is the --new-run family, which the guidance now names.
    assert "--new-run" in message, "the refusal does not say how to get past it"


def test_a_recipe_edited_after_sealing_refuses_to_run(tmp_path, monkeypatch):
    """The case the bundle recompute never covered, and its test did not either.

    Extraction reads recipes LIVE (extract.py resolves cfg["recipes_root"]) and then
    stamps each receipt with hashes read from the sealed hashes.json. So editing a
    recipe and re-extracting into the same run made every receipt assert the hash of a
    recipe that did not run — and `junior diff-with-code-context` would have reported
    "values changed, recipe unchanged", the opposite of the truth.

    The recompute above cannot catch it: it hashes `<run>/code/`, which an edit to the
    working tree never touches. The test beside it tampered with a file INSIDE the
    bundle, so a gate blind to the live tree looked covered."""
    from apps_and_interfaces.command_line_interface import (
        _recipes_that_changed_since_sealing,
    )

    live = tmp_path / "recipes" / "basic" / "thing" / "v1"
    live.mkdir(parents=True)
    (live / "thing_v1_recipe.yaml").write_text("reranking:\n  top_n: 20\n", encoding="utf-8")
    sealed_root = tmp_path / "code" / "recipes"
    sealed = sealed_root / "basic" / "thing" / "v1"
    sealed.mkdir(parents=True)
    (sealed / "thing_v1_recipe.yaml").write_text("reranking:\n  top_n: 20\n", encoding="utf-8")

    cfg = {"recipes_root": str(tmp_path / "recipes"), "recipes": ["thing"]}
    assert _recipes_that_changed_since_sealing(cfg, tmp_path / "code") == []

    (live / "thing_v1_recipe.yaml").write_text("reranking:\n  top_n: 13\n", encoding="utf-8")

    drifted = _recipes_that_changed_since_sealing(cfg, tmp_path / "code")
    assert drifted, "a recipe edited after sealing was not noticed"
    assert "thing" in drifted[0] and "recipe_hash" in drifted[0], drifted


def test_a_recipe_added_after_sealing_is_not_drift(tmp_path):
    """A shared checkout gains recipes while runs are in flight. A run never used one
    that did not exist when it was sealed, so it cannot have drifted."""
    from apps_and_interfaces.command_line_interface import (
        _recipes_that_changed_since_sealing,
    )

    for root in ("recipes/basic/thing/v1", "code/recipes/basic/thing/v1"):
        d = tmp_path / root
        d.mkdir(parents=True)
        (d / "thing_v1_recipe.yaml").write_text("a: 1\n", encoding="utf-8")
    newcomer = tmp_path / "recipes" / "basic" / "later" / "v1"
    newcomer.mkdir(parents=True)
    (newcomer / "later_v1_recipe.yaml").write_text("b: 2\n", encoding="utf-8")

    cfg = {"recipes_root": str(tmp_path / "recipes"), "recipes": ["thing"]}

    assert _recipes_that_changed_since_sealing(cfg, tmp_path / "code") == []


def test_a_hashing_failure_does_not_block_a_run(tmp_path):
    """This gate protects an audit claim; it must not become a way for a run to die."""
    from apps_and_interfaces.command_line_interface import (
        _recipes_that_changed_since_sealing,
    )

    assert _recipes_that_changed_since_sealing({}, tmp_path / "code") == []
    assert _recipes_that_changed_since_sealing(
        {"recipes_root": str(tmp_path / "gone")}, tmp_path / "code"
    ) == []
