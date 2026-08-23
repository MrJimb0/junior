"""A stage command seals the run itself, and never seals over a changed config.

Sealing is bookkeeping the operator should not have to know about, so the first stage
command to touch a run writes the code bundle. The part that must not soften: once a
bundle exists it is the run's contract, and a config that no longer matches it is an
error rather than a reason to re-seal.
"""
from __future__ import annotations

from pathlib import Path

import click
import pytest
import yaml

from apps_and_interfaces.command_line_interface import (
    _ensure_sealed_bundle,
    _require_sealed_bundle,
)

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "20260101_000000_aa"


def _config(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "run_id": RUN_ID,
        "output_root": str(tmp_path / "data"),
        "recipes_root": str(REPO / "var_extraction_recipes"),
        "encoder": {"model_id": "./models/x", "max_tokens": 512, "pooling": "mean"},
        "chunker": {"kind": "token_window", "overlap": 128},
    }
    cfg.update(overrides)
    return cfg


def test_first_stage_seals_the_run(tmp_path):
    run_root = tmp_path / "run"
    assert not (run_root / "code" / "code.lock.json").exists()

    code_lock_hash = _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    assert code_lock_hash.startswith("sha256:")
    assert (run_root / "code" / "code.lock.json").is_file()
    assert (run_root / "code" / "hashes.json").is_file()
    assert (run_root / "code" / "recipes").is_dir(), "recipe tree missing from the bundle"


def test_auto_seal_writes_the_manifest_so_the_run_can_be_closed_out(tmp_path):
    """`summarize` reads manifest.json. A bundle without one leaves the run
    unfinishable — the failure only shows up at the end, after all the work."""
    import json

    run_root = tmp_path / "run"
    _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    manifest_path = run_root / "manifest.json"
    assert manifest_path.is_file(), "auto-seal left the run without a manifest"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = manifest.get("payload", manifest)
    assert payload["run_id"] == RUN_ID
    assert payload["code_lock_hash"].startswith("sha256:")


def test_second_stage_reuses_the_first_stage_bundle(tmp_path):
    run_root = tmp_path / "run"
    cfg = _config(tmp_path)
    first = _ensure_sealed_bundle(cfg=cfg, run_root=run_root)
    second = _ensure_sealed_bundle(cfg=cfg, run_root=run_root)
    assert first == second, "the run was re-sealed instead of reusing its bundle"


def test_auto_seal_does_not_paper_over_a_drifted_config(tmp_path):
    """The whole point of the bundle: a run is locked to the config that started it."""
    run_root = tmp_path / "run"
    _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    drifted = _config(tmp_path)
    drifted["encoder"] = {**drifted["encoder"], "max_tokens": 256}
    with pytest.raises(click.ClickException) as raised:
        _ensure_sealed_bundle(cfg=drifted, run_root=run_root)

    message = str(raised.value)
    # Says what changed in words the operator recognises, and how to get past it —
    # matching on behaviour, not phrasing, so rewording the message is not a failure.
    assert "embedding" in message or "index" in message, message
    assert "--new-run" in message, "the message does not say how to proceed"


def test_auto_sealed_bundle_passes_the_same_gate_an_explicit_seal_writes(tmp_path):
    run_root = tmp_path / "run"
    cfg = _config(tmp_path)
    _ensure_sealed_bundle(cfg=cfg, run_root=run_root)
    # The gate the stage commands run, unchanged, against an auto-written bundle.
    assert _require_sealed_bundle(cfg=cfg, run_root=run_root).startswith("sha256:")


def test_the_sealed_config_is_what_was_handed_in(tmp_path):
    run_root = tmp_path / "run"
    cfg = _config(tmp_path)
    _ensure_sealed_bundle(cfg=cfg, run_root=run_root)

    sealed = yaml.safe_load(
        (run_root / "code" / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    assert sealed["encoder"] == cfg["encoder"]
    assert sealed["chunker"] == cfg["chunker"]
