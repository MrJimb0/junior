"""Once a run is sealed, every door back into it checks the seal — none rewrites it.

Three doors re-enter a sealed run: the per-stage commands, `junior run` (and the app's
Run button, which spawns it), and `junior seal`. Re-sealing deletes and recopies code/
and rewrites hashes.json — the provenance every existing receipt claims — and, because
resume compares recorded hashes against the sealed index, a re-seal under edited code
silently turns finished variables back into pending ones. So the contract is: an
absent bundle is written once; an existing bundle is verified and drift-checked, and
drift refuses with a next step that keeps the corpus-inheritance path open.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

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


# ── the run-invariant comparison sees additions and removals ────────────────────

def test_an_invariant_setting_added_after_sealing_is_drift(tmp_path):
    """`key in cfg and key in sealed_cfg` was the old test, and it let a setting that
    was ABSENT at sealing and set afterwards walk straight past the gate — absent-then-
    set changes the prompts exactly as much as set-then-changed does."""
    run_root = tmp_path / "run"
    _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    with pytest.raises(click.ClickException) as refusal:
        _require_sealed_bundle(
            cfg=_config(tmp_path, max_chunks_per_prompt=12), run_root=run_root,
        )

    assert "how many chunks" in str(refusal.value)


def test_an_invariant_setting_removed_after_sealing_is_drift(tmp_path):
    run_root = tmp_path / "run"
    _ensure_sealed_bundle(
        cfg=_config(tmp_path, max_provenance_retries=2), run_root=run_root,
    )

    with pytest.raises(click.ClickException) as refusal:
        _require_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    assert "re-asks" in str(refusal.value)


# ── a tree that cannot be hashed refuses instead of waving through ──────────────

def test_an_unhashable_recipe_tree_refuses_instead_of_waving_through(tmp_path):
    """One malformed file used to disable the whole recipe-drift gate silently —
    the least-checkable tree became the most trusted one."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
        recipes_that_changed_since_sealing,
    )

    for root in ("recipes/basic/thing/v1", "code/recipes/basic/thing/v1"):
        d = tmp_path / root
        d.mkdir(parents=True)
        (d / "thing_v1_recipe.yaml").write_text("a: 1\n", encoding="utf-8")
    broken = tmp_path / "recipes" / "basic" / "thing" / "v1" / "thing_v1_output_schema.json"
    broken.write_text("{ not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="could not be checked"):
        recipes_that_changed_since_sealing(tmp_path / "recipes", tmp_path / "code" / "recipes")


# ── a variable the bundle never sealed does not run under this run id ───────────

def test_a_variable_added_to_the_plan_after_sealing_refuses(tmp_path):
    """A variable added to ``recipes:`` after sealing would run a recipe the bundle
    never snapshotted: its receipts would carry no sealed hashes and the run's
    provenance would claim less than it ran."""
    run_root = tmp_path / "run"
    _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)

    with pytest.raises(click.ClickException) as refusal:
        _require_sealed_bundle(
            cfg=_config(tmp_path, recipes=["date_of_birth", "added_after_sealing"]),
            run_root=run_root,
        )

    assert "added_after_sealing" in str(refusal.value)
    assert "sealed bundle" in str(refusal.value)


# ── `junior run` re-entry: verify and continue, never re-seal ───────────────────

def test_continuing_a_sealed_run_returns_the_hash_without_rewriting(tmp_path):
    from jr_pipeline.runtime_infrastructure.cohort_runner import _continue_the_sealed_run

    run_root = tmp_path / "run"
    sealed_hash = _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)
    hashes_before = (run_root / "code" / "hashes.json").read_bytes()
    lock_before = (run_root / "code" / "code.lock.json").read_bytes()

    continued = _continue_the_sealed_run(run_root, _config(tmp_path))

    assert continued == sealed_hash
    assert (run_root / "code" / "hashes.json").read_bytes() == hashes_before, (
        "re-entering a run rewrote the provenance its receipts claim"
    )
    assert (run_root / "code" / "code.lock.json").read_bytes() == lock_before


def test_continuing_a_sealed_run_refuses_drift_and_names_the_inheritance_path(tmp_path):
    """The refusal's next steps must include the one that keeps the corpus: telling a
    recipe-editor only `--run-id <new-name>` sends them to a full rebuild the
    `extract --new-run` path exists to avoid."""
    from jr_pipeline.runtime_infrastructure.cohort_runner import _continue_the_sealed_run

    run_root = tmp_path / "run"
    _ensure_sealed_bundle(cfg=_config(tmp_path), run_root=run_root)
    hashes_before = (run_root / "code" / "hashes.json").read_bytes()

    with pytest.raises(RuntimeError) as refusal:
        _continue_the_sealed_run(
            run_root, _config(tmp_path, allowlist_path=str(tmp_path / "other.yaml")),
        )

    assert "allow-list" in str(refusal.value)
    assert "extract --new-run" in str(refusal.value)
    assert (run_root / "code" / "hashes.json").read_bytes() == hashes_before


# ── mid-cohort recipe edits stop the run instead of mis-stamping receipts ───────

def test_a_recipe_edited_mid_cohort_stops_extraction_before_the_next_patient(
    tmp_path, monkeypatch, capsys,
):
    """Extraction reads recipes live while receipts stamp the sealed hashes, so every
    patient extracted after a mid-run edit would claim provenance that did not run."""
    import shutil
    import types as _types

    import jr_pipeline.pipeline_steps.step_7_extract_variables.extract as extract_module
    from jr_pipeline.runtime_infrastructure import cohort_runner
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    live = tmp_path / "recipes" / "basic" / "thing" / "v1"
    live.mkdir(parents=True)
    (live / "thing_v1_recipe.yaml").write_text("reranking:\n  top_n: 20\n", encoding="utf-8")
    sealed_recipes = phi_intermediate_run_dir(RUN_ID) / "code" / "recipes"
    shutil.copytree(tmp_path / "recipes", sealed_recipes)
    (live / "thing_v1_recipe.yaml").write_text("reranking:\n  top_n: 13\n", encoding="utf-8")

    extracted = []
    monkeypatch.setattr(
        extract_module, "run_extract_one",
        lambda **kw: extracted.append(kw["patient_id"]) or {"variables": {}, "n_failed": 0},
    )
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text("allowed_endpoints: []\n", encoding="utf-8")
    settings = _types.SimpleNamespace(variables=["thing"])
    cfg = {"run_id": RUN_ID, "recipes_root": str(tmp_path / "recipes"),
           "allowlist_path": str(allowlist)}

    status = cohort_runner._run_extract(["P1", "P2"], settings, cfg)

    assert extracted == [], "extracted a patient under a recipe the seal does not hold"
    assert status == {"P1": "stopped: recipes changed mid-run",
                      "P2": "stopped: recipes changed mid-run"}
    out = capsys.readouterr().out
    assert "recipe changed while this run was extracting" in out
    assert "extract --new-run" in out


# ── `junior seal` on an already-sealed run keeps the bundle ─────────────────────

def test_resealing_the_same_settings_keeps_the_bundle_and_updates_the_roster(tmp_path):
    """Sealing ahead of a SLURM fan-out runs `seal` after a stage may already have
    auto-sealed. Same settings: the bundle stays byte-identical; the roster (the
    reason to re-run seal) is still refreshed."""
    import yaml as _yaml
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main as cli_main

    (tmp_path / "charts").mkdir()
    config = tmp_path / "junior.yaml"
    config.write_text(_yaml.safe_dump({
        "run_id": RUN_ID,
        "source_root": str(tmp_path / "charts"),
        "output_root": str(tmp_path / "data"),
        "recipes_root": str(REPO / "var_extraction_recipes"),
        "encoder": {"model_id": "./models/x", "max_tokens": 512, "pooling": "mean"},
        "chunker": {"kind": "token_window", "overlap": 128},
    }), encoding="utf-8")
    runner = CliRunner()

    first = runner.invoke(cli_main, ["seal", "--config", str(config)])
    assert first.exit_code == 0, first.output
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )
    code_dir = phi_intermediate_run_dir(RUN_ID, Path(tmp_path / "data")) / "code"
    hashes_before = (code_dir / "hashes.json").read_bytes()

    roster = tmp_path / "patients.txt"
    roster.write_text("P1\nP2\n", encoding="utf-8")
    second = runner.invoke(
        cli_main, ["seal", "--config", str(config), "--patients-file", str(roster)],
    )

    assert second.exit_code == 0, second.output
    assert "already sealed" in second.output
    assert (code_dir / "hashes.json").read_bytes() == hashes_before
    manifest = json.loads((code_dir.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["payload"]["n_target_patients"] == 2
