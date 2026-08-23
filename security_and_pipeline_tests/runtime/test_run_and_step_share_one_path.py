"""`jr-pipeline run` and the per-step commands must be one path, not two.

`run` seals via the cohort runner; the per-step commands then check the config they
are handed against that sealed bundle. If the runner rewrites any config value on the
way in — resolving a relative model path, injecting a chunker the file never named,
dropping the index block — the same config file that produced the run is rejected as
drifted, and the per-step commands become a separate workflow with their own config
dialect. These tests pin the round trip.
"""
from pathlib import Path

import pytest

from apps_and_interfaces.command_line_interface import _require_sealed_bundle
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    build_and_seal,
)
from jr_pipeline.runtime_infrastructure.cohort_runner import (
    _build_step_configs,
    _sealed_config,
    cohort_settings_from_config,
)
from jr_pipeline.runtime_infrastructure.config_loading import load_config

REPO = Path(__file__).resolve().parents[2]
LAPTOP_CONFIG = REPO / "deployment" / "local" / "laptop.yaml"
RUN_ID = "20260101_000000_aa"


@pytest.fixture
def laptop_config(monkeypatch, tmp_path) -> dict:
    monkeypatch.setenv("JR_RUN_ID", RUN_ID)
    monkeypatch.setenv("JR_SOURCE_ROOT", str(REPO / "examples"))
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    return load_config(LAPTOP_CONFIG)


def _seal_as_run_does(cfg: dict, run_root: Path) -> None:
    settings = cohort_settings_from_config(cfg, variables=())
    cfgs = _build_step_configs(settings, RUN_ID)
    build_and_seal(
        run_id=RUN_ID,
        run_root=run_root,
        repo_root=REPO,
        cfg=_sealed_config(settings, RUN_ID, cfgs),
        entry_point={"argv_program": "pytest", "argv": [], "step": "seal"},
        recipes_root=Path(settings.recipes_root),
    )


def test_step_command_accepts_the_bundle_run_sealed(laptop_config, tmp_path):
    run_root = tmp_path / "run"
    _seal_as_run_does(laptop_config, run_root)

    # The gate the per-step commands run. Raises ClickException on drift.
    code_lock_hash = _require_sealed_bundle(cfg=laptop_config, run_root=run_root)
    assert code_lock_hash.startswith("sha256:")


def test_step_command_still_rejects_a_genuinely_drifted_config(laptop_config, tmp_path):
    """The round trip must not have been bought by making the gate inert."""
    import click

    run_root = tmp_path / "run"
    _seal_as_run_does(laptop_config, run_root)

    drifted = dict(laptop_config)
    drifted["encoder"] = {**laptop_config["encoder"], "max_tokens": 256}
    with pytest.raises(click.ClickException) as raised:
        _require_sealed_bundle(cfg=drifted, run_root=run_root)
    # Matched on behaviour rather than phrasing: the gate must fire and must name
    # what changed, but the wording is a UI decision that will keep moving.
    assert "embedding" in str(raised.value) or "index" in str(raised.value)


def test_config_keys_survive_into_the_sealed_config(laptop_config):
    """Values the config names must reach the bundle unrewritten."""
    settings = cohort_settings_from_config(laptop_config, variables=())
    sealed = _sealed_config(settings, RUN_ID, _build_step_configs(settings, RUN_ID))

    assert sealed["encoder"] == laptop_config["encoder"], "encoder was rewritten"
    assert sealed["index"] == laptop_config["index"], "index block was dropped"
    assert sealed["chunker"] == laptop_config["chunker"], "chunker was rewritten"


def test_site_column_map_reaches_the_stage_configs(laptop_config):
    """The per-step CLI hands each stage the whole config file, so the cohort path
    must hand the same interpretation keys — or `run` and the app read charts with
    generic column aliases while `junior embed` reads them with the site's map."""
    cfg = dict(laptop_config)
    cfg["chart_columns_file"] = str(REPO / "deployment" / "local" / "laptop.yaml")
    cfg["chunk_metadata_columns"] = {"notes": {"author": ["note_author"]}}
    cfg["text_column"] = "note_text"

    settings = cohort_settings_from_config(cfg, variables=())
    cfgs = _build_step_configs(settings, RUN_ID)

    for stage in ("ingest", "embed"):
        assert cfgs[stage]["chart_columns_file"] == cfg["chart_columns_file"], stage
        assert cfgs[stage]["chunk_metadata_columns"] == cfg["chunk_metadata_columns"], stage
    assert cfgs["embed"]["text_column"] == "note_text"


def test_an_unmapped_cohort_gets_no_null_map_keys(laptop_config):
    """Absent means absent: a null chart_columns_file in a stage config would read
    as "this run explicitly set no map", which is not what an unset setting means."""
    settings = cohort_settings_from_config(dict(laptop_config), variables=())
    cfgs = _build_step_configs(settings, RUN_ID)
    for stage in ("ingest", "embed"):
        assert "chart_columns_file" not in cfgs[stage]
        assert "chunk_metadata_columns" not in cfgs[stage]
    assert "text_column" not in cfgs["embed"]
