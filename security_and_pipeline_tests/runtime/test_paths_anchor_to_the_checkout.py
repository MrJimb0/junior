"""Config paths mean "in the checkout", not "wherever you happened to be standing".

The shipped configs name repo locations as `./models/...` and `./var_extraction_recipes`.
Once the CLI could be run from any directory, a bare relative path resolved against the
caller's cwd and found nothing — and the way that failed is the reason these tests
exist: a missing extraction model does not raise, it produces an empty answer with a
warning, so a run reports success and a null value that reads as "not in the chart"
rather than "we never looked".
"""
from __future__ import annotations

from pathlib import Path

import yaml

from apps_and_interfaces.command_line_interface import load_cohort_config
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_allowlist import (
    load_allowlist,
)

REPO = Path(__file__).resolve().parents[2]


def test_a_relative_model_dir_resolves_to_the_checkout(tmp_path, monkeypatch):
    """The failure this prevents: run from elsewhere, get a null DOB and no error."""
    monkeypatch.chdir(tmp_path)
    allowlist_path = tmp_path / "allowlist.yaml"
    allowlist_path.write_text(
        yaml.safe_dump({"allowed_endpoints": [{
            "name": "local_qwen",
            "url": "./models",
            "provider": "local_hf",
            "attestation": "self_hosted",
        }]}),
        encoding="utf-8",
    )

    endpoint = load_allowlist(allowlist_path).get("local_qwen")
    assert Path(endpoint.url).is_absolute(), "model dir left relative to the caller's cwd"
    assert Path(endpoint.url).is_dir(), f"anchored to a directory that does not exist: {endpoint.url}"


def test_a_hub_model_name_is_left_alone(tmp_path, monkeypatch):
    """`Qwen/Qwen2.5-3B` is a name on the HuggingFace Hub, not a path. Turning it into
    an absolute path would break every endpoint that downloads its model."""
    monkeypatch.chdir(tmp_path)
    allowlist_path = tmp_path / "allowlist.yaml"
    allowlist_path.write_text(
        yaml.safe_dump({"allowed_endpoints": [{
            "name": "hub",
            "url": "Qwen/Qwen2.5-0.5B-Instruct",
            "provider": "local_hf",
            "attestation": "self_hosted",
        }]}),
        encoding="utf-8",
    )

    assert load_allowlist(allowlist_path).get("hub").url == "Qwen/Qwen2.5-0.5B-Instruct"


def test_config_paths_anchor_to_the_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("JR_RUN_ID", "JR_DATA_ROOT", "JUNIOR_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "junior.yaml"
    config.write_text(
        "run_id: R1\n"
        "recipes_root: ./var_extraction_recipes\n"
        "allowlist_path: ./deployment/local/llm_allowlist_local3b.yaml\n"
        "encoder:\n  model_id: ./models/embedding/x\n",
        encoding="utf-8",
    )

    cfg, _ = load_cohort_config(config, None)

    assert cfg["recipes_root"] == str(REPO / "var_extraction_recipes")
    assert cfg["allowlist_path"] == str(REPO / "deployment/local/llm_allowlist_local3b.yaml")
    assert cfg["encoder"]["model_id"] == str(REPO / "models/embedding/x")


def test_an_absolute_path_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("JR_RUN_ID", "JR_DATA_ROOT", "JUNIOR_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    elsewhere = tmp_path / "somewhere" / "recipes"
    config = tmp_path / "junior.yaml"
    config.write_text(f"run_id: R1\nrecipes_root: {elsewhere}\n", encoding="utf-8")

    cfg, _ = load_cohort_config(config, None)
    assert cfg["recipes_root"] == str(elsewhere)


def test_a_column_map_beside_the_settings_file_is_found(tmp_path, monkeypatch):
    """`junior columns` writes the map beside the settings file and records the bare
    name; a project living outside the checkout must still find it — resolving
    against the repo root alone broke the CLI's own advertised flow for every
    such project."""
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "lakeside"
    project.mkdir()
    (project / "site_map.yaml").write_text(
        yaml.safe_dump({"chunk_metadata_columns": {"notes": {"text_columns": ["NoteText"]}}}),
        encoding="utf-8",
    )
    config = project / "junior.yaml"
    config.write_text(
        yaml.safe_dump({
            "output_root": str(tmp_path / "data"),
            "chart_columns_file": "site_map.yaml",
        }),
        encoding="utf-8",
    )

    cfg, _ = load_cohort_config(config, None)
    assert cfg["chart_columns_file"] == str(project / "site_map.yaml")


def test_a_repo_spelling_of_the_column_map_still_anchors_to_the_checkout(tmp_path, monkeypatch):
    """A deployment/<site>/... spelling in a config anywhere on disk means the
    checkout's copy, exactly as before."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "junior.yaml"
    config.write_text(
        yaml.safe_dump({
            "output_root": str(tmp_path / "data"),
            "chart_columns_file": "deployment/example_site/example_export_column_map.yaml",
        }),
        encoding="utf-8",
    )

    cfg, _ = load_cohort_config(config, None)
    assert Path(cfg["chart_columns_file"]).is_absolute()
    assert Path(cfg["chart_columns_file"]).is_file(), "repo copy not found"
