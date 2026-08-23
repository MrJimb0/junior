"""Archetype is gated against a controlled vocabulary at load — it is written
verbatim into the irreversible exhaust, so an uncontrolled typo permanently
fragments the pooled corpus. A novel pattern uses the `archetype_proposed` seam."""
from pathlib import Path

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    VALID_ARCHETYPES,
    load_recipe,
)


def _write_recipe(tmp_path: Path, archetype_lines: list[str]) -> Path:
    var_dir = tmp_path / "var_extraction_recipes" / "myvar" / "v1"
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / "myvar_v1_output_schema.json").write_text('{"type": "object"}', encoding="utf-8")
    lines = [
        "name: myvar",
        "version: v1",
        *archetype_lines,
        "output_schema: myvar_v1_output_schema.json",
        "llm: {model: local_qwen}",
        "steps:",
        "  - id: only",
        "    kind: llm_only",
        "    prompt: prompts/p.md",
    ]
    recipe = var_dir / "myvar_v1_recipe.yaml"
    recipe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (var_dir / "prompts").mkdir(exist_ok=True)
    (var_dir / "prompts" / "p.md").write_text("# SYSTEM\nx\n# USER\n{{ patient_id }}\n", encoding="utf-8")
    return recipe


def test_known_archetype_loads(tmp_path):
    r = _write_recipe(tmp_path, ["archetype: enumeration"])
    assert load_recipe(r).front_matter["archetype"] == "enumeration"


def test_unknown_archetype_fails_load(tmp_path):
    r = _write_recipe(tmp_path, ["archetype: event_sequnce"])  # typo
    with pytest.raises(ValueError, match="unknown archetype"):
        load_recipe(r)


def test_missing_archetype_fails_load(tmp_path):
    r = _write_recipe(tmp_path, ["# (no archetype)"])
    with pytest.raises(ValueError, match="missing required 'archetype'"):
        load_recipe(r)


def test_proposed_archetype_is_the_discovery_seam(tmp_path):
    r = _write_recipe(tmp_path, ["archetype: longitudinal_trend", "archetype_proposed: true"])
    spec = load_recipe(r)
    assert spec.front_matter["archetype"] == "longitudinal_trend"  # allowed, flagged for promotion


def test_vocab_matches_the_six_known():
    assert VALID_ARCHETYPES == {
        "point_fact", "temporal_anchor", "event_sequence",
        "classification_adjudication", "enumeration", "causal",
    }
