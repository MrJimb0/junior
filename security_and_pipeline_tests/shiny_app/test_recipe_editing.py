"""Starting a variable is a copy plus ten renames, so the app does it — and every
edit is gated by the pipeline's own recipe loader.

The CLI's `new-variable` was reverted with the ruling that authoring belongs in the
app; this is that surface. Two properties carry all the safety: a scaffolded copy is
a DRAFT the loader refuses to run (its prompts still ask the template's question),
and a save that the loader refuses never stands — the file on disk rolls back and
the loader's message reaches the editor.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import recipe_editing
import yaml

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe


def _a_recipes_tree(tmp_path: Path) -> Path:
    """A template shaped like the real ones: prompts, a schema, a python helper, and
    every cross-reference written as the `<variable>_<version>_` filename prefix."""
    root = tmp_path / "var_extraction_recipes"
    v1 = root / "basic" / "date_of_birth" / "v1"
    (v1 / "prompts").mkdir(parents=True)
    (v1 / "date_of_birth_v1_recipe.yaml").write_text(
        "name: date_of_birth\n"
        "version: v1\n"
        "collection: basic\n"
        "archetype: point_fact\n"
        "output_schema: date_of_birth_v1_output_schema.json\n"
        "llm:\n  model: local_qwen\n  max_tokens: 512\n"
        "depends_on: []\n"
        "steps:\n"
        "  - id: grab\n"
        "    kind: retrieve_and_prompt\n"
        "    retrieval:\n      kind: direct_parquet\n      table: demographics\n"
        "    prompt: prompts/date_of_birth_v1_demographics.md\n"
        "  - id: estimate\n"
        "    kind: python\n"
        "    module: date_of_birth_v1_python_helper.estimate_from_age\n",
        encoding="utf-8",
    )
    (v1 / "date_of_birth_v1_output_schema.json").write_text(
        json.dumps({"title": "date_of_birth output", "type": "object",
                    "properties": {"date_of_birth": {"type": ["string", "null"]}}}),
        encoding="utf-8",
    )
    (v1 / "date_of_birth_v1_python_helper.py").write_text(
        "def estimate_from_age(ctx):\n    return {}\n", encoding="utf-8")
    (v1 / "prompts" / "date_of_birth_v1_demographics.md").write_text(
        "Find the patient's date of birth.\n", encoding="utf-8")
    return root


def test_the_template_itself_loads(tmp_path):
    """The fixture must be a recipe the loader accepts, or every assertion below is
    about a broken template rather than about the scaffold."""
    root = _a_recipes_tree(tmp_path)
    load_recipe(root / "basic" / "date_of_birth" / "v1" / "date_of_birth_v1_recipe.yaml")


# ── scaffolding a new variable ──────────────────────────────────────────────────

def test_every_reference_in_a_scaffold_resolves_and_nothing_keeps_the_old_name(tmp_path):
    root = _a_recipes_tree(tmp_path)

    destination = recipe_editing.scaffold_new_recipe(
        "date_of_death", "date_of_birth", recipes_root=root,
    )

    assert destination == root / "basic" / "date_of_death" / "v1"
    raw = yaml.safe_load((destination / "date_of_death_v1_recipe.yaml").read_text())
    assert raw["name"] == "date_of_death"
    assert (destination / raw["output_schema"]).is_file()
    for step in raw["steps"]:
        if "prompt" in step:
            assert (destination / step["prompt"]).is_file()
        if "module" in step:
            helper_file = step["module"].rsplit(".", 1)[0] + ".py"
            assert (destination / helper_file).is_file()
    leftovers = [p.name for p in destination.rglob("*") if "date_of_birth" in p.name]
    assert leftovers == [], f"files still carrying the template's name: {leftovers}"
    assert "date_of_birth_v1" not in (destination / "date_of_death_v1_recipe.yaml").read_text()


def test_a_scaffold_is_a_draft_the_loader_refuses(tmp_path):
    root = _a_recipes_tree(tmp_path)
    destination = recipe_editing.scaffold_new_recipe(
        "date_of_death", "date_of_birth", recipes_root=root,
    )

    assert recipe_editing.is_draft(destination)
    with pytest.raises(ValueError, match="not finished yet"):
        load_recipe(destination / "date_of_death_v1_recipe.yaml")


def test_finishing_a_draft_makes_it_loadable(tmp_path):
    root = _a_recipes_tree(tmp_path)
    destination = recipe_editing.scaffold_new_recipe(
        "date_of_death", "date_of_birth", recipes_root=root,
    )

    refusal = recipe_editing.finish_draft(destination)

    assert refusal == ""
    assert not recipe_editing.is_draft(destination)
    load_recipe(destination / "date_of_death_v1_recipe.yaml")  # accepts it now


def test_a_bad_name_and_a_taken_name_are_refused_before_anything_exists(tmp_path):
    root = _a_recipes_tree(tmp_path)

    with pytest.raises(ValueError, match="usable variable name"):
        recipe_editing.scaffold_new_recipe("Date Of Death!", "date_of_birth", recipes_root=root)
    with pytest.raises(ValueError, match="already exists"):
        recipe_editing.scaffold_new_recipe("date_of_birth", "date_of_birth", recipes_root=root)
    assert not (root / "basic" / "Date Of Death!").exists()


def test_a_name_used_in_another_collection_is_refused(tmp_path):
    """A variable name resolves across every collection, so a scaffold landing in one
    collection must not shadow a variable that lives in another — that ambiguity
    would follow every run from then on (the collection-uniqueness guardrail)."""
    root = _a_recipes_tree(tmp_path)
    elsewhere = root / "oncology" / "date_of_death" / "v1"
    elsewhere.mkdir(parents=True)

    with pytest.raises(ValueError, match="already exists"):
        recipe_editing.scaffold_new_recipe("date_of_death", "date_of_birth", recipes_root=root)
    assert not (root / "basic" / "date_of_death").exists()


# ── a new version of an existing variable ───────────────────────────────────────

def test_a_new_version_is_rewired_loadable_and_not_a_draft(tmp_path):
    root = _a_recipes_tree(tmp_path)

    destination = recipe_editing.scaffold_new_version("date_of_birth", recipes_root=root)

    assert destination == root / "basic" / "date_of_birth" / "v2"
    assert not recipe_editing.is_draft(destination), (
        "the content IS this variable's — a version bump is not a draft"
    )
    spec = load_recipe(destination / "date_of_birth_v2_recipe.yaml")
    assert spec.version == "v2"
    assert not list(destination.rglob("*_v1_*")), "files still carrying the old version"


# ── editing, with the loader as the gate ────────────────────────────────────────

def test_an_edit_the_loader_refuses_is_rolled_back_with_its_message(tmp_path):
    root = _a_recipes_tree(tmp_path)
    version_dir = root / "basic" / "date_of_birth" / "v1"
    recipe_rel = "date_of_birth_v1_recipe.yaml"
    before = recipe_editing.read_file(version_dir, recipe_rel)

    refusal = recipe_editing.save_file(
        version_dir, recipe_rel, before.replace("version: v1", "version: v9"),
    )

    assert refusal != "", "an edit the loader refuses must not stand"
    assert "v9" in refusal or "version" in refusal
    assert recipe_editing.read_file(version_dir, recipe_rel) == before, (
        "the refused edit is still on disk"
    )


def test_a_sound_edit_stands(tmp_path):
    root = _a_recipes_tree(tmp_path)
    version_dir = root / "basic" / "date_of_birth" / "v1"
    prompt_rel = "prompts/date_of_birth_v1_demographics.md"

    refusal = recipe_editing.save_file(
        version_dir, prompt_rel, "Find the patient's DATE OF BIRTH, exactly as written.\n",
    )

    assert refusal == ""
    assert "exactly as written" in recipe_editing.read_file(version_dir, prompt_rel)


def test_a_draft_stays_editable_while_it_is_a_draft(tmp_path):
    """The one loader error a save tolerates is the draft refusal itself — otherwise
    a scaffold could never be edited into shape through the surface built for it."""
    root = _a_recipes_tree(tmp_path)
    destination = recipe_editing.scaffold_new_recipe(
        "date_of_death", "date_of_birth", recipes_root=root,
    )

    refusal = recipe_editing.save_file(
        destination, "prompts/date_of_death_v1_demographics.md",
        "Find the patient's date of death.\n",
    )

    assert refusal == ""
    assert recipe_editing.is_draft(destination), "editing a draft must not finish it"


def test_a_broken_draft_cannot_be_finished(tmp_path):
    root = _a_recipes_tree(tmp_path)
    destination = recipe_editing.scaffold_new_recipe(
        "date_of_death", "date_of_birth", recipes_root=root,
    )
    schema = destination / "date_of_death_v1_output_schema.json"
    schema.write_text("{ not json", encoding="utf-8")

    refusal = recipe_editing.finish_draft(destination)

    assert refusal != ""
    assert recipe_editing.is_draft(destination), (
        "a draft that does not load was finished anyway"
    )


def test_the_editor_cannot_reach_outside_the_recipe_version(tmp_path):
    root = _a_recipes_tree(tmp_path)
    version_dir = root / "basic" / "date_of_birth" / "v1"

    with pytest.raises(ValueError, match="not inside"):
        recipe_editing.read_file(version_dir, "../../../secrets.yaml")
    with pytest.raises(ValueError, match="not an editable"):
        recipe_editing.save_file(version_dir, "notes.txt", "x")


# ── listing ─────────────────────────────────────────────────────────────────────

def test_listing_labels_drafts_as_drafts(tmp_path):
    root = _a_recipes_tree(tmp_path)
    recipe_editing.scaffold_new_recipe("date_of_death", "date_of_birth", recipes_root=root)

    labels = {v.label for v in recipe_editing.list_recipe_versions(recipes_root=root)}

    assert "basic/date_of_birth (v1)" in labels
    assert "basic/date_of_death (v1) · draft" in labels
