"""How many chunks reach a prompt is a fact about the machine, not about the question.

A recipe's `top_n` is a research judgement: how much evidence does this variable need.
The hardware in front of it is a separate constraint, and the two were not separable —
making the shipped breast-oncology recipes run on a 48 GiB laptop meant editing `top_n`
inside recipes that are shared, versioned, and cited by sealed runs.

Chunks run to 512 tokens, so 20 of them is a ~10k-token prompt. Measured with the local
3B: 20 chunks peaks at 64 GiB and swaps until the run is killed; 7 peaks at 14 GiB and a
patient takes under two minutes. `max_chunks_per_prompt` in the project's settings caps
whatever a recipe asks for, so the recipe stays what a researcher wrote and the machine
says what it can hold. Unset, the recipe decides — which is what a cluster wants.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _top_n_of(recipe_path: Path) -> int:
    return int((yaml.safe_load(recipe_path.read_text())["reranking"])["top_n"])


def test_the_ceiling_caps_what_a_recipe_asks_for():
    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (  # noqa: E501
        StepContext,
    )

    assert "max_chunks_per_prompt" in StepContext.__dataclass_fields__, (
        "the ceiling must travel with the run, not be read from a step's own config"
    )


def test_the_step_routes_its_top_n_through_the_ceiling_policy():
    """The behaviour is tested on the policy function below; what remains to pin is
    that the step actually consults it, on the context's own ceiling, where top_n is
    decided — otherwise the policy could pass every test while nothing calls it."""
    import inspect

    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import (
        retrieve_and_prompt_step,
    )

    source = inspect.getsource(retrieve_and_prompt_step.RetrieveAndPromptHandler)
    assert "chunks_that_may_reach_one_prompt(top_n, ctx.max_chunks_per_prompt)" in source


def test_the_shipped_recipes_keep_their_own_judgement():
    """The point of the ceiling is that these did not have to be edited. If a future
    change narrows them for a laptop's sake, the separation has been lost again."""
    for recipe in (
        "oncology/general_oncology/stage/v1/stage_v1_recipe.yaml",
        "oncology/breast_oncology/date_of_diagnosis/v1/date_of_diagnosis_v1_recipe.yaml",
    ):
        assert _top_n_of(REPO / "var_extraction_recipes" / recipe) == 20, (
            f"{recipe} was narrowed in place; set max_chunks_per_prompt instead"
        )


def test_the_laptop_default_carries_a_ceiling():
    bundled = yaml.safe_load((REPO / "deployment/local/laptop.yaml").read_text())

    assert bundled.get("max_chunks_per_prompt"), (
        "a project made on a laptop inherits this file; without a ceiling it inherits "
        "prompts its machine cannot generate"
    )
    assert bundled["max_chunks_per_prompt"] <= 10


def test_the_ceiling_is_locked_to_the_run_it_started_with():
    """It decides what evidence the model saw, so it decides the values. Left free, a
    cohort could be extracted with seven chunks for some patients and twelve for others
    — not one dataset with a documented evidence budget, but two silently merged, under
    a sealed config naming only one of them.

    Widening it mid-cohort while debugging is a real thing to want; the answer is a
    fresh --run-id, which the drift error already offers."""
    from apps_and_interfaces.command_line_interface import _RUN_INVARIANT_CFG_KEYS

    assert "max_chunks_per_prompt" in _RUN_INVARIANT_CFG_KEYS


def test_every_invariant_key_has_a_plain_words_translation():
    """The internal key says nothing to someone who just edited a YAML and re-ran,
    which is how people reach the drift error. Tied to the key LIST, not to one
    phrase: a key added without a translation would reach the operator as
    `cfg.max_provenance_retries` — which is exactly how max_provenance_retries
    shipped, while its sibling had words."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (
        IN_PLAIN_WORDS,
        RUN_INVARIANT_CFG_KEYS,
    )

    for key in RUN_INVARIANT_CFG_KEYS:
        assert f"cfg.{key}" in IN_PLAIN_WORDS, f"cfg.{key} would reach the operator raw"
        assert "cfg." not in IN_PLAIN_WORDS[f"cfg.{key}"]


@pytest.mark.parametrize("recipe_asks,ceiling,expected", [(20, 7, 7), (5, 7, 5), (20, None, 20), (20, 0, 20)])
def test_the_ceiling_policy(recipe_asks, ceiling, expected):
    """Lower of the two; no ceiling (or an unset 0) means the recipe decides. Driven
    through the step's own policy function — the earlier version computed min() inside
    the test, which passed with the feature deleted."""
    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.retrieve_and_prompt_step import (
        chunks_that_may_reach_one_prompt,
    )

    assert chunks_that_may_reach_one_prompt(recipe_asks, ceiling) == expected


def _preflight(cfg: dict, monkeypatch, capsys) -> str:
    """The extract confirmation, with someone at the keyboard to see it."""
    from apps_and_interfaces.command_line_interface import _confirm_variables, _planned_variables

    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: True
    )
    monkeypatch.setattr("click.confirm", lambda *a, **k: False)
    _confirm_variables(cfg, _planned_variables(cfg), 9, "junior_x.yaml")
    return capsys.readouterr().out


def _cfg(**over) -> dict:
    base = {
        "recipes": ["stage"],
        "recipes_root": str(REPO / "var_extraction_recipes"),
        "allowlist_path": str(REPO / "deployment/local/llm_allowlist_local3b.yaml"),
    }
    base.update(over)
    return base


def test_the_preflight_says_how_much_chart_text_each_question_gets(monkeypatch, capsys):
    """The setting that decides whether a run finishes on this machine was the one thing
    the screen did not say. It showed the endpoint's token ceiling, which is the
    backstop, not the budget anyone tunes."""
    shown = _preflight(_cfg(max_chunks_per_prompt=7), monkeypatch, capsys)

    assert "7 chunks per prompt" in shown, shown
    assert "ask for up to 20" in shown, "it must say what it is capping"
    assert "junior_x.yaml" in shown, "no pointer to where the ceiling lives"


def test_no_ceiling_is_called_out_rather_than_left_blank(monkeypatch, capsys):
    shown = _preflight(_cfg(), monkeypatch, capsys)

    assert "no machine ceiling set" in shown, shown
    assert "max_chunks_per_prompt" in shown, "the setting to add is unnamed"


def test_a_ceiling_wider_than_the_recipes_does_not_claim_to_cap(monkeypatch, capsys):
    """On an A100 the ceiling may exceed what any recipe asks for. Saying it capped
    something would be false."""
    shown = _preflight(_cfg(max_chunks_per_prompt=50), monkeypatch, capsys)

    assert "50 chunks per prompt" in shown
    assert "capped here" not in shown


def test_a_generated_local_allowlist_carries_the_configured_safety_settings(tmp_path):
    """`junior run` in local mode generates a run-scoped allowlist. Generating a bare
    endpoint quietly discarded the configured one: the run path extracted with no
    prompt ceiling and no dtype while `junior extract` under the same config honoured
    both — and an oversized prompt on an in-process model does not fail, it takes the
    machine down with nothing in the log."""
    import types as _types

    import yaml as _yaml

    from jr_pipeline.runtime_infrastructure.cohort_runner import _resolve_extract_allowlist

    configured = tmp_path / "allowlist.yaml"
    configured.write_text(_yaml.safe_dump({"allowed_endpoints": [{
        "name": "local_qwen", "url": "./models/x", "provider": "local_hf",
        "attestation": "self_hosted", "default_model": "./models/x",
        "dtype": "float16", "max_prompt_tokens_cap": 8000,
    }]}), encoding="utf-8")
    settings = _types.SimpleNamespace(
        llm_mode="local", llm_local_model_path=tmp_path / "another_model",
        llm_allowlist=configured, data_root=tmp_path / "data",
    )

    generated = _resolve_extract_allowlist(settings, "20260101_000000_aaaa")
    endpoint = _yaml.safe_load(generated.read_text(encoding="utf-8"))["allowed_endpoints"][0]

    assert endpoint["max_prompt_tokens_cap"] == 8000
    assert endpoint["dtype"] == "float16"
    # Identity stays with the generator: the model the operator picked is the model.
    assert endpoint["url"] == str((tmp_path / "another_model").resolve())
    assert "allow_download" not in endpoint


def test_a_local_run_with_no_ceiling_anywhere_says_so_out_loud(tmp_path, capsys):
    import types as _types

    import yaml as _yaml

    from jr_pipeline.runtime_infrastructure.cohort_runner import _resolve_extract_allowlist

    settings = _types.SimpleNamespace(
        llm_mode="local", llm_local_model_path=tmp_path / "m",
        llm_allowlist=tmp_path / "missing.yaml", data_root=tmp_path / "data",
    )

    generated = _resolve_extract_allowlist(settings, "20260101_000000_aaaa")

    endpoint = _yaml.safe_load(generated.read_text(encoding="utf-8"))["allowed_endpoints"][0]
    assert "max_prompt_tokens_cap" not in endpoint
    warning = capsys.readouterr().out
    assert "no prompt-token ceiling" in warning
    assert "take this machine down" in warning


def test_the_sealed_config_names_the_allowlist_extraction_actually_loads(tmp_path):
    """sealed_config_base restores the file's own allowlist_path (the per-step gate
    compares the handed config against the sealed one, so the file's spelling must
    win there) — but for a local-mode run the loaded file is the generated
    run_config/ one, and provenance has to say so."""
    import types as _types

    from jr_pipeline.runtime_infrastructure.cohort_runner import _sealed_config

    settings = _types.SimpleNamespace(
        project="p", data_root=tmp_path / "data", index_options=None,
        sealed_config_base={"allowlist_path": "./deployment/local/llm_allowlist_local3b.yaml"},
    )
    cfgs = {
        "ingest": {"source_root": str(tmp_path / "charts")},
        "extract": {
            "recipes_root": "r", "recipes": [], "encoder": {}, "chunker": {},
            "allowlist_path": str(tmp_path / "data" / "run_config" / "local_llm_allowlist.yaml"),
        },
    }

    sealed = _sealed_config(settings, "20260101_000000_aaaa", cfgs)

    assert sealed["allowlist_path"] == "./deployment/local/llm_allowlist_local3b.yaml"
    assert sealed["allowlist_path_used"] == str(
        tmp_path / "data" / "run_config" / "local_llm_allowlist.yaml"
    )
