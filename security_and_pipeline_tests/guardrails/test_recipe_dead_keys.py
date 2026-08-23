"""Dead and engine-default recipe keys are rejected loudly, not silently absorbed.

The top-level `retrieval`/`model_policy`/`feedback` blocks and `evidence.formatting_style`
are dead -- nothing reads them. `temperature` and `response_format` are engine defaults the
engine fixes (greedy decoding + JSON output), not recipe knobs. The loader must raise on a
recipe carrying any of these, while a recipe that only uses the live surface (step-level
retrieval, evidence.max_context_tokens, llm.model/max_tokens) still loads.
A set-difference against EvidenceSpec's own fields polices the closed `evidence:` block; an
explicit deny list catches the dead top-level blocks and the two engine-default llm keys."""
from pathlib import Path

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    load_recipe,
)
from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files


def _write_recipe(tmp_path: Path, body_lines: list[str], llm_line: str = "llm: {model: local_qwen}") -> Path:
    var_dir = tmp_path / "var_extraction_recipes" / "myvar" / "v1"
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / "myvar_v1_output_schema.json").write_text('{"type": "object"}', encoding="utf-8")
    lines = [
        "name: myvar",
        "version: v1",
        "archetype: point_fact",
        "output_schema: myvar_v1_output_schema.json",
        llm_line,
        *body_lines,
    ]
    recipe = var_dir / "myvar_v1_recipe.yaml"
    recipe.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (var_dir / "prompts").mkdir(exist_ok=True)
    (var_dir / "prompts" / "p.md").write_text("# SYSTEM\nx\n# USER\n{{ patient_id }}\n", encoding="utf-8")
    return recipe


_VALID_STEPS = [
    "steps:",
    "  - id: grab",
    "    kind: retrieve_and_prompt",
    "    retrieval:",  # step-level retrieval config is LIVE and must keep loading
    "      kind: hybrid",
    "      query: foo",
    "      k: 5",
    "    prompt: prompts/p.md",
]


def test_valid_recipe_still_loads(tmp_path):
    """The live surface (step-level retrieval) must not be falsely rejected."""
    r = _write_recipe(tmp_path, _VALID_STEPS)
    spec = load_recipe(r)
    assert spec.steps[0].config["retrieval"]["kind"] == "hybrid"


def test_valid_evidence_keys_load(tmp_path):
    r = _write_recipe(tmp_path, [
        "evidence:",
        "  max_context_tokens: 120000",
        *_VALID_STEPS,
    ])
    spec = load_recipe(r)
    assert spec.evidence.max_context_tokens == 120000


@pytest.mark.parametrize("dead_key", ["retrieval", "model_policy", "feedback"])
def test_removed_top_level_key_rejected(tmp_path, dead_key):
    r = _write_recipe(tmp_path, [f"{dead_key}: {{kind: hybrid}}", *_VALID_STEPS])
    with pytest.raises(ValueError, match="were removed"):
        load_recipe(r)


def test_evidence_formatting_style_rejected(tmp_path):
    r = _write_recipe(tmp_path, [
        "evidence:",
        "  formatting_style: plain_text",
        *_VALID_STEPS,
    ])
    with pytest.raises(ValueError, match="formatting_style"):
        load_recipe(r)


def test_unknown_evidence_key_rejected(tmp_path):
    r = _write_recipe(tmp_path, [
        "evidence:",
        "  max_context_tokens: 120000",
        "  typo_key: 1",
        *_VALID_STEPS,
    ])
    with pytest.raises(ValueError, match="under 'evidence:'"):
        load_recipe(r)


# --- temperature / response_format are engine defaults, not recipe knobs ---

def test_engine_defaults_applied_when_recipe_omits_them(tmp_path):
    """A clean `llm:` block (model + max_tokens only) gets the engine-fixed
    temperature and response_format."""
    r = _write_recipe(tmp_path, _VALID_STEPS, llm_line="llm: {model: local_qwen, max_tokens: 256}")
    spec = load_recipe(r)
    assert spec.llm.temperature == 0.0
    assert spec.llm.response_format == "json_object"
    assert spec.llm.max_tokens == 256


@pytest.mark.parametrize(
    "llm_line",
    [
        "llm: {model: local_qwen, temperature: 0.7}",
        "llm: {model: local_qwen, temperature: 0.0}",  # even the engine value is rejected
    ],
)
def test_llm_temperature_rejected(tmp_path, llm_line):
    r = _write_recipe(tmp_path, _VALID_STEPS, llm_line=llm_line)
    with pytest.raises(ValueError, match="engine defaults"):
        load_recipe(r)


@pytest.mark.parametrize(
    "llm_line",
    [
        "llm: {model: local_qwen, response_format: text}",
        "llm: {model: local_qwen, response_format: json_object}",
    ],
)
def test_llm_response_format_rejected(tmp_path, llm_line):
    r = _write_recipe(tmp_path, _VALID_STEPS, llm_line=llm_line)
    with pytest.raises(ValueError, match="engine defaults"):
        load_recipe(r)


def test_no_production_recipe_declares_engine_defaults():
    """No shipped recipe declares llm.temperature/response_format."""
    import yaml

    offenders = []
    for ry in finished_recipe_files():
        llm = (yaml.safe_load(ry.read_text(encoding="utf-8")) or {}).get("llm") or {}
        present = [k for k in ("temperature", "response_format") if k in llm]
        if present:
            offenders.append((ry.name, present))
    assert not offenders, f"recipes still declaring engine-default llm keys: {offenders}"
