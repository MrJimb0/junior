"""A step that raised must keep saying so, whatever its `stop_if` thinks.

`stop_if` asks "is this answer good enough to stop early?" — but a step that raised
produced no answer for it to look at, and the runner handed it an empty dict instead.
A condition phrased as "nothing found yet" is then true, the recipe stops, and the same
summary dict the runner already appended has its status rewritten from failed to
stopped. "failed" is the one word step 8 scans for to decide the variable was not fully
read, so the erroring step vanishes from steps_that_errored, the last step that did
return something supplies the answer, its "nothing found" default passes the schema, and
the variable is reported ok with a null value — the run says the chart is silent about a
question it never managed to ask.

Every stop_if in the recipe tree today opens with `data is not none and ...`, so this
does not fire on the shipped recipes. It fires on the next one written without that
guard, which is exactly the kind of authoring detail nobody would expect to decide
whether a broken extraction is reported as broken.
"""
from __future__ import annotations

import json

import polars as pl

import jr_pipeline.pipeline_steps.step_7_extract_variables.extract as extract_module
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepResult,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
)

RUN_ID = "20990101_000000"
PATIENT = "Test_Patient"

# True the moment `data` is empty, which is what a failed step is handed.
STOP_IF_WITHOUT_A_NONE_GUARD = "not data"


class _StepsWhereTheSecondOneBlowsUp:
    """The first step returns the recipe's "nothing found" default (a shape the schema
    accepts); the second — the one that would have read the chart — raises."""

    def execute(self, ctx):
        if ctx.step.id == "read_the_table":
            return StepResult(data={"stage": None}, receipt_payload={})
        raise RuntimeError("the chart store never opened")


def _a_recipe_whose_second_step_stops_on_an_empty_answer(recipes_root):
    version_dir = recipes_root / "stage" / "v1"
    version_dir.mkdir(parents=True)
    (version_dir / "stage_v1_output_schema.json").write_text(
        json.dumps({
            "type": "object",
            "properties": {"stage": {"type": ["string", "null"]}},
            "required": ["stage"],
        }),
        encoding="utf-8",
    )
    (version_dir / "stage_v1_recipe.yaml").write_text(
        "name: stage\n"
        "version: v1\n"
        "collection: basic\n"
        "archetype: point_fact\n"
        "output_schema: stage_v1_output_schema.json\n"
        "llm:\n"
        "  model: local_qwen\n"
        "depends_on: []\n"
        "steps:\n"
        "  - id: read_the_table\n"
        "    kind: python\n"
        "    module: stage_v1_python_helper.read_the_table\n"
        "  - id: read_the_notes\n"
        "    kind: python\n"
        "    module: stage_v1_python_helper.read_the_notes\n"
        f"    stop_if: {STOP_IF_WITHOUT_A_NONE_GUARD!r}\n",
        encoding="utf-8",
    )


class _CacheThatIsNeverUsed:
    def close(self):
        pass


def _extract_one_patient(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(extract_module, "build_step",
                        lambda kind: _StepsWhereTheSecondOneBlowsUp())
    recipes_root = tmp_path / "recipes"
    _a_recipe_whose_second_step_stops_on_an_empty_answer(recipes_root)

    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    patient_out.mkdir(parents=True)
    # The chart store opens this on the way in; the steps here never read from it.
    pl.DataFrame({"chunk_id": ["c1"]}).write_parquet(patient_out / "chunk_index.parquet")

    extract_module._run_extract_one_body(
        cfg={"run_id": RUN_ID, "recipes_root": str(recipes_root), "recipes": ["stage"]},
        patient_id=PATIENT,
        llm_cache=_CacheThatIsNeverUsed(),
    )
    return extract_output_dir(patient_out) / "stage"


def test_a_stop_if_does_not_rewrite_a_failed_step_as_stopped(tmp_path, monkeypatch):
    """The status field itself: stopped means the recipe had what it came for, failed
    means a step blew up. Overwrite one with the other and the transcript no longer
    records that anything went wrong."""
    variable_dir = _extract_one_patient(tmp_path, monkeypatch)

    transcript = json.loads((variable_dir / "transcript.json").read_text(encoding="utf-8"))
    recorded = {step["step_id"]: step["status"] for step in transcript["steps"]}
    assert recorded["read_the_notes"] == "failed"


def test_a_variable_whose_step_blew_up_under_a_stop_if_is_not_reported_ok(
    tmp_path, monkeypatch
):
    """What the rewritten status costs downstream: the erroring step drops out of
    steps_that_errored, the earlier step's "nothing found" default passes the schema, and
    a question that was never asked comes back as an answered null."""
    variable_dir = _extract_one_patient(tmp_path, monkeypatch)

    payload = json.loads((variable_dir / "result.json").read_text(encoding="utf-8"))["payload"]
    assert payload["steps_that_errored"] == ["read_the_notes"]
    assert payload["ok"] is False


class _StepsThatBothSucceed:
    """The first step finds the answer; the second would overwrite it if it ran."""

    def execute(self, ctx):
        if ctx.step.id == "read_the_table":
            return StepResult(data={"stage": "IIA"}, receipt_payload={})
        return StepResult(data={"stage": "IV"}, receipt_payload={})


def test_a_satisfied_stop_if_still_stops_the_recipe(tmp_path, monkeypatch):
    """The counterpart, and the reason the guard above is not just `if False`.

    Narrowing stop_if to skip failed steps must not have cost it its job. Without this,
    the whole early exit could be deleted and every stop_if test in the repo would still
    pass — and a recipe meant to stop after a cheap gate would run its later steps
    anyway, paying for model calls it does not need and letting a later step's answer
    overwrite the one the recipe already had."""
    monkeypatch.setattr(extract_module, "build_step", lambda kind: _StepsThatBothSucceed())
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    recipes_root = tmp_path / "recipes"
    _a_recipe_whose_second_step_stops_on_an_empty_answer(recipes_root)
    # Move the early exit onto the FIRST step, where it has something to stop: satisfied
    # once that step has produced a stage, so the second must never run.
    recipe_file = recipes_root / "stage" / "v1" / "stage_v1_recipe.yaml"
    recipe_file.write_text(
        recipe_file.read_text(encoding="utf-8")
        .replace(f"    stop_if: {STOP_IF_WITHOUT_A_NONE_GUARD!r}\n", "")
        .replace(
            "  - id: read_the_table\n    kind: python\n"
            "    module: stage_v1_python_helper.read_the_table\n",
            "  - id: read_the_table\n    kind: python\n"
            "    module: stage_v1_python_helper.read_the_table\n"
            "    stop_if: \"data is not none and data.get('stage') is not none\"\n",
        ),
        encoding="utf-8",
    )

    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    patient_out.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["c1"]}).write_parquet(patient_out / "chunk_index.parquet")
    extract_module._run_extract_one_body(
        cfg={"run_id": RUN_ID, "recipes_root": str(recipes_root), "recipes": ["stage"]},
        patient_id=PATIENT,
        llm_cache=_CacheThatIsNeverUsed(),
    )
    variable_dir = extract_output_dir(patient_out) / "stage"

    transcript = json.loads((variable_dir / "transcript.json").read_text(encoding="utf-8"))
    recorded = {step["step_id"]: step["status"] for step in transcript["steps"]}
    assert recorded["read_the_table"] == "stopped", (
        "the recipe did not stop after the step whose stop_if was satisfied"
    )
    assert recorded["read_the_notes"] == "skipped", "the recipe ran on past its early exit"
    payload = json.loads((variable_dir / "result.json").read_text(encoding="utf-8"))["payload"]
    assert payload["data"]["stage"] == "IIA", "a step past the early exit overwrote the answer"
