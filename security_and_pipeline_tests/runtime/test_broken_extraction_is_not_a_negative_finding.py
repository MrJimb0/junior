"""A run that broke must not be readable as a chart that simply said nothing.

The missing-model fix closed one door into the silent-null path and left the corridor
open. Downstream, a step that ERRORED was demoted to a warning: a recipe every one of
whose steps blew up still ended with its final helper's "nothing found" default, a
passing schema, ``ok: true``, ``n_failed: 0``, and a run reported clean. Two more ways
in were just as quiet — the commonest reason a variable comes back empty (no chart text
ever reached the model) was recorded in a receipt nobody opens, and a validation rule
that could not be LOADED was written down exactly like a rule that ran and found a
problem. No recipe in var_extraction_recipes/ declares validation.rules yet, so that
second one guards the recipes that will: a project pointed at a recipe tree with no
``_shared_validation_rules/`` folder would check nothing at all and say nothing about it.

The distinction all three turn on: a step that ERRORED never read its part of the chart,
which is a different thing from a clinical rule that read the answer and found it
implausible. Clinical rules stay diagnostics that never fail the variable (ADR 0025) —
that behaviour is deliberate and is guarded here too, because the obvious over-correction
is to start failing runs on clinical plausibility.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import (
    retrieve_and_prompt_step as retrieve_and_prompt_module,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.run_validation_rules import (
    run_recipe_validation_rules,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
    organize_per_recipe_outputs,
    record_extract_completion,
)
from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
)

RUN_ID = "20990101_000000"
REPO_ROOT = Path(__file__).resolve().parents[2]


# --- harness ----------------------------------------------------------------------


def _schema(recipes_root: Path) -> Path:
    """A schema every "nothing found" default satisfies — a null answer is a legitimate
    negative, which is exactly why a broken run looks like one."""
    path = recipes_root / "stage.schema.json"
    path.write_text(
        json.dumps({
            "type": "object",
            "properties": {"stage": {"type": ["string", "null"]}},
            "required": ["stage"],
        }),
        encoding="utf-8",
    )
    return path


def _write_transcript(
    patient_out: Path,
    recipes_root: Path,
    *,
    variable: str = "stage",
    data_final=None,
    errors: list[str] | None = None,
    steps: list[dict] | None = None,
    validation_rules: list | None = None,
) -> None:
    variable_dir = extract_output_dir(patient_out) / variable
    variable_dir.mkdir(parents=True, exist_ok=True)
    (variable_dir / "transcript.json").write_text(
        json.dumps({
            "sensitivity": "high",
            "run_id": RUN_ID,
            "patient_id": "P",
            "variable": variable,
            "recipe": {"name": variable, "version": "v1"},
            "recipe_path": str(recipes_root / variable / "v1" / "recipe.yaml"),
            "output_schema_path": str(_schema(recipes_root)),
            "validation_rules": validation_rules or [],
            "depends_on": [],
            "data_final": data_final,
            "errors": errors or [],
            "steps": steps or [],
            "total_elapsed_s": 0.1,
            "provider_config_hash": None,
            "code_lock_hash": None,
            "step_evidence": [],
            "provenance_rejected": [],
        }),
        encoding="utf-8",
    )


def _organize(patient_out: Path, recipes_root: Path, variable: str = "stage") -> dict:
    return organize_per_recipe_outputs(
        run_id=RUN_ID,
        patient_id="P",
        patient_out=patient_out,
        recipes_root=recipes_root,
        ordered_names=[variable],
        code_lock_hash=None,
    )


def _result_payload(patient_out: Path, variable: str = "stage") -> dict:
    result_file = extract_output_dir(patient_out) / variable / "result.json"
    return json.loads(result_file.read_text(encoding="utf-8"))["payload"]


def _invariants_payload(patient_out: Path, variable: str = "stage") -> dict:
    path = artifact_path("invariants", patient_root=patient_out, variable=variable)
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


def _write_rule(recipes_root: Path, name: str, body: str) -> None:
    rules_dir = recipes_root / "_shared_validation_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / f"{name}.py").write_text(body, encoding="utf-8")


def _errored(step_id: str) -> dict:
    return {"step_id": step_id, "kind": "retrieve_and_prompt", "status": "failed",
            "receipt_path": f"steps/{step_id}/receipt.json", "elapsed_s": 0.01}


def _completed(step_id: str, kind: str = "retrieve_and_prompt") -> dict:
    return {"step_id": step_id, "kind": kind, "status": "completed",
            "receipt_path": f"steps/{step_id}/receipt.json", "elapsed_s": 0.01}


# --- a variable whose steps errored -------------------------------------------------


def test_a_recipe_whose_every_step_errored_does_not_report_ok(tmp_path, monkeypatch):
    """The corridor itself. Every step blew up, the final helper emitted its default,
    the schema accepted it — and the variable was reported clean."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": None},                       # the "nothing found" default
        errors=["evidence: RuntimeError: index missing", "finalize: RuntimeError: index missing"],
        steps=[_errored("evidence"), _errored("finalize")],
    )

    results = _organize(patient_out, recipes_root)

    assert results["stage"]["ok"] is False
    assert _result_payload(patient_out)["ok"] is False


def test_a_recipe_whose_steps_errored_counts_as_a_failed_variable(tmp_path, monkeypatch):
    """n_failed is what the stage transition and the terminal both read; a broken
    variable that never reaches it leaves the run reported clean."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": None}, errors=["evidence: RuntimeError: boom"],
        steps=[_errored("evidence"), _completed("finalize", kind="python")],
    )
    results = _organize(patient_out, recipes_root)

    n_failed = record_extract_completion(
        run_root=tmp_path / "run", run_id=RUN_ID, patient_id="P",
        requested_recipes=["stage"], results=results, code_lock_hash=None,
    )
    assert n_failed == 1


def test_the_result_file_names_the_steps_that_errored(tmp_path, monkeypatch):
    """Whoever reads this dataset later has to tell a real negative from a broken run,
    and they read result.json — so the broken steps are named there, not just counted."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": None}, errors=["evidence: RuntimeError: boom"],
        steps=[_errored("evidence"), _completed("finalize", kind="python")],
    )
    _organize(patient_out, recipes_root)

    payload = _result_payload(patient_out)
    assert payload["steps_that_errored"] == ["evidence"]
    reason = " ".join(payload["errors"])
    assert "evidence" in reason
    assert "not that the chart is silent" in reason, "the error does not say what a null means"


def test_a_step_that_finished_while_reporting_a_problem_is_still_ok(tmp_path, monkeypatch):
    """The over-correction to avoid. A step that RAN and complained (an unparseable
    model response a later step recovered from) is a warning; only a step that ERRORED
    means part of the chart went unread. Fail this one and every recovered hiccup starts
    reading as a broken extraction."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"},
        errors=["evidence: parsed response was not a JSON object"],
        steps=[_completed("evidence"), _completed("finalize", kind="python")],
    )
    results = _organize(patient_out, recipes_root)

    assert results["stage"]["ok"] is True
    assert results["stage"]["warnings"] == ["evidence: parsed response was not a JSON object"]
    assert _result_payload(patient_out)["steps_that_errored"] == []


# --- ADR 0025: a clinical rule that fires is a diagnostic, not a failure -------------


def test_a_failing_clinical_rule_does_not_fail_the_variable(tmp_path, monkeypatch):
    """ADR 0025. A rule that read the answer and found it clinically implausible records
    a violation and the batch carries on — a flaky answer is not a pipeline break, and
    the operator decides. This must survive the ok change above."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_rule(recipes_root, "stage_rule_that_flags_implausible",
                "def check(results):\n    return ['stage IIA cannot follow stage IV']\n")
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"}, steps=[_completed("evidence")],
        validation_rules=["stage_rule_that_flags_implausible"],
    )

    results = _organize(patient_out, recipes_root)

    assert results["stage"]["ok"] is True, "a clinical rule firing must not fail the variable"
    invariants = _invariants_payload(patient_out)
    assert invariants["rules_failed"] == 1        # the violation IS recorded
    assert invariants["rules_did_not_run"] == 0   # it ran; it just did not like the answer


# --- a validation rule that could not be loaded never checked anything ---------------


def test_a_missing_rule_file_is_not_counted_as_a_rule_that_ran(tmp_path):
    """Point a project's recipes_root at a tree with no _shared_validation_rules/ and
    every declared rule for every patient reports 'Rule file not found'. Recorded
    identically to a rule that ran and found a problem, that reads as a checked value
    while the run reports clean."""
    result = run_recipe_validation_rules(
        recipes_root=tmp_path / "recipes_with_no_rules_folder",
        rule_names=["stage_is_plausible"],
        extraction_results={"stage": {"data": {"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"}}},
        patient_id="P", variable="stage", run_id=RUN_ID,
    )

    assert result.rules_did_not_run == 1
    assert result.rules_failed == 0, "a rule that never ran was counted as a rule that failed"
    assert result.rules_passed == 0, "a rule that never ran must never read as passing"
    [rule] = result.results
    assert rule.did_not_run is True
    assert "Did not run" in rule.message and "not checked" in rule.message


def test_a_rule_with_no_check_function_did_not_run(tmp_path):
    _write_rule(tmp_path, "stage_rule_without_check", "SOME_TABLE = {}\n")  # no check()
    result = run_recipe_validation_rules(
        recipes_root=tmp_path, rule_names=["stage_rule_without_check"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )
    assert result.rules_did_not_run == 1 and result.rules_failed == 0


def test_a_rule_that_raises_did_not_run(tmp_path):
    _write_rule(tmp_path, "stage_raises", "def check(results):\n    raise ValueError('bad rule')\n")
    result = run_recipe_validation_rules(
        recipes_root=tmp_path, rule_names=["stage_raises"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )
    assert result.rules_did_not_run == 1 and result.rules_failed == 0


def test_a_rule_that_ran_and_found_a_problem_is_still_counted_as_failed(tmp_path):
    """The counterpart — separating 'did not run' must not have cost the real signal."""
    _write_rule(tmp_path, "stage_rule_that_finds_a_problem",
                "def check(results):\n    return ['stage IIA cannot follow stage IV']\n")
    result = run_recipe_validation_rules(
        recipes_root=tmp_path, rule_names=["stage_rule_that_finds_a_problem"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )
    assert result.rules_failed == 1 and result.rules_did_not_run == 0
    assert result.results[0].did_not_run is False


def test_a_rule_that_never_ran_says_so_on_screen(tmp_path, monkeypatch):
    """Nothing reads invariants.json unprompted, so an unchecked cohort was silent."""
    import jr_pipeline.pipeline_steps.step_7_extract_variables.run_validation_rules as module

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(module, "_log", types.SimpleNamespace(
        warning=lambda event, extra_=None: events.append((event, extra_ or {}))
    ))
    run_recipe_validation_rules(
        recipes_root=tmp_path / "recipes_with_no_rules_folder",
        rule_names=["stage_is_plausible"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )

    [(event, fields)] = events
    assert event == "validation_rules_did_not_run"
    assert fields["rules"] == ["stage_is_plausible"]
    assert "NOT checked" in fields["hint"], "the note does not say the value went unchecked"


def test_the_number_of_rules_that_never_ran_reaches_the_result_file(tmp_path, monkeypatch):
    """invariants.json holds the detail; result.json is the file anyone opens."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"}, steps=[_completed("evidence")],
        validation_rules=["stage_is_plausible", "stage_matches_pathology"],  # neither file exists
    )
    _organize(patient_out, recipes_root)

    assert _result_payload(patient_out)["validation_rules_that_did_not_run"] == 2


# --- the counts are counts of RULES ---------------------------------------------------


def test_a_rule_that_reports_several_problems_counts_as_one_rule(tmp_path):
    """One rule_result is appended per PROBLEM, so a rule returning three of them counted
    as three rules — a recipe declaring two rules reported rules_checked=4, and the
    documented identity checked = passed + failed + did_not_run held only by inflating
    both sides."""
    _write_rule(tmp_path, "stage_rule_with_three_problems",
                "def check(results):\n    return ['one', 'two', 'three']\n")
    result = run_recipe_validation_rules(
        recipes_root=tmp_path,
        rule_names=["stage_rule_with_three_problems", "stage_rule_with_no_file"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )

    assert result.rules_checked == 2, "rules_checked is not the number of rules declared"
    assert result.rules_failed == 1, "one rule with three problems is not three rules"
    assert result.rules_did_not_run == 1
    assert len(result.results) == 4, "every problem must still get its own entry"


def test_the_rules_that_never_ran_are_counted_out_of_the_rules_declared(tmp_path, monkeypatch):
    """The hint is the sentence an operator acts on. Counting rule_results made its
    denominator larger than the recipe: two declared rules read as "1 of 4 never ran"."""
    import jr_pipeline.pipeline_steps.step_7_extract_variables.run_validation_rules as module

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(module, "_log", types.SimpleNamespace(
        warning=lambda event, extra_=None: events.append((event, extra_ or {}))
    ))
    _write_rule(tmp_path, "stage_rule_with_three_problems_on_screen",
                "def check(results):\n    return ['one', 'two', 'three']\n")
    run_recipe_validation_rules(
        recipes_root=tmp_path,
        rule_names=["stage_rule_with_three_problems_on_screen", "stage_rule_with_no_file"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )

    [(_, fields)] = [e for e in events if e[0] == "validation_rules_did_not_run"]
    assert "1 of 2 validation rule(s) never ran" in fields["hint"], fields["hint"]


def test_a_rule_that_raises_does_not_put_a_python_class_name_on_screen(tmp_path, monkeypatch):
    """workbench.py prints invariants.json verbatim for a person to read, so these
    messages are an operator surface: "ValueError: bad rule" is a stack-trace fragment,
    and alone among the three did-not-run messages it named nothing to do next."""
    import jr_pipeline.pipeline_steps.step_7_extract_variables.run_validation_rules as module

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(module, "_log", types.SimpleNamespace(
        warning=lambda event, extra_=None: events.append((event, extra_ or {}))
    ))
    _write_rule(tmp_path, "stage_raises_on_screen",
                "def check(results):\n    raise ValueError('bad rule')\n")
    result = run_recipe_validation_rules(
        recipes_root=tmp_path, rule_names=["stage_raises_on_screen"],
        extraction_results={}, patient_id="P", variable="stage", run_id=RUN_ID,
    )

    [rule] = result.results
    assert "ValueError" not in rule.message, "a Python exception class name reached the operator"
    assert "validation.rules" in rule.message, "the message names nothing to do next"
    assert "bad rule" in rule.message, "the rule's own words were dropped"
    # The class name still separates a rule file that would not import from a rule that
    # blew up on the data, so it stays — in the log, where a type name belongs.
    [(_, fields)] = [e for e in events if e[0] == "validation_rule_raised"]
    assert fields["error_type"] == "ValueError"


def test_the_result_file_survives_the_rules_machinery_blowing_up(tmp_path, monkeypatch):
    """The rules now run BEFORE result.json is written, so a crash inside them took the
    variable's answer and evidence pointers down with it. Nothing coerces validation.rules
    entries, so a non-name entry reaches the rule_results and pydantic rejects it —
    that must cost the run, not the answer already on disk."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    patient_out, recipes_root = tmp_path / "patient", tmp_path / "recipes"
    recipes_root.mkdir()
    _write_transcript(
        patient_out, recipes_root,
        data_final={"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"}, steps=[_completed("evidence")],
        validation_rules=[{"rule": "stage_is_plausible"}],  # not a rule name
    )

    with pytest.raises(Exception):  # noqa: B017 — the point is that it still fails loudly
        _organize(patient_out, recipes_root)

    payload = _result_payload(patient_out)
    assert payload["data"] == {"stage": "IIA", "evidence_chunk_id": "p1:clinical_note:3:0"}, "the extracted answer was lost with the crash"
    assert payload["validation_rules_that_did_not_run"] is None, \
        "nothing counted the rules, so the count must not read as zero went unchecked"


# --- the commonest empty answer ------------------------------------------------------


class _RetrieverThatFindsNothing:
    info = types.SimpleNamespace(kind="bm25", version="stub", config={})
    score_normalization = "raw"

    def query(self, corpus, *, text, k):
        return []


class _ProviderThatMustNotBeCalled:
    def chat(self, req):
        raise AssertionError("the model must not be called when there is no evidence")

    def provider_config(self):
        return {"endpoint_name": "stub"}


def test_empty_evidence_says_so_on_screen(tmp_path, monkeypatch):
    """The commonest way a variable comes back empty — the reranking filters emptied the
    candidate pool, the recipe's source_file does not exist for this patient, or no
    chunk's text resolved. Its two neighbours in the same class (a hybrid retriever
    member failing, a direct_parquet column not being found) both log and reach the
    terminal as a note; this one set a parse_error into a receipt and said nothing, so
    the answer read as "the chart does not say"."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(retrieve_and_prompt_module, "build_retriever",
                        lambda *a, **k: _RetrieverThatFindsNothing())
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(retrieve_and_prompt_module, "_log", types.SimpleNamespace(
        warning=lambda event, extra_=None: events.append((event, extra_ or {}))
    ))

    recipe = load_recipe(
        next((REPO_ROOT / "var_extraction_recipes").rglob("genetics_germline_v1_recipe.yaml"))
    )
    recipe.steps[0].config.pop("also_include_tables", None)  # no builder rows either
    result = retrieve_and_prompt_module.RetrieveAndPromptHandler().execute(StepContext(
        recipe=recipe,
        step=recipe.steps[0],
        patient_id="Test_Patient",
        corpus=types.SimpleNamespace(patient_root=tmp_path / "pt", row_for=lambda _cid: None),
        provider=_ProviderThatMustNotBeCalled(),
        provider_config_hash=None,
        llm_cache=None,
        run_id=RUN_ID,
        code_lock_hash=None,
    ))

    assert "empty_evidence" in (result.error or "")     # unchanged: still recorded
    emitted = [fields for event, fields in events if event == "empty_evidence"]
    assert emitted, "no empty_evidence event was logged"
    [fields] = emitted
    assert fields["variable"] == recipe.name and fields["step_id"] == recipe.steps[0].id
    assert fields["n_candidates_retrieved"] == 0
    hint = fields["hint"]
    assert "we never looked" in hint
    assert "source_file" in hint, "the note does not name what to check"
