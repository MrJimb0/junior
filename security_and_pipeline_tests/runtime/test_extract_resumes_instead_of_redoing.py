"""`junior extract` resumes: it does the work that is missing or failed, and no more.

It used to recompute everything, while the CLI told people "re-running extract retries
only what failed". One real run showed what that costs — a finished patient
re-extracted from scratch, twenty minutes, every value already on disk and correct.

The skip is per VARIABLE per patient, which is the narrowest unit the system can stand
behind: a variable is what result.json records ok for, and the steps inside one share
state and run in order, so half a variable is not a resumable thing. A variable is only
skipped when its recorded result says it succeeded AND was produced under the same
code, recipe, schema and helper hashes this run would use now.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from jr_pipeline.pipeline_steps.step_7_extract_variables import extract as extract_module
from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import (
    _settings_a_result_was_made_under,
    _why_it_failed_last_time,
    _work_already_done,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
)

SETTINGS = {
    "code_lock_hash": "sha256:" + "c" * 64, "recipe_hash": "sha256:recipe",
    "schema_hash": "sha256:schema", "python_helpers_hash": "sha256:helpers",
}


def _a_result(tmp_path: Path, variable: str, *, ok: bool, **produced_by) -> Path:
    d = tmp_path / "extract" / variable
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({
        "payload": {"ok": ok, "variable": variable, "data": {variable: "a value"}},
        "produced_by": {**SETTINGS, **produced_by},
    }), encoding="utf-8")
    return d / "result.json"


def test_successful_work_is_skipped(tmp_path):
    _a_result(tmp_path, "stage", ok=True)

    kept = _work_already_done(tmp_path, "stage", SETTINGS)

    assert kept is not None
    # Handed back with its data, because a variable that depends on this one reads the
    # value from there. Skipping without it would look like a variable that ran empty.
    assert kept["data"] == {"stage": "a value"}


def test_failed_work_is_retried(tmp_path):
    _a_result(tmp_path, "stage", ok=False)

    assert _work_already_done(tmp_path, "stage", SETTINGS) is None
    assert _why_it_failed_last_time(tmp_path, "stage") is not None


def test_missing_work_is_done(tmp_path):
    assert _work_already_done(tmp_path, "never_extracted", SETTINGS) is None
    # Never attempted is not the same as attempted and failed; the display says so.
    assert _why_it_failed_last_time(tmp_path, "never_extracted") is None


def test_a_patient_with_some_of_each_retries_only_what_needs_it(tmp_path):
    _a_result(tmp_path, "date_of_birth", ok=True)
    _a_result(tmp_path, "stage", ok=False)

    assert _work_already_done(tmp_path, "date_of_birth", SETTINGS) is not None
    assert _work_already_done(tmp_path, "stage", SETTINGS) is None
    assert _work_already_done(tmp_path, "date_of_diagnosis", SETTINGS) is None


def test_a_changed_recipe_does_not_let_stale_success_be_reused(tmp_path):
    """The CLI refuses a recipe edit inside a run before extraction is reached. This is
    the second line: if anything did get past, a result made by a different recipe is
    not evidence that this recipe's work is done."""
    _a_result(tmp_path, "stage", ok=True)

    for changed in ("recipe_hash", "schema_hash", "python_helpers_hash", "code_lock_hash"):
        assert _work_already_done(
            tmp_path, "stage", {**SETTINGS, changed: "sha256:something-else"}
        ) is None, f"a result made under a different {changed} was reused"


def test_a_result_that_cannot_say_what_made_it_is_redone(tmp_path):
    """Runs sealed before those hashes were recorded re-extract once. That is the right
    way round: the alternative is trusting an answer nothing can vouch for."""
    d = tmp_path / "extract" / "stage"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({
        "payload": {"ok": True, "data": {"stage": "x"}},
        "produced_by": {"run_id": "old"},
    }), encoding="utf-8")

    assert _work_already_done(tmp_path, "stage", SETTINGS) is None


def test_an_unreadable_result_is_redone_not_trusted(tmp_path):
    d = tmp_path / "extract" / "stage"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{ truncated", encoding="utf-8")

    assert _work_already_done(tmp_path, "stage", SETTINGS) is None


def test_the_skip_costs_no_model_load(tmp_path):
    """Deciding NOT to run a variable must not build a provider. If it did, resuming a
    finished patient would cost the same model load as extracting it, and on a laptop
    that load is most of the wall clock."""
    import inspect

    source = inspect.getsource(_settings_a_result_was_made_under)

    assert "provider" not in source.replace("provider_config_hash", "").replace(
        "building a provider", "").replace("provider_config_hash", "")
    assert set(_settings_a_result_was_made_under(
        code_lock_hash="c", per_recipe={"recipe_hash": "r"},
    )) == {"code_lock_hash", "recipe_hash", "schema_hash", "python_helpers_hash"}


# --- end to end: a skipped variable does no work at all ----------------------------

RUN_ID, PATIENT = "20990101_000000", "Test_Patient"


class _CacheThatIsNeverUsed:
    def close(self):
        pass


class _AStepThatCountsHowOftenItRuns:
    times_run = 0

    def execute(self, ctx):
        from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (  # noqa: E501
            StepResult,
        )

        type(self).times_run += 1
        # A structured evidence pointer: pipeline-authored and self-verifying, so it is
        # grounded without a retrieval having shown anything. A bare chunk id would be
        # rejected as ungrounded here, the value would fail, and a failed value is
        # correctly not skippable — which would test the wrong thing.
        return StepResult(
            data={"stage": "IIA",
                  "evidence_chunk_id":
                      f"patient:{PATIENT}:structured:filehash:row:rowhash:column:stage"},
            receipt_payload={},
        )


def _a_recipe(recipes_root):
    version_dir = recipes_root / "basic" / "stage" / "v1"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "stage_v1_output_schema.json").write_text(json.dumps({
        "type": "object",
        "properties": {"stage": {"type": ["string", "null"]},
                       "evidence_chunk_id": {"type": ["string", "null"]}},
        "required": ["stage"],
    }), encoding="utf-8")
    (version_dir / "stage_v1_recipe.yaml").write_text(
        "name: stage\nversion: v1\ncollection: basic\narchetype: point_fact\n"
        "output_schema: stage_v1_output_schema.json\n"
        "llm:\n  model: local_qwen\ndepends_on: []\n"
        "steps:\n  - id: read_it\n    kind: python\n"
        "    module: stage_v1_python_helper.read_it\n",
        encoding="utf-8",
    )


def _a_sealed_bundle(run_root):
    """A run records its per-recipe hashes when it is sealed, and resume compares
    against them. An UNSEALED run has none, and now redoes everything rather than
    trusting a result it cannot identify — so a fixture that skips sealing is testing
    the refusal, not the resume."""
    code = run_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "hashes.json").write_text(json.dumps({"payload": {"per_recipe": {
        "stage/v1": {"recipe_hash": "sha256:" + "a" * 64, "schema_hash": "sha256:" + "b" * 64,
                     "python_helpers_hash": "sha256:" + "d" * 64},
    }}}), encoding="utf-8")


def _extract(tmp_path, monkeypatch, *, force=False, seen=None):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(extract_module, "build_step", lambda kind: _AStepThatCountsHowOftenItRuns())
    recipes_root = tmp_path / "recipes"
    _a_recipe(recipes_root)
    _a_sealed_bundle(phi_patient_run_dir(RUN_ID, PATIENT).parent.parent)
    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    patient_out.mkdir(parents=True, exist_ok=True)
    if not (patient_out / "chunk_index.parquet").exists():
        pl.DataFrame({"chunk_id": ["c1"]}).write_parquet(patient_out / "chunk_index.parquet")
    extract_module._run_extract_one_body(
        cfg={"run_id": RUN_ID, "recipes_root": str(recipes_root), "recipes": ["stage"]},
        patient_id=PATIENT,
        llm_cache=_CacheThatIsNeverUsed(),
        code_lock_hash="sha256:" + "c" * 64,
        force=force,
        on_variable=(lambda name, state, secs: seen.append((name, state))) if seen is not None else None,
    )
    return extract_output_dir(patient_out) / "stage"


def test_a_rerun_does_no_work_for_a_variable_that_succeeded(tmp_path, monkeypatch):
    """The claim the CLI was making and the code was not keeping. Counted at the step,
    so it cannot pass by the model cache quietly answering instead."""
    _AStepThatCountsHowOftenItRuns.times_run = 0

    _extract(tmp_path, monkeypatch)
    after_first = _AStepThatCountsHowOftenItRuns.times_run
    assert after_first == 1

    seen: list[tuple[str, str]] = []
    variable_dir = _extract(tmp_path, monkeypatch, seen=seen)

    assert _AStepThatCountsHowOftenItRuns.times_run == after_first, (
        "the variable was extracted again even though it had already succeeded"
    )
    assert ("stage", "already complete") in seen
    # And the answer is still there afterwards -- skipping must not mean discarding.
    assert json.loads((variable_dir / "result.json").read_text())["payload"]["ok"] is True


def test_force_does_the_work_again(tmp_path, monkeypatch):
    _AStepThatCountsHowOftenItRuns.times_run = 0
    _extract(tmp_path, monkeypatch)

    _extract(tmp_path, monkeypatch, force=True)

    assert _AStepThatCountsHowOftenItRuns.times_run == 2


def test_resume_does_not_depend_on_the_model_cache(tmp_path, monkeypatch):
    """The cache is an optimisation. Correct resume is decided by recorded extraction
    state, so it holds with no cache at all -- which is what `_CacheThatIsNeverUsed` is."""
    _AStepThatCountsHowOftenItRuns.times_run = 0
    _extract(tmp_path, monkeypatch)

    _extract(tmp_path, monkeypatch)

    assert _AStepThatCountsHowOftenItRuns.times_run == 1


def test_a_variable_is_not_called_done_and_then_failed(tmp_path, monkeypatch):
    """Whether a variable succeeded is decided by step 8, against its schema and its
    provenance — a recipe whose steps all ran cleanly can still fail there. Reporting
    the outcome while the steps finish means reporting it before it is known, and a
    display that says "done" and then "failed" about the same variable, two lines
    apart, is worse than one that waits."""
    _AStepThatCountsHowOftenItRuns.times_run = 0
    seen: list[tuple[str, str]] = []

    _extract(tmp_path, monkeypatch, seen=seen)

    terminal = [state for name, state in seen if state in ("done", "failed")]
    assert len(terminal) == 1, f"a variable reported more than one outcome: {seen}"


def test_each_variable_finishes_before_the_next_one_starts(tmp_path, monkeypatch):
    """The display has to look like the work: running, done, running, done.

    Organizing every variable after the loop meant the outcome was only knowable at the
    end, so the display printed every start and then every finish — three "running"
    lines, then three "done" lines, with the timings detached from the variables they
    belonged to. Organizing each variable as it finishes is what makes the sequence
    honest, and it is safe because the loop already runs in dependency order."""
    _AStepThatCountsHowOftenItRuns.times_run = 0
    seen: list[tuple[str, str]] = []

    _extract(tmp_path, monkeypatch, seen=seen)

    for index, (name, state) in enumerate(seen):
        if state in ("running", "retrying"):
            assert index + 1 < len(seen), f"{name} started and never resolved"
            next_name, next_state = seen[index + 1]
            assert next_name == name and next_state in ("done", "failed"), (
                f"{name} was left open while {next_name} started: {seen}"
            )


def test_an_unsealed_run_redoes_everything(tmp_path, monkeypatch):
    """The skip compares against the hashes a run records when it is sealed. With none
    recorded there is nothing to compare, and that has to mean redo.

    It meant the opposite: an absent expected value was skipped over, so all four
    comparisons were vacuous and any ok result was kept. That fired hardest exactly
    where least was known — an unsealed run, or a recipe version bumped past the sealed
    index — and it is how `compare-models` came to report every model with the first
    model's answers."""
    _AStepThatCountsHowOftenItRuns.times_run = 0
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(extract_module, "build_step", lambda kind: _AStepThatCountsHowOftenItRuns())
    recipes_root = tmp_path / "recipes"
    _a_recipe(recipes_root)
    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    patient_out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"chunk_id": ["c1"]}).write_parquet(patient_out / "chunk_index.parquet")

    def _run_unsealed():
        extract_module._run_extract_one_body(
            cfg={"run_id": RUN_ID, "recipes_root": str(recipes_root), "recipes": ["stage"]},
            patient_id=PATIENT, llm_cache=_CacheThatIsNeverUsed(),
        )

    _run_unsealed()
    _run_unsealed()

    assert _AStepThatCountsHowOftenItRuns.times_run == 2, (
        "an unsealed run skipped work it had no way to identify"
    )


def test_compare_models_forces_re_extraction():
    """Its scratch run id is derived from the source run, so it is the same id every
    time, and it re-runs the same variable deliberately under a different endpoint.
    Resume would hand back the previous model's answer — and _read_fingerprint would
    then report it with the previous model's receipts, so the comparison looks complete
    and every row is the first model."""
    import inspect

    from jr_pipeline.evaluating_pipeline_performance import compare_llm_models

    source = inspect.getsource(compare_llm_models.compare_models)

    assert "force=True" in source


def test_a_rebuilt_corpus_invalidates_a_finished_answer(tmp_path, monkeypatch):
    """The scenario none of the compared hashes can see: a chart was wrong, the operator
    fixes it and re-runs ingest/embed/index --force under the SAME run id. Every hash
    the skip compares identifies the CODE, and all of them come from the run's sealed
    index, so none of them move. Without this, extract skips every ok variable, never
    reads the new document, and reports the run completed."""
    import os
    import time

    _AStepThatCountsHowOftenItRuns.times_run = 0
    _extract(tmp_path, monkeypatch)
    assert _AStepThatCountsHowOftenItRuns.times_run == 1

    _extract(tmp_path, monkeypatch)
    assert _AStepThatCountsHowOftenItRuns.times_run == 1, "sanity: an unchanged corpus resumes"

    # embed rewrites the chunk index; only its timestamp is observable from here.
    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    later = time.time() + 10
    os.utime(patient_out / "chunk_index.parquet", (later, later))

    _extract(tmp_path, monkeypatch)

    assert _AStepThatCountsHowOftenItRuns.times_run == 2, (
        "the variable was skipped even though its passages had been rebuilt"
    )


def test_an_inherited_corpus_is_not_mistaken_for_a_changed_one(tmp_path, monkeypatch):
    """`extract --new-run` hard links the corpus forward, so the artifacts keep the
    source run's timestamps. That must read as the same corpus, or every inherited run
    would redo work it just inherited in order to avoid."""
    _AStepThatCountsHowOftenItRuns.times_run = 0
    _extract(tmp_path, monkeypatch)

    patient_out = phi_patient_run_dir(RUN_ID, PATIENT)
    result_file = extract_output_dir(patient_out) / "stage" / "result.json"

    assert not extract_module._the_corpus_changed_after(result_file, patient_out)


def test_a_reingested_structured_table_invalidates_a_finished_answer(tmp_path):
    """The watched set is everything extraction reads, not the three embed/index files.
    `direct_parquet` retrieval reads structured/<table>.parquet straight off disk — for
    a patient with a structured date it decides the whole variable — so a re-ingested
    demographics table must not leave 'already complete' answers that never read it."""
    import os
    import time

    patient_out = tmp_path / "patient"
    (patient_out / "structured").mkdir(parents=True)
    table = patient_out / "structured" / "demographics.parquet"
    table.write_bytes(b"rows")
    result_file = _a_result(patient_out, "stage", ok=True)

    assert not extract_module._the_corpus_changed_after(result_file, patient_out)

    later = time.time() + 10
    os.utime(table, (later, later))

    assert extract_module._the_corpus_changed_after(result_file, patient_out), (
        "a re-ingested structured table did not invalidate the finished answer"
    )


def test_a_directory_timestamp_alone_does_not_invalidate(tmp_path):
    """Hard-linking an inherited corpus creates FRESH directories holding files that
    keep the source inode's timestamps. The judgement is the newest file beneath an
    entry, never the directory's own mtime — or every inherited run would redo the
    work it just inherited in order to avoid."""
    import os
    import time

    patient_out = tmp_path / "patient"
    (patient_out / "structured").mkdir(parents=True)
    (patient_out / "structured" / "demographics.parquet").write_bytes(b"rows")
    result_file = _a_result(patient_out, "stage", ok=True)

    later = time.time() + 10
    os.utime(patient_out / "structured", (later, later))

    assert not extract_module._the_corpus_changed_after(result_file, patient_out)


def test_a_rewritten_source_snapshot_alone_does_not_invalidate(tmp_path):
    """Ingest rewrites source_snapshot.json even when nothing changed, so watching it
    would turn every no-op re-ingest into a full cohort redo. When a chart really
    changed, the structured tables move too, and those are watched."""
    import os
    import time

    patient_out = tmp_path / "patient"
    patient_out.mkdir(parents=True)
    snapshot = patient_out / "source_snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    result_file = _a_result(patient_out, "stage", ok=True)

    later = time.time() + 10
    os.utime(snapshot, (later, later))

    assert not extract_module._the_corpus_changed_after(result_file, patient_out)
