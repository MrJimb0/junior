"""Step 8: per-recipe output organization.

A recipe is one extraction task (e.g. "extract the stage", "extract the receptors").
This module reads each recipe's ``transcript.json`` (the run record step 7 wrote) and
produces the patient-identifiable (CONTAINS_PHI) output files: ``result.json`` (the
extracted answer), the per-recipe validation ``invariants.json`` (rule-check results),
and the evidence files (which source text backed the answer). It is pure (no model
calls) and safe to re-run -- re-running over the same transcripts rewrites the same
files and re-derives the same results, spending no model tokens.

Returns ``{variable: {data, ok, errors, warnings, steps, steps_that_errored}}`` so the
caller can run the cross-variable invariants (consistency checks spanning several
variables) plus run accounting over every variable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.run_validation_rules import (
    run_recipe_validation_rules,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.validate_recipe_output_schema import (
    validate_output,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.extraction_outcome import (
    emit_extraction_outcome,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.write_evidence_artifacts import (
    write_evidence_artifacts,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_selection_trace,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)
from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path, write_artifact
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    extract_output_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_log = get_logger("organize_output")


def record_extract_completion(
    *,
    run_root: Path,
    run_id: str,
    patient_id: str,
    requested_recipes: list[str],
    results: dict[str, dict[str, Any]],
    code_lock_hash: str | None,
) -> int:
    """Run accounting: count the variables that came back not-ok and record
    the single running->completed transition for the extract step. Returns n_failed.
    A not-ok variable keeps the step from looking cleanly 'completed' downstream (the
    reason string carries the count); a hard crash instead flips the step to 'failed'
    in run_extract_one's outer wrapper, which owns that transition plus the
    model-response cache (llm_cache)."""
    n_failed = sum(1 for v in results.values() if not v.get("ok", False))
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="extract"),
        from_state="running",
        to_state="completed",
        reason=(
            f"extract recipes={requested_recipes} ({n_failed} with errors)"
            if n_failed else f"extract recipes={requested_recipes}"
        ),
        step_context="extract",
        code_lock_hash=code_lock_hash,
    )
    return n_failed


def organize_per_recipe_outputs(
    *,
    run_id: str,
    patient_id: str,
    patient_out: Path,
    recipes_root: Path,
    ordered_names: list[str],
    code_lock_hash: str | None,
    results_so_far: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Organize every recipe that ran during this invocation (i.e. has a transcript),
    in dependency order (a recipe that depends on another runs after it). A recipe
    with no transcript did not run this invocation and is skipped; any result.json a
    prior run wrote for it is left untouched.

    ``results_so_far`` lets a caller organize one variable at a time while keeping the
    accumulated context the validation rules read. Extraction does that so a variable's
    outcome is known the moment it finishes rather than after every variable has, which
    is what lets the display report each one as it happens instead of printing every
    start and then every finish."""
    results: dict[str, dict[str, Any]] = dict(results_so_far or {})
    for name in ordered_names:
        transcript_path = extract_output_dir(patient_out) / name / "transcript.json"
        if not transcript_path.is_file():
            continue  # didn't run this invocation -- nothing to (re)organize
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        results[name] = _organize_one(
            run_id=run_id,
            patient_id=patient_id,
            patient_out=patient_out,
            recipes_root=recipes_root,
            transcript=transcript,
            # Where this transcript actually is right now, which is what its step
            # receipts are recorded relative to. A run that was written on a cluster and
            # copied to a laptop is read here from its new home.
            variable_dir=transcript_path.parent,
            results_so_far=results,
            code_lock_hash=code_lock_hash,
        )
    return results


def _output_schema_path(transcript: dict[str, Any], recipes_root: Path) -> Path:
    """This variable's output schema, found from the machine reading the run now.

    A run records the schema relative to the recipe library, so it is resolved against
    the library this machine has. Runs written before that are absolute, and those are
    honoured as they stand — on the machine that extracted them they are still right,
    and re-anchoring a path that already points at a file would be guessing."""
    recorded = Path(transcript["output_schema_path"])
    return recorded if recorded.is_absolute() else Path(recipes_root) / recorded


def _organize_one(
    *,
    run_id: str,
    patient_id: str,
    patient_out: Path,
    recipes_root: Path,
    transcript: dict[str, Any],
    variable_dir: Path,
    results_so_far: dict[str, dict[str, Any]],
    code_lock_hash: str | None,
) -> dict[str, Any]:
    variable = transcript["variable"]
    data_final = transcript.get("data_final")
    step_errors = list(transcript.get("errors") or [])
    steps = list(transcript.get("steps") or [])

    # A step that ERRORED never read the part of the chart it was there to read. That is
    # a different thing from a clinical validation rule finding the answer implausible
    # (ADR 0025: those are diagnostics and never fail the variable) and a different thing
    # again from a step that finished while reporting a problem a later step recovered
    # from. Without the distinction, a recipe whose every step errored still ends with
    # the recipe's final helper emitting its "nothing found" default, a passing schema,
    # and a variable reported ok -- "we never looked" reading as "not in the chart".
    steps_that_errored = [
        str(recorded_step.get("step_id"))
        for recorded_step in steps
        if recorded_step.get("status") == "failed"
    ]

    validation = validate_output(
        data_final, _output_schema_path(transcript, recipes_root)
    )
    # every field holding an extracted clinical value must point back to the source
    # text span it came from (its evidence_chunk_id); record any that don't. An empty
    # list => every value is traceable to its source (the toy/end-to-end contract).
    unprovenanced = find_unprovenanced_value_paths(data_final)

    # What "ok" means: every step ran, the FINAL extracted data matches the recipe's
    # output schema, and every extracted value can be traced to the span it came from.
    # An error a step reported while still finishing is a warning, not a failure, so a
    # recovered intermediate error does not permanently fail the variable.
    #
    # Provenance is checked HERE rather than left to each recipe's schema. The extract
    # step nulls an evidence pointer it finds ungrounded, so whether that nulling failed
    # the variable used to depend on whether the recipe author happened to write the
    # field as required and non-empty. Recipes disagreed: breast_clinical_stage_ajcc7
    # declares evidence_chunk_id `minLength: 1` and required, so a nulled pointer failed
    # validation; stage_v1 declares it `["string", "null"]` and optional, so the same
    # nulling was schema-legal and the value shipped ok=true with an empty warnings list
    # and a fabricated citation behind it. A guard whose enforcement varies per recipe is
    # not a guard, so `ok` now answers the question directly.
    ok = validation.ok and not steps_that_errored and not unprovenanced
    warnings = step_errors
    errors_out = list(validation.errors) if not validation.ok else []
    if steps_that_errored:
        errors_out.append(
            "these steps errored, so this variable was not fully read: "
            + ", ".join(steps_that_errored)
            + ". An empty answer here means the extraction broke, not that the chart is "
            "silent; each step's reason is in this result's warnings."
        )
    if unprovenanced:
        # Say which of the two it was. "Cited nothing" is a model that declined to
        # answer the provenance question; "cited a passage it was never shown" is a
        # model that answered it by inventing one. They call for different responses,
        # and by this point the pointer is null in both cases, so only the transcript
        # can still tell them apart.
        # A rejection names the POINTER it nulled ("data.clinical_tnm.evidence_chunk_id");
        # an unprovenanced path names the OBJECT left without one ("data.clinical_tnm").
        # So the two are matched by containment, never by equality.
        fabricated = [str(entry.get("path")) for entry in (transcript.get("provenance_rejected") or [])]

        def _citation_was_invented(value_path: str) -> bool:
            return any(
                rejected == value_path or rejected.startswith(value_path + ".")
                for rejected in fabricated
            )

        errors_out.append(
            "these values carry no source span, so they are unsupported claims: "
            + ", ".join(
                path + (" (its citation was rejected as ungrounded)"
                        if _citation_was_invented(path) else "")
                for path in unprovenanced
            )
            + ". The value itself matched the recipe's schema; what is missing is the "
            "evidence for it."
        )

    organized = {
        "data": data_final,
        "ok": ok,
        "errors": errors_out,
        "warnings": warnings,
        "steps": steps,
        "steps_that_errored": steps_that_errored,
    }

    # per-recipe validation rules, run over the results gathered so far (pass the real
    # recipes_root directory, never re-derive it from a single recipe's yaml path).
    # Run before result.json is written so its count of rules that never ran can go into
    # that file: invariants.json holds the detail, but result.json is what gets read.
    # The write sits in a finally because result.json is the only place this variable's
    # answer and evidence pointers survive; a rules machinery that itself blows up (a
    # validation.rules entry that is not a rule name, say) must not also take those with
    # it. The failure still propagates once the file is on disk.
    accumulated = {**results_so_far, variable: organized}
    validation_result = None
    try:
        validation_result = run_recipe_validation_rules(
            recipes_root=recipes_root,
            rule_names=list(transcript.get("validation_rules") or []),
            extraction_results=accumulated,
            patient_id=patient_id,
            variable=variable,
            run_id=run_id,
        )
    finally:
        result_payload = {
            "variable": variable,
            "patient_id": patient_id,
            "ok": ok,
            "data": data_final,
            "errors": errors_out,
            "warnings": warnings,
            "unprovenanced_value_paths": unprovenanced,
            # Evidence pointers the extract step rejected as fabricated (cited a passage
            # the model was never shown); nulled there, so the values also appear in
            # unprovenanced_value_paths above. Surfaced so the reviewer can tell a value
            # that cited nothing from one whose citation was made up.
            "provenance_rejected": transcript.get("provenance_rejected") or [],
            "steps": steps,
            # Named here, not just counted, so someone reading this dataset later can tell
            # a real negative from a broken extraction without opening the transcript.
            "steps_that_errored": steps_that_errored,
            # What happened to the answer on its way through the recipe: carried
            # through so a row reading ok=TRUE with nothing in it can be told from a
            # chart that is genuinely silent. Recorded in extract.py, which says why.
            "answer_came_from_step": transcript.get("answer_came_from_step"),
            "content_found_by_step": transcript.get("content_found_by_step") or {},
            # Validation rules that never ran (no rule file, no check() function, or the
            # rule raised). A rule that did not run is not a rule that passed -- nothing
            # checked this value -- and per ADR 0025 that is recorded, not fatal. null,
            # never 0, when the rules machinery itself failed: nothing counted the rules,
            # and "0 rules went unchecked" is a wrong number to hand an operator.
            "validation_rules_that_did_not_run": (
                validation_result.rules_did_not_run if validation_result else None
            ),
            "total_elapsed_s": transcript.get("total_elapsed_s"),
            "recipe": transcript.get("recipe"),
        }
        env = envelope_for(
            artifact_type="variable_result",
            sensitivity="medium",
            stream="data",
            run_id=run_id,
            step="extract",
            patient_id=patient_id,
            variable=variable,
            payload=result_payload,
            code_lock_hash=code_lock_hash,
            provider_config_hash=transcript.get("provider_config_hash"),
            recipe_hash=transcript.get("recipe_hash"),
            schema_hash=transcript.get("schema_hash"),
            python_helpers_hash=transcript.get("python_helpers_hash"),
        )
        write_artifact(env, patient_root=patient_out)  # -> extract/<variable>/result.json

    inv_env = envelope_for(
        artifact_type="invariants",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="extract",
        patient_id=patient_id,
        variable=variable,
        payload=validation_result.model_dump(),
        code_lock_hash=code_lock_hash,
    )
    # invariants uses best-effort validation (writes even if the envelope is invalid ->
    # log, don't raise), so it does not go through write_artifact; the artifact registry
    # still owns its path.
    inv_env["content_hash"] = hash_artifact_payload(inv_env)
    try:
        validate_artifact(inv_env, "invariants")
    except Exception as e:  # non-fatal, but surface rather than swallow silently
        _log.warning("invariants_validate_failed", extra_={"variable": variable, "error": str(e)})
    atomic_write_json(artifact_path("invariants", patient_root=patient_out, variable=variable), inv_env)

    # Patient-identifiable (PHI) evidence files + the de-identified (NO_PHI) record of
    # which chunks were selected, for this recipe's retrieve-and-prompt steps -- both
    # built from the transcript's step_evidence.
    step_evidence = list(transcript.get("step_evidence") or [])
    write_evidence_artifacts(run_id, patient_id, step_evidence)
    emit_selection_traces(run_id, patient_id, step_evidence)

    # De-identified (NO_PHI) extraction_outcome -- a small "did it work, and what did it
    # cost" record that seeds later quality analysis. Best-effort: this telemetry must
    # never break organization or raise across the PHI -> NO_PHI boundary.
    try:
        emit_extraction_outcome(
            run_id=run_id, patient_id=patient_id, transcript=transcript,
            ok=ok, data_final=data_final, errors=errors_out + warnings,
            variable_dir=variable_dir, validation_passed=validation.ok,
        )
    except Exception as e:  # noqa: BLE001 — this telemetry is non-fatal
        _log.warning("extraction_outcome_failed", extra_={"variable": variable, "error": str(e)})

    return organized


def emit_selection_traces(
    run_id: str,
    patient_id: str,
    step_evidence: list[dict[str, Any]],
) -> None:
    """Emit the de-identified (NO_PHI) record of which evidence chunks were selected,
    for each step that recorded one. This record carries no free text and no patient
    ids by construction (it uses salted stand-in keys plus a fixed vocabulary of
    document types, both enforced inside record_selection_trace). Best-effort -- this
    telemetry must never break organization, and must never raise across the
    PHI -> NO_PHI boundary."""
    for entry in step_evidence:
        trace = entry.get("trace")
        if not trace:
            continue
        try:
            record_selection_trace(run_id=run_id, patient_id=patient_id, **trace)
        except Exception as e:  # noqa: BLE001 — this telemetry is non-fatal
            _log.warning("selection_trace_failed", extra_={"variable": entry.get("variable"), "error": str(e)})
