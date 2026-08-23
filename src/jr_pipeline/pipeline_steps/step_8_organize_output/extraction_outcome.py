"""Build + emit the per-recipe extraction_outcome record (a de-identified "telemetry"
record left behind by each extraction).

For each (patient, recipe) this captures *did the extraction succeed, and if not why,
and what did it cost* -- never the extracted value, the prompt, the model's response,
or any exception text. Everything is derived at step 8 from the transcript (run
record), the computed result, and the step receipts. These are read on the
patient-identifiable (PHI) side, but only counts and fixed category labels leave:

* ``ok`` / ``value_shape`` / ``n_value_fields`` from the final data (its shape, not its
  content);
* ``error_kind`` / ``parse_failure_kind`` -- the error strings are read only to pick a
  category by the error's TYPE; the raw message itself never enters the record; and
* token cost (``prompt/completion/total_tokens``, and whether every response was
  ``cached``) summed across the receipts of every step that called a model.

The real patient id is replaced by a one-way keyed stand-in (an HMAC surrogate) via
``emit``'s ``phi_keys``. Richer signals (why the model stopped, how many retries) are
not yet passed through from step 7 (deferred).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    common_header,
)
from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    ErrorKind,
    ParseFailureKind,
    ValueShape,
)
from jr_pipeline.runtime_infrastructure.exhaust.writer import emit
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_UNKNOWN = "unknown"
_log = get_logger("extraction_outcome")


def _value_shape(data: Any) -> str:
    if data is None:
        return ValueShape.NULL.value
    if isinstance(data, list):
        return ValueShape.LIST.value
    if isinstance(data, dict):
        return ValueShape.OBJECT.value
    return ValueShape.SCALAR.value


def _n_value_fields(data: Any) -> int:
    if isinstance(data, dict):
        return sum(1 for v in data.values() if v is not None)
    if isinstance(data, list):
        return len(data)
    return 0 if data is None else 1


def _bucket_error(errors: list[str], validation_passed: bool = True) -> tuple[str, str]:
    """Classify by error TYPE only -- the raw string is read to pick a category,
    never stored. Returns (error_kind, parse_failure_kind)."""
    blob = " ".join(str(e) for e in errors).lower()
    if "jsondecode" in blob:
        return ErrorKind.PARSE_FAILURE.value, ParseFailureKind.JSON_DECODE_ERROR.value
    if "empty content" in blob:
        return ErrorKind.PARSE_FAILURE.value, ParseFailureKind.EMPTY_CONTENT.value
    if "not a json" in blob:
        return ErrorKind.PARSE_FAILURE.value, ParseFailureKind.NOT_A_JSON_OBJECT.value
    if "empty_evidence" in blob:
        return ErrorKind.EMPTY_EVIDENCE.value, ParseFailureKind.NONE.value
    if validation_passed and errors:
        # Something went wrong and the output schema was satisfied, so whatever it was,
        # it was not the value being invalid. Calling it a validation failure sends
        # whoever reads this telemetry looking at the recipe's schema for a fault that
        # is in the step that crashed before it.
        return ErrorKind.UNKNOWN.value, ParseFailureKind.NONE.value
    if errors:
        return ErrorKind.VALIDATION_FAILURE.value, ParseFailureKind.NONE.value
    return ErrorKind.UNKNOWN.value, ParseFailureKind.NONE.value


def _economics_from_receipts(
    steps: list[dict[str, Any]], variable_dir: Path | None
) -> dict[str, Any]:
    """Add up what every step of this variable spent, across all its receipts.

    A recipe is several model calls — retrieve and prompt, then a repair, then a
    finalizer — and reporting the first one's tokens as the variable's cost understates
    a five-step recipe by most of what it spent. The number is read as the cost of
    extracting the variable, so it has to be the sum. It carries no count of how many
    calls it covers: ExtractionOutcome has no field for one, and a number the schema
    does not declare is dropped on the way out rather than reaching anybody.

    A step records its receipt as a path relative to its own variable directory, which is
    the right thing to keep on disk -- a run stays readable after the folder is moved. So
    the reader must join it back onto that directory, and ``variable_dir`` has to be the
    folder the transcript was actually read from. Joined onto anything else -- the
    directory the command was typed in, or a location the run recorded back when it was
    written on some other machine -- it finds nothing, and every token count in the cost
    table comes out null while the table still looks complete: a run that spent thousands
    of tokens is indistinguishable from one that spent none.

    A step with no receipt path was skipped and cost nothing, so it is passed over in
    silence; a receipt that was recorded and cannot be read is said out loud."""
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    calls = 0
    every_call_was_cached = True
    for step in steps:
        receipt_path = step.get("receipt_path")
        if not receipt_path:
            continue
        if variable_dir is None:
            _log.warning("step_receipt_location_unknown", extra_={"receipt": receipt_path})
            return {}
        try:
            env = json.loads((variable_dir / receipt_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _log.warning("step_receipt_unreadable", extra_={"receipt": receipt_path})
            continue
        econ = (env.get("payload") or {}).get("economics")
        if not econ:
            continue
        calls += 1
        for field in totals:
            counted = econ.get(field)
            if isinstance(counted, int):
                totals[field] += counted
        # A run is only free if nothing in it went to the model, so one live call makes
        # the whole variable uncached.
        every_call_was_cached = every_call_was_cached and bool(econ.get("cached"))
    if not calls:
        return {}
    return {**totals, "cached": every_call_was_cached}


def emit_extraction_outcome(
    *,
    run_id: str,
    patient_id: str,
    transcript: dict[str, Any],
    ok: bool,
    data_final: Any,
    errors: list[str],
    variable_dir: Path | None = None,
    data_root: Path | None = None,
    validation_passed: bool = True,
) -> Path:
    """Build + emit one extraction_outcome record from a recipe's transcript + result.

    ``variable_dir`` is the folder this transcript was read from; the step receipts it
    names are recorded relative to it. A caller that cannot say where the transcript
    lives leaves the token columns empty rather than reporting a run as free."""
    recipe = transcript.get("recipe") or {}
    step_evidence = list(transcript.get("step_evidence") or [])
    trace = next((e.get("trace") for e in step_evidence if e.get("trace")), {}) or {}

    header = common_header(
        run_id=run_id,
        recipe_id=recipe.get("name") or transcript.get("variable") or _UNKNOWN,
        recipe_version=recipe.get("version") or _UNKNOWN,
        archetype=trace.get("archetype"),
        encoder_fingerprint=trace.get("encoder_fingerprint", _UNKNOWN),
        reranker_fingerprint=trace.get("reranker_fingerprint", _UNKNOWN),
        chunking_config_id=trace.get("chunking_config_id", _UNKNOWN),
        extractor_fingerprint=transcript.get("provider_config_hash") or _UNKNOWN,
        code_lock_hash=transcript.get("code_lock_hash") or _UNKNOWN,
    )
    error_kind, parse_failure_kind = (
        (ErrorKind.NONE.value, ParseFailureKind.NONE.value) if ok
        else _bucket_error(errors, validation_passed=validation_passed)
    )
    record = {
        **header,
        "ok": bool(ok),
        "value_shape": _value_shape(data_final),
        "n_value_fields": _n_value_fields(data_final),
        "error_kind": error_kind,
        "parse_failure_kind": parse_failure_kind,
        **_economics_from_receipts(
            list(transcript.get("steps") or []),
            Path(variable_dir) if variable_dir is not None else None,
        ),
    }
    return emit(
        "extraction_outcome", record, run_id=run_id,
        phi_keys={"patient_surrogate": patient_id}, data_root=data_root,
    )
