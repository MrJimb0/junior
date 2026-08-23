"""Step 8 organizes per-recipe output from transcripts.

Covers the ok-semantics (a recovered intermediate step error is a warning,
not a failure) and the result.json envelope/path contract downstream readers depend on.
"""
from __future__ import annotations

import json
from pathlib import Path

from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
    organize_per_recipe_outputs,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
)


def _schema(tmp_path: Path) -> Path:
    p = tmp_path / "dob.schema.json"
    p.write_text(
        json.dumps({
            "type": "object",
            "properties": {"date_of_birth": {"type": "string", "format": "date"}},
            "required": ["date_of_birth"],
        }),
        encoding="utf-8",
    )
    return p


def _write_transcript(
    patient_out: Path, variable: str, *, data_final, errors, schema_path: Path,
    provenance_rejected=None,
) -> None:
    vdir = extract_output_dir(patient_out) / variable
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "transcript.json").write_text(
        json.dumps({
            "sensitivity": "high",
            "run_id": "20990101_000000",
            "patient_id": "P",
            "variable": variable,
            "recipe": {"name": variable, "version": "v1"},
            "recipe_path": str(patient_out / "x" / "v1" / "r.yaml"),
            "output_schema_path": str(schema_path),
            "validation_rules": [],
            "depends_on": [],
            "data_final": data_final,
            "errors": errors,
            "steps": [],
            "total_elapsed_s": 0.1,
            "provider_config_hash": None,
            "code_lock_hash": None,
            "step_evidence": [],
            "provenance_rejected": provenance_rejected or [],
        }),
        encoding="utf-8",
    )


def _result_payload(patient_out: Path, variable: str) -> dict:
    env = json.loads((extract_output_dir(patient_out) / variable / "result.json").read_text())
    return env["payload"]


def _organize(patient_out: Path, tmp_path: Path, names: list[str]) -> dict:
    return organize_per_recipe_outputs(
        run_id="20990101_000000",
        patient_id="P",
        patient_out=patient_out,
        recipes_root=tmp_path,
        ordered_names=names,
        code_lock_hash=None,
    )


def test_recovered_intermediate_error_does_not_fail_ok(tmp_path):
    # a step errored, but the FINAL data is schema-valid -> ok True, error -> warning
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "date_of_birth",
        data_final={"date_of_birth": "1957-04-15", "dob_evidence_chunk_id": "p1:demographics:0:TABLE"},
        errors=["step_a: transient hiccup"], schema_path=_schema(tmp_path),
    )
    results = _organize(patient_out, tmp_path, ["date_of_birth"])

    assert results["date_of_birth"]["ok"] is True
    assert results["date_of_birth"]["warnings"] == ["step_a: transient hiccup"]
    assert results["date_of_birth"]["errors"] == []

    payload = _result_payload(patient_out, "date_of_birth")
    assert payload["ok"] is True
    assert payload["warnings"] == ["step_a: transient hiccup"]
    assert payload["data"] == {"date_of_birth": "1957-04-15", "dob_evidence_chunk_id": "p1:demographics:0:TABLE"}
    assert payload["variable"] == "date_of_birth"  # path/shape contract for downstream readers


def test_invalid_final_data_fails_ok_with_errors(tmp_path):
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "date_of_birth",
        data_final={"date_of_birth": "not-a-date", "dob_evidence_chunk_id": "p1:demographics:0:TABLE"},  # bad format -> validation fails
        errors=[], schema_path=_schema(tmp_path),
    )
    results = _organize(patient_out, tmp_path, ["date_of_birth"])

    assert results["date_of_birth"]["ok"] is False
    assert results["date_of_birth"]["errors"]  # validation errors surfaced
    assert _result_payload(patient_out, "date_of_birth")["ok"] is False


def test_missing_final_data_fails_ok(tmp_path):
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "date_of_birth",
        data_final=None, errors=["all steps failed"], schema_path=_schema(tmp_path),
    )
    results = _organize(patient_out, tmp_path, ["date_of_birth"])
    assert results["date_of_birth"]["ok"] is False
    assert results["date_of_birth"]["warnings"] == ["all steps failed"]


def test_provenance_rejected_surfaces_from_transcript_into_result_json(tmp_path):
    # The extract step records fabricated-citation rejections in the transcript; step 8 must
    # surface them into result.json so a reviewer can tell "cited nothing" from "made up a
    # citation". Without this the rejection is recorded but invisible downstream.
    patient_out = tmp_path / "patient"
    rejected = [{"path": "date_of_birth_evidence_chunk_id", "chunk_id": "fabricated:99",
                 "step_id": "prompt"}]
    _write_transcript(
        patient_out, "date_of_birth",
        data_final={"date_of_birth": "1957-04-15", "dob_evidence_chunk_id": "p1:demographics:0:TABLE"}, errors=[], schema_path=_schema(tmp_path),
        provenance_rejected=rejected,
    )
    _organize(patient_out, tmp_path, ["date_of_birth"])
    assert _result_payload(patient_out, "date_of_birth")["provenance_rejected"] == rejected


def test_missing_transcript_is_skipped(tmp_path):
    patient_out = tmp_path / "patient"
    results = _organize(patient_out, tmp_path, ["never_ran"])
    assert results == {}


def test_emit_selection_traces_writes_id_free_no_phi_trace(tmp_path, monkeypatch):
    # step 8 emits the NO_PHI selection trace from the transcript's step_evidence.
    # Entries without a trace are skipped; the emitted trace must carry no raw id
    # (salted surrogates only).
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
        emit_selection_traces,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        no_phi_exhaust_shards_dir,
    )

    step_evidence = [
        {"variable": "v/s", "evidence_packet": {}},  # no trace -> skipped
        {"variable": "date_of_birth/extract", "evidence_packet": {}, "trace": {
            "recipe_id": "date_of_birth", "step_id": "extract", "archetype": "point_fact",
            "retriever_kind": "bm25", "total_evidence_tokens": 4096,
            "ranked": [{
                "chunk_id": "Test_Patient:notes:0:0", "doc_type": "progress_note",
                "rank": 1, "prior_rank": 1, "score": 0.9, "token_count": 5, "in_bundle": True,
            }],
            "recipe_version": "v1", "encoder_fingerprint": "abc",
            "reranker_fingerprint": "def", "chunking_config_id": "ghi",
            "cited_chunk_ids": ["Test_Patient:notes:0:0"],
        }},
    ]
    emit_selection_traces("20990101_000000", "Test_Patient", step_evidence)

    shard_dir = no_phi_exhaust_shards_dir("20990101_000000") / "selection_judgment"
    shards = list(shard_dir.glob("*.jsonl"))
    assert len(shards) == 1  # exactly one trace emitted -- the trace-less entry was skipped
    raw = shards[0].read_text(encoding="utf-8")
    assert raw.count("\n") == 1                       # one record line
    assert "Test_Patient" not in raw                 # patient id -> HMAC surrogate
    assert "Test_Patient:notes:0:0" not in raw       # raw chunk id never travels


def test_emit_selection_traces_never_raises_on_bad_trace(tmp_path, monkeypatch):
    # telemetry must never break organize, nor cross the seam, on a malformed trace
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
        emit_selection_traces,
    )
    emit_selection_traces("20990101_000000", "P", [{"variable": "v/s", "trace": {"bogus": True}}])


def _nullable_pointer_schema(tmp_path: Path) -> Path:
    """A schema shaped like stage_v1: the evidence pointer is nullable and optional.

    This is the shape that let a fabricated citation through. Contrast a schema
    declaring the same field `minLength: 1` and required -- there, nulling the
    pointer fails schema validation on its own."""
    p = tmp_path / "stage.schema.json"
    p.write_text(
        json.dumps({
            "type": "object",
            "properties": {
                "clinical_tnm": {"oneOf": [{
                    "type": "object",
                    "properties": {
                        "cT": {"type": ["string", "null"]},
                        "evidence_chunk_id": {"type": ["string", "null"]},
                    },
                }, {"type": "null"}]},
            },
        }),
        encoding="utf-8",
    )
    return p


def test_a_value_whose_citation_was_rejected_is_not_ok(tmp_path):
    """The live failure this pins, seen on a real chart.

    The model returned clinical_tnm = cT1c/N0/M0 citing clinical_note:368:0 -- a chunk
    that appears nowhere in the 16 it was shown. The extract step caught it and nulled
    the pointer, exactly as designed. Then the value shipped `ok: true` with an empty
    warnings list, because stage_v1's schema permits a null pointer, so validation had
    no complaint. A reviewer reading `ok` got a staged tumor with nothing behind it.

    Provenance is not a per-recipe matter of taste, so `ok` answers it directly now."""
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "stage",
        data_final={"clinical_tnm": {"cT": "T1c", "evidence_chunk_id": None}},
        errors=[], schema_path=_nullable_pointer_schema(tmp_path),
        provenance_rejected=[{
            "path": "data.clinical_tnm.evidence_chunk_id",
            "chunk_id": "Patient_B:clinical_note:368:0",
            "step": "finalize",
        }],
    )
    results = _organize(patient_out, tmp_path, ["stage"])

    assert results["stage"]["ok"] is False, "a value with no source span shipped as ok"

    payload = _result_payload(patient_out, "stage")
    assert payload["ok"] is False
    assert payload["unprovenanced_value_paths"] == ["data.clinical_tnm"]
    assert payload["provenance_rejected"], "the rejection must still be on the receipt"
    reason = " ".join(payload["errors"])
    assert "data.clinical_tnm" in reason
    # A reviewer has to be able to tell an invented citation from an absent one; by this
    # point the pointer reads null either way.
    assert "rejected as ungrounded" in reason, reason


def test_a_value_that_simply_cited_nothing_says_so_differently(tmp_path):
    """Same null pointer, no rejection behind it -- the model never offered a citation
    rather than inventing one. Both are unsupported claims and both fail; the message
    must not accuse the model of fabricating when it did not."""
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "stage",
        data_final={"clinical_tnm": {"cT": "T1c", "evidence_chunk_id": None}},
        errors=[], schema_path=_nullable_pointer_schema(tmp_path),
    )
    results = _organize(patient_out, tmp_path, ["stage"])

    assert results["stage"]["ok"] is False
    reason = " ".join(_result_payload(patient_out, "stage")["errors"])
    assert "data.clinical_tnm" in reason
    assert "rejected as ungrounded" not in reason, reason


def test_a_grounded_value_under_the_same_schema_is_still_ok(tmp_path):
    """The gate must fail ungrounded values, not every value under a lenient schema."""
    patient_out = tmp_path / "patient"
    _write_transcript(
        patient_out, "stage",
        data_final={"clinical_tnm": {
            "cT": "T1c", "evidence_chunk_id": "Patient_B:clinical_note:357:5",
        }},
        errors=[], schema_path=_nullable_pointer_schema(tmp_path),
    )
    results = _organize(patient_out, tmp_path, ["stage"])

    assert results["stage"]["ok"] is True
    assert _result_payload(patient_out, "stage")["unprovenanced_value_paths"] == []
