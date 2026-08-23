"""Per-item / per-event grounding in the finalizes: cited, or reported uncited.

Per-item array objects (primaries[], genes[], lines[]) and finalize-created / re-affirmed
values (date_of_birth estimate, date_of_diagnosis re-affirmation) each carry the chunk the
deciding pass CITED. A pass that cited nothing leaves the pointer null.

These tests used to assert the opposite: that an uncited item was grounded on the top
chunk the pass read, so that find_unprovenanced_value_paths came back empty. That is the
whole objection to it — the fallback was satisfying the gate rather than the reader. The
substituted chunk WAS shown to the model, so it is grounded by construction; the value
then ships ok=true with a real quote beside it in the answer table that does not support
it. What these now assert is that an uncited value stays uncited and is REPORTED, which
is the only outcome a reviewer can act on.

These edges are NOT exercised by Test_Patient (its model happened to cite, or the arrays
were empty), so they are unit-tested here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths as unprovenanced,
)

RECIPES = Path(__file__).resolve().parents[2] / "var_extraction_recipes"


def _load(rel: str):
    p = RECIPES / rel
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _step(data, read):
    return {"data": data, "evidence_chunk_ids": read}


def test_genetics_germline_an_uncited_gene_is_reported_not_grounded():
    mod = _load("oncology/general_oncology/genetics_germline/v1/genetics_germline_v1_python_helper.py")
    step_context = SimpleNamespace(step_outputs={"extract_germline": _step(
        {"genetic_testing_done": True, "genes": [{"gene": "BRCA1", "result": "mutated"}]},
        ["Pt:clinical_note:1:4"])})
    out = mod.finalize(step_context).data
    assert out["genes"][0]["evidence_chunk_id"] is None
    # Both levels: the testing_done determination at the top is itself an uncited claim.
    assert unprovenanced(out) == ["data", "data.genes[0]"]


def test_genetics_somatics_an_uncited_gene_is_reported_not_grounded():
    mod = _load("oncology/general_oncology/genetics_somatics/v1/genetics_somatics_v1_python_helper.py")
    step_context = SimpleNamespace(step_outputs={"extract_somatic": _step(
        {"somatic_testing_done": True, "genes": [{"gene": "PIK3CA", "result": "mutated"}]},
        ["Pt:clinical_note:1:4"])})
    out = mod.finalize(step_context).data
    assert out["genes"][0]["evidence_chunk_id"] is None
    # Both levels: the testing_done determination at the top is itself an uncited claim.
    assert unprovenanced(out) == ["data", "data.genes[0]"]


def test_date_of_diagnosis_terse_reaffirmation_preserves_pointer():
    mod = _load("oncology/breast_oncology/date_of_diagnosis/v1/date_of_diagnosis_v1_python_helper.py")
    step_context = SimpleNamespace(step_outputs={
        "pathology_snippets": _step({"date_original_diagnosis": "2024-02-15", "original_certainty": 90,
                                     "original_evidence_chunk_id": "Pt:pathology_report:0:0"}, ["Pt:pathology_report:0:0"]),
        "pathology_full": _step({"date_original_diagnosis": "2024-02-15"}, ["Pt:pathology_report:1:0"]),
        "refine_clinical": _step({}, []),
        "repair_loco_met": _step({}, []),
    })
    out = mod.merge_passes(step_context).data
    # a later pass re-affirming the same date without re-citing must NOT clobber the pointer
    assert out["original_evidence_chunk_id"] == "Pt:pathology_report:0:0"
    assert unprovenanced(out) == []


def test_date_of_diagnosis_an_uncited_date_keeps_a_null_pointer():
    mod = _load("oncology/breast_oncology/date_of_diagnosis/v1/date_of_diagnosis_v1_python_helper.py")
    step_context = SimpleNamespace(step_outputs={
        "pathology_snippets": _step({"date_original_diagnosis": "2024-02-15"}, ["Pt:pathology_report:0:0"]),
        "pathology_full": _step({}, []),
        "refine_clinical": _step({}, []),
        "repair_loco_met": _step({}, []),
    })
    out = mod.merge_passes(step_context).data
    assert out["date_original_diagnosis"] == "2024-02-15"
    assert out["original_evidence_chunk_id"] is None
    assert unprovenanced(out) == ["data"]


def test_date_of_birth_an_uncited_age_estimate_is_reported_not_grounded():
    mod = _load("basic/date_of_birth/v1/date_of_birth_v1_python_helper.py")
    step_context = SimpleNamespace(step_outputs={
        "demographics_grab": _step({"date_of_birth": None}, ["Pt:demographics:0:TABLE"]),
        "bm25_explicit": _step({"date_of_birth": None, "age_years": 40, "doc_date": "2020-01-01",
                                "dob_evidence_chunk_id": None}, ["Pt:clinical_note:3:0"]),
        "age_date": _step({"date_of_birth": None, "age_years": 67, "doc_date": "2024-05-01",
                           "dob_evidence_chunk_id": None}, ["Pt:clinical_note:9:0"]),
    })
    out = mod.estimate_from_age(step_context).data
    # The estimate still stands — it is derived from an age the pass reported. What it
    # no longer gets is the pass's top chunk stapled on as though that passage stated
    # the date of birth, which it does not: it stated an age.
    assert out["estimated_date_of_birth"] == "1957-XX-XX"
    assert out["dob_evidence_chunk_id"] is None
    assert unprovenanced(out) == ["data"]
