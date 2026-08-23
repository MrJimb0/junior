"""The site-agnostic `stage` + `breast_receptors` finalize helpers.

stage is diagnosis-anchored: the finalize surfaces the original-diagnosis group with a
clinical/pathologic basis (rather than a later recurrence's group), grounds every
populated object, drops an ungrounded group, and stays clean under the step-8 provenance
validator. breast_receptors must yield all-null (clean) for a non-breast chart.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "var_extraction_recipes"
STAGE_HELPER = RECIPES / "oncology" / "general_oncology" / "stage" / "v1" / "stage_v1_python_helper.py"
RECEPTOR_HELPER = (
    RECIPES / "oncology" / "breast_oncology" / "breast_receptors" / "v1"
    / "breast_receptors_v1_python_helper.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stage(
    pathologic: dict | None = None,
    clinical: dict | None = None,
    posttreatment: dict | None = None,
) -> dict:
    mod = _load(STAGE_HELPER, "stage_helper_test")
    ctx = SimpleNamespace(step_outputs={
        "pathologic_tnm": {"data": pathologic or {}},
        "clinical_tnm": {"data": clinical or {}},
        "posttreatment_path_stage": {"data": posttreatment or {}},
    })
    return mod.finalize(ctx).data


def _receptors(data: dict | None = None, repair: dict | None = None) -> dict:
    mod = _load(RECEPTOR_HELPER, "receptor_helper_test")
    ctx = SimpleNamespace(step_outputs={
        "receptors": {"data": data or {}},
        "clinical_receptor_repair": {"data": repair or {}},
    })
    return mod.finalize(ctx).data


# ── stage: the Test_Patient case (pathologic wins, dx-anchored, provenanced) ──

def test_pathologic_stage_wins_and_is_grounded():
    out = _stage(
        pathologic={"pT": "T2a", "pN": "N0", "pM": "M0", "overall_group": "IIA",
                    "evidence_chunk_id": "P:pathology_report:0:2"},
        clinical={"cT": "T2a", "cN": "N0", "cM": "M0", "overall_group": "IIA",
                  "evidence_chunk_id": "P:clinical_note:0:2"},
    )
    assert out["stage_at_diagnosis"]["overall_group"] == "IIA"
    assert out["stage_at_diagnosis"]["basis"] == "pathologic"
    assert out["stage_at_diagnosis"]["evidence_chunk_id"] == "P:pathology_report:0:2"
    assert out["clinical_tnm"] == {
        "cT": "T2a", "cN": "N0", "cM": "M0", "evidence_chunk_id": "P:clinical_note:0:2"}
    assert out["pathologic_tnm"]["pT"] == "T2a"
    assert find_unprovenanced_value_paths(out) == []


def test_clinical_only_uses_clinical_basis():
    # de novo metastatic, never resected -> clinical staging only
    out = _stage(clinical={"cT": "T2", "cN": "N1", "cM": "M1", "overall_group": "IV",
                           "evidence_chunk_id": "P:clinical_note:0:0"})
    assert out["stage_at_diagnosis"] == {
        "overall_group": "IV", "basis": "clinical", "evidence_chunk_id": "P:clinical_note:0:0"}
    assert out["pathologic_tnm"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_nothing_found_is_all_null_and_clean():
    out = _stage()
    assert out["stage_at_diagnosis"] == {"overall_group": None, "basis": None, "evidence_chunk_id": None}
    assert out["clinical_tnm"] is None and out["pathologic_tnm"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_ungrounded_group_is_dropped():
    # a group with no citable chunk anywhere is not a usable claim -> nulled
    out = _stage(clinical={"overall_group": "IIIB", "evidence_chunk_id": None})
    assert out["stage_at_diagnosis"]["overall_group"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_ungrounded_tnm_object_is_dropped():
    # each TNM object cites its OWN pass; the clinical pass omitted the chunk, so its
    # TNM object is dropped (not cross-cited to the pathologic chunk). The grounded
    # pathologic object survives, and provenance stays clean.
    out = _stage(
        pathologic={"pT": "T3", "evidence_chunk_id": "P:pathology_report:0:1"},
        clinical={"cT": "T3", "cN": "N0", "evidence_chunk_id": None},
    )
    assert out["clinical_tnm"] is None
    assert out["pathologic_tnm"]["evidence_chunk_id"] == "P:pathology_report:0:1"
    assert find_unprovenanced_value_paths(out) == []


def test_both_tnm_and_group_dropped_when_nothing_cites():
    # both passes return TNM + a group but NO chunk (the prompt default null) ->
    # every ungrounded value is dropped; the output is all-null and clean.
    out = _stage(
        pathologic={"pT": "T2a", "pN": "N0", "pM": "M0", "overall_group": "IIA", "evidence_chunk_id": None},
        clinical={"cT": "T2a", "cN": "N0", "cM": "M0", "overall_group": "IIA", "evidence_chunk_id": None},
    )
    assert out["pathologic_tnm"] is None and out["clinical_tnm"] is None
    assert out["stage_at_diagnosis"] == {"overall_group": None, "basis": None, "evidence_chunk_id": None}
    assert find_unprovenanced_value_paths(out) == []


def test_occult_group_is_representable():
    # "occult" is a valid group value both prompts instruct the model to emit; it must
    # survive normalization rather than being uppercased out of its own enum.
    out = _stage(pathologic={"pT": "TX", "pN": "N0", "pM": "M0", "overall_group": "occult",
                             "evidence_chunk_id": "P:pathology_report:0:0"})
    assert out["stage_at_diagnosis"]["overall_group"] == "occult"
    assert out["stage_at_diagnosis"]["basis"] == "pathologic"
    assert find_unprovenanced_value_paths(out) == []


def test_basis_and_citation_track_the_group_source():
    # the group is stated only in the clinical note; the pathologic pass has a stray
    # TNM token but no group. basis must be 'clinical' and the group must cite the
    # clinical chunk -- not be mislabeled 'pathologic' / cited to the pathology chunk.
    out = _stage(
        pathologic={"pT": "T2a", "evidence_chunk_id": "P:pathology_report:0:1"},  # TNM, no group
        clinical={"overall_group": "IIIA", "evidence_chunk_id": "P:clinical_note:0:3"},
    )
    sad = out["stage_at_diagnosis"]
    assert sad["overall_group"] == "IIIA"
    assert sad["basis"] == "clinical"
    assert sad["evidence_chunk_id"] == "P:clinical_note:0:3"
    assert find_unprovenanced_value_paths(out) == []


def test_posttreatment_stage_is_preserved_without_replacing_diagnosis_stage():
    out = _stage(
        pathologic={
            "pT": "T2",
            "pN": "N1",
            "overall_group": "IIB",
            "evidence_chunk_id": "P:pathology_report:0:0",
        },
        posttreatment={
            "ypT": "T1",
            "ypN": "N0",
            "rcb_class": "I",
            "evidence_chunk_id": "P:pathology_report:2:0",
        },
    )
    assert out["stage_at_diagnosis"]["overall_group"] == "IIB"
    assert out["posttreatment_path_stage"]["ypT"] == "T1"
    assert out["posttreatment_path_stage"]["rcb_class"] == "I"
    assert find_unprovenanced_value_paths(out) == []


def test_group_normalization_and_rejection():
    mod = _load(STAGE_HELPER, "stage_helper_norm")
    assert mod._group("Stage IIA") == "IIA"
    assert mod._group("iib") == "IIB"
    assert mod._group("II") == "II"
    assert mod._group("occult") == "occult"   # the lone lowercase token, any case
    assert mod._group("OCCULT") == "occult"
    assert mod._group("unknown") is None
    assert mod._group("not a stage") is None
    assert mod._group(4) is None


# ── breast_receptors: null on a non-breast chart, grounded when found ──

def test_receptors_all_null_when_absent_is_clean():
    out = _receptors({})
    assert out["er_status"] is None and out["her2_status"] is None
    assert out["evidence_chunk_id"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_receptors_grounded_when_found():
    out = _receptors({"er_status": "positive", "pr_status": "negative",
                      "her2_status": "negative", "grade": 2, "tumor_size_cm": 1.8,
                      "margins": "clear", "evidence_chunk_id": "P:pathology_report:0:0"})
    assert out["er_status"] == "positive" and out["grade"] == 2
    assert out["evidence_chunk_id"] == "P:pathology_report:0:0"
    assert find_unprovenanced_value_paths(out) == []


def test_clinical_notes_fill_only_missing_receptors():
    out = _receptors(
        {
            "er_status": "positive",
            "pr_status": None,
            "evidence_chunk_id": "P:pathology_report:0:0",
        },
        {
            "er_status": "negative",
            "pr_status": "positive",
            "her2_status": "negative",
            "evidence_chunk_id": "P:clinical_note:3:0",
        },
    )
    assert out["er_status"] == "positive"
    assert out["pr_status"] == "positive"
    assert out["her2_status"] == "negative"
    assert out["evidence_chunk_ids"] == [
        "P:pathology_report:0:0",
        "P:clinical_note:3:0",
    ]


def test_receptors_found_without_evidence_are_dropped():
    # the model found ER+/grade but cited no chunk -> ungrounded -> all-null.
    # closes the er/pr/her2 _status validator gap at the helper level (those fields
    # bypass the generic provenance check, so the helper must guarantee grounding).
    out = _receptors({"er_status": "positive", "her2_status": "negative", "grade": 2,
                      "evidence_chunk_id": None})
    assert out["er_status"] is None and out["her2_status"] is None and out["grade"] is None
    assert out["evidence_chunk_id"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_receptors_reject_invalid_enums():
    out = _receptors({"er_status": "equivocal", "her2_status": "3+", "grade": 7,
                      "margins": "involved", "evidence_chunk_id": "P:pathology_report:0:0"})
    assert out["er_status"] is None and out["her2_status"] is None
    assert out["grade"] is None and out["margins"] is None
    # everything invalid -> nothing found -> evidence dropped, still clean
    assert out["evidence_chunk_id"] is None
    assert find_unprovenanced_value_paths(out) == []
