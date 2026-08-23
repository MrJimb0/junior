"""The date_of_death finalize helper must ground the vital-status /
date-of-death determination.

`vital_status` ("dead"/"alive") is a real clinical claim that the step-8 provenance
validator checks, so the finalize step must emit a non-empty `evidence_chunk_id` for
every positive determination. The chunk is the model's own citation when it names a
chunk the pass actually read, otherwise the top chunk the pass read -- so a small local
model that forgets to cite still yields honest provenance, and a hallucinated chunk id
the model was never shown is rejected.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
    find_unprovenanced_value_paths,
)

REPO = Path(__file__).resolve().parents[2]
HELPER = (
    REPO / "var_extraction_recipes" / "basic" / "date_of_death" / "v1"
    / "date_of_death_v1_python_helper.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("dod_helper_test", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _merge(
    *,
    demographics=None,
    death_search=None,
    eol=None,
    performance=None,
    recent=None,
    vital=None,
):
    """Build a StepContext-like object. Each pass is (data, read_chunk_ids)."""
    mod = _load()

    def entry(pair):
        data, read = pair if pair else ({}, [])
        return {"data": data, "evidence_chunk_ids": read}

    ctx = SimpleNamespace(step_outputs={
        "demographics_grab": entry(demographics),
        "death_search": entry(death_search),
        "eol_trajectory": entry(eol),
        "performance_status": entry(performance),
        "recent_activity": entry(recent),
        "vital_status_inference": entry(vital),
    })
    return mod.merge_passes(ctx).data


# ── the Test_Patient case: alive, grounded on the demographics row ──

def test_alive_is_grounded_by_demographics_chunk():
    out = _merge(
        demographics=(
            {"vital_status": "alive", "date_of_death": None, "confidence": 90,
             "evidence_chunk_id": "Pt:demographics:0:TABLE"},
            ["Pt:demographics:0:TABLE"],
        ),
        vital=(
            {"vital_status": "alive", "confidence": 70, "evidence_chunk_id": "Pt:clinical_note.csv:5:0"},
            ["Pt:clinical_note.csv:5:0"],
        ),
    )
    assert out["vital_status"] == "alive"
    assert out["date_of_death"] is None
    # the latest definite signal (vital inference pass) owns the grounding
    assert out["evidence_chunk_id"] == "Pt:clinical_note.csv:5:0"
    assert find_unprovenanced_value_paths(out) == []


def test_alive_grounds_by_read_fallback_when_model_omits_citation():
    # a 3B model that returns no evidence_chunk_id still yields honest provenance:
    # the helper falls back to the top chunk the pass actually read.
    out = _merge(
        demographics=(
            {"vital_status": "alive", "date_of_death": None, "evidence_chunk_id": None},
            ["Pt:demographics:0:TABLE"],
        ),
    )
    assert out["vital_status"] == "alive"
    assert out["evidence_chunk_id"] == "Pt:demographics:0:TABLE"
    assert find_unprovenanced_value_paths(out) == []


def test_holistic_adjudication_can_cite_a_summary_pass_chunk():
    out = _merge(
        eol=(
            {"strong_signals": ["hospice"], "evidence_chunk_id": "Pt:clinical_note.csv:8:0"},
            ["Pt:clinical_note.csv:8:0"],
        ),
        recent=(
            {"unexplained_silence_after_decline": True},
            ["Pt:clinical_note.csv:9:0"],
        ),
        vital=(
            {
                "vital_status": "dead",
                "confidence": 80,
                "evidence_chunk_id": "Pt:clinical_note.csv:8:0",
            },
            [],
        ),
    )
    assert out["vital_status"] == "dead"
    assert out["evidence_chunk_id"] == "Pt:clinical_note.csv:8:0"
    assert find_unprovenanced_value_paths(out) == []


# ── deceased paths ──

def test_explicit_date_in_demographics_grounds_the_date():
    out = _merge(
        demographics=(
            {"vital_status": "dead", "date_of_death": "2025-03-14",
             "evidence_chunk_id": "Pt:demographics:0:TABLE"},
            ["Pt:demographics:0:TABLE"],
        ),
    )
    assert out["date_of_death"] == "2025-03-14"
    assert out["vital_status"] == "dead"
    assert out["evidence_chunk_id"] == "Pt:demographics:0:TABLE"
    assert find_unprovenanced_value_paths(out) == []


def test_death_found_only_in_a_note_grounds_on_the_note():
    # demographics says nothing; a clinical note confirms death + date.
    out = _merge(
        demographics=({"vital_status": "unknown", "date_of_death": None, "evidence_chunk_id": None},
                      ["Pt:demographics:0:TABLE"]),
        death_search=(
            {"vital_status": "dead", "date_of_death": "2025-06-XX",
             "evidence_chunk_id": "Pt:clinical_note.csv:9:1"},
            ["Pt:clinical_note.csv:9:1", "Pt:clinical_note.csv:2:0"],
        ),
    )
    assert out["date_of_death"] == "2025-06-XX"
    assert out["vital_status"] == "dead"
    assert out["evidence_chunk_id"] == "Pt:clinical_note.csv:9:1"
    assert find_unprovenanced_value_paths(out) == []


def test_dead_without_a_date_is_still_grounded():
    # a deceased indicator with no calendar date: vital_status=dead, date null, but the
    # determination is still grounded on the row that carried the indicator.
    out = _merge(
        demographics=(
            {"vital_status": "dead", "date_of_death": None, "evidence_chunk_id": "Pt:demographics:0:TABLE"},
            ["Pt:demographics:0:TABLE"],
        ),
    )
    assert out["date_of_death"] is None
    assert out["vital_status"] == "dead"
    assert out["evidence_chunk_id"] == "Pt:demographics:0:TABLE"
    assert find_unprovenanced_value_paths(out) == []


# ── unknown / inference ──

def test_unknown_is_grounded_on_the_chart_examined():
    # no pass returns dead/alive; the inference pass read trajectory notes -> the
    # "unknown" determination is grounded on what was actually examined (p3 first).
    out = _merge(
        demographics=({"vital_status": "unknown", "evidence_chunk_id": None}, ["Pt:demographics:0:TABLE"]),
        vital=({"vital_status": "unknown", "evidence_chunk_id": None}, ["Pt:clinical_note.csv:7:0"]),
    )
    assert out["vital_status"] == "unknown"
    assert out["evidence_chunk_id"] == "Pt:clinical_note.csv:7:0"
    assert find_unprovenanced_value_paths(out) == []


# ── robustness: citations are validated, the no-evidence floor is honest ──

def test_hallucinated_citation_is_rejected_in_favor_of_a_read_chunk():
    # the model cites a chunk it was never shown -> fall back to the chunk it did read.
    out = _merge(
        demographics=(
            {"vital_status": "alive", "evidence_chunk_id": "Pt:NOT_SHOWN:99:9"},
            ["Pt:demographics:0:TABLE"],
        ),
    )
    assert out["evidence_chunk_id"] == "Pt:demographics:0:TABLE"
    assert find_unprovenanced_value_paths(out) == []


def test_date_grounding_wins_over_status_grounding():
    # an explicit date (strongest claim) owns the object-level pointer even when a later
    # pass also reports a vital status from a different chunk.
    out = _merge(
        demographics=(
            {"vital_status": "dead", "date_of_death": "2024-12-01",
             "evidence_chunk_id": "Pt:demographics:0:TABLE"},
            ["Pt:demographics:0:TABLE"],
        ),
        vital=({"vital_status": "dead", "evidence_chunk_id": "Pt:clinical_note.csv:3:0"},
               ["Pt:clinical_note.csv:3:0"]),
    )
    assert out["date_of_death"] == "2024-12-01"
    assert out["evidence_chunk_id"] == "Pt:demographics:0:TABLE"
    assert find_unprovenanced_value_paths(out) == []


def test_empty_chart_yields_unknown_placeholder_and_stays_clean():
    # nothing read by any pass (no demographics, no notes -- a sparse real chart): the
    # determination is "unknown" with no chunk to cite. "unknown" is a non-answer
    # placeholder, not a claim, so the validator does NOT flag it as ungrounded --
    # there is nothing to ground a non-determination on (symmetric with stage /
    # breast_receptors nulling "nothing found").
    out = _merge()
    assert out["vital_status"] == "unknown"
    assert out["date_of_death"] is None
    assert out["evidence_chunk_id"] is None
    assert find_unprovenanced_value_paths(out) == []


def test_confidence_carried_and_bool_rejected():
    out = _merge(
        demographics=({"vital_status": "alive", "confidence": True,
                       "evidence_chunk_id": "Pt:demographics:0:TABLE"},
                      ["Pt:demographics:0:TABLE"]),
        vital=({"vital_status": "alive", "confidence": 65,
                "evidence_chunk_id": "Pt:clinical_note.csv:1:0"},
               ["Pt:clinical_note.csv:1:0"]),
    )
    assert out["confidence"] == 65  # bool True ignored; last real int wins


# ── explicit death outranks soft inference; a stray date w/o a death is nulled ──

def test_explicit_death_outranks_alive_inference():
    # death_search makes an explicit death determination (post-enrichment, a high-bar
    # died/expired/pronounced statement) -> it is authoritative and is NOT downgraded by a
    # demographics 'alive' flag or the soft trajectory inference (p3), so a documented death and
    # its date survive.
    out = _merge(
        demographics=({"vital_status": "alive", "date_of_death": None,
                       "evidence_chunk_id": "Pt:demographics:0:TABLE"},
                      ["Pt:demographics:0:TABLE"]),
        death_search=({"vital_status": "dead", "date_of_death": "2026-01-15",
                       "evidence_chunk_id": "Pt:clinical_note.csv:1:0"},
                      ["Pt:clinical_note.csv:1:0"]),
        vital=({"vital_status": "alive", "evidence_chunk_id": "Pt:clinical_note.csv:7:5"},
               ["Pt:clinical_note.csv:7:5"]),
    )
    assert out["vital_status"] == "dead"
    assert out["date_of_death"] == "2026-01-15"
    # grounded on the death statement, not the inference
    assert out["evidence_chunk_id"] == "Pt:clinical_note.csv:1:0"
    assert find_unprovenanced_value_paths(out) == []


def test_unknown_status_nulls_a_stray_partial_date():
    # a date with NO confirmed-dead determination (no explicit death from p1/p2) is a
    # contradiction the guard drops -> date nulled. The explicit-death precedence doesn't
    # fire here (death_search says 'unknown', not 'dead'), so the guard still governs.
    out = _merge(
        death_search=({"vital_status": "unknown", "date_of_death": "2025-09-30",
                       "evidence_chunk_id": "Pt:clinical_note.csv:2:0"},
                      ["Pt:clinical_note.csv:2:0"]),
    )
    assert out["vital_status"] == "unknown"
    assert out["date_of_death"] is None
    assert find_unprovenanced_value_paths(out) == []
