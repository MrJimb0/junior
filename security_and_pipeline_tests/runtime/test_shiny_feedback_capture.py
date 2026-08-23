"""A correction from the Shiny app produces a file the training collector loads — and
that the eval scorer reads. Ties the two feedback surfaces to one on-disk contract.
"""
import importlib.util
import json
from pathlib import Path

from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import _load_corrections
from jr_pipeline.model_training.collect_feedback_from_shiny import (
    collect_all_feedback,
    filter_by_type,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
)

REPO = Path(__file__).resolve().parents[2]
_FC = REPO / "apps_and_interfaces/shiny_review_app/feedback_capture.py"
RUN = "shiny_demo_test"


def _load_module():
    spec = importlib.util.spec_from_file_location("feedback_capture_under_test", _FC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_correction_roundtrips_through_training_collector(tmp_path):
    fc = _load_module()
    path = fc.write_correction(
        patient_id="Test_Patient", variable="date_of_diagnosis",
        correct_value="2024-01-22", original_value="2024-02-15", run_id=RUN, dr=tmp_path,
    )
    assert path.name == "Test_Patient__date_of_diagnosis.json"

    records = collect_all_feedback(clinician_feedback_dir(RUN, tmp_path))
    assert len(records) == 1
    corr = filter_by_type(records, "extraction_correction")
    assert corr[0]["patient_id"] == "Test_Patient"
    assert corr[0]["variable"] == "date_of_diagnosis"
    assert corr[0]["correct_value"] == "2024-01-22"
    assert corr[0]["run_id"] == RUN


def test_reflagging_appends_under_the_same_run(tmp_path):
    fc = _load_module()
    fc.write_correction(patient_id="p", variable="v", correct_value="a", run_id=RUN, dr=tmp_path)
    path = fc.write_correction(patient_id="p", variable="v", correct_value="b", run_id=RUN, dr=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert len(record["feedback"]) == 2  # appended, not clobbered
    assert record["run_id"] == RUN


def test_feedback_without_a_run_id_is_refused_not_minted(tmp_path):
    """There is no demo mode to mint an id for anymore: a label that names no run can
    never be joined back to the extraction it judged, so the writer refuses."""
    import pytest as _pytest

    fc = _load_module()
    with _pytest.raises(ValueError, match="run id"):
        fc.write_correction(patient_id="p", variable="v", correct_value="a", dr=tmp_path)
    assert not (tmp_path / "CONTAINS_PHI").exists()


def test_eval_scorer_reads_widget_output(tmp_path):
    # The eval scorer must consume exactly what the widget writes.
    fc = _load_module()
    fc.write_correction(
        patient_id="Test_Patient", variable="date_of_diagnosis",
        correct_value="2024-01-22", run_id=RUN, dr=tmp_path,
    )
    corrections = _load_corrections(clinician_feedback_dir(RUN, tmp_path), "date_of_diagnosis")
    assert corrections["Test_Patient"]["date_of_diagnosis"] == "2024-01-22"


def test_confirmation_persists_with_reviewer_identity(tmp_path):
    # a confirm-correct action supplies the sensitivity denominator, tagged
    # with who reviewed it (Snorkel accuracy-weighting needs the labeler identity).
    fc = _load_module()
    fc.write_confirmation(
        patient_id="Test_Patient", variable="date_of_diagnosis", reviewed_value="2024-02-15",
        run_id=RUN, reviewer_id="dr_smith", reviewer_role="expert", dr=tmp_path,
    )
    records = collect_all_feedback(clinician_feedback_dir(RUN, tmp_path))
    confirms = filter_by_type(records, "extraction_confirmation")
    assert len(confirms) == 1
    assert confirms[0]["agreed"] is True
    assert confirms[0]["reviewer_id"] == "dr_smith"
    assert confirms[0]["reviewer_role"] == "expert"


def test_correction_and_confirmation_coexist(tmp_path):
    fc = _load_module()
    fc.write_correction(patient_id="p", variable="v", correct_value="a", run_id=RUN, dr=tmp_path)
    path = fc.write_confirmation(patient_id="p", variable="v", run_id=RUN, dr=tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))
    types = [e["type"] for e in record["feedback"]]
    assert types == ["extraction_correction", "extraction_confirmation"]


def test_sampling_frame_records_the_draw(tmp_path):
    fc = _load_module()
    path = fc.write_sampling_frame(
        drawn=[{"patient_id": "p1", "variable": "date_of_diagnosis"},
               {"patient_id": "p2", "variable": "date_of_diagnosis"}],
        rule="random_2_per_variable", run_id=RUN, dr=tmp_path,
    )
    frame = json.loads(path.read_text(encoding="utf-8"))
    assert frame["rule"] == "random_2_per_variable"
    assert frame["n_drawn"] == 2 and len(frame["drawn"]) == 2


def test_chunk_relevance_producer_feeds_the_reranker_consumers(tmp_path):
    # write_chunk_relevance is the producer for the chunk_relevance consumers
    # (they filter type=='chunk_relevance').
    fc = _load_module()
    fc.write_chunk_relevance(
        patient_id="Test_Patient", variable="date_of_diagnosis",
        chunk_id="Test_Patient:notes:0:0", relevant=True, run_id=RUN,
        reviewer_id="dr_smith", reviewer_role="expert", dr=tmp_path,
    )
    records = collect_all_feedback(clinician_feedback_dir(RUN, tmp_path))
    rel = filter_by_type(records, "chunk_relevance")
    assert len(rel) == 1
    assert rel[0]["chunk_id"] == "Test_Patient:notes:0:0"
    assert rel[0]["relevant"] is True and rel[0]["reviewer_role"] == "expert"


def test_correction_carries_provenance(tmp_path):
    # a correction is attributable to a specific extraction.
    fc = _load_module()
    path = fc.write_correction(
        patient_id="p", variable="v", correct_value="a", run_id=RUN, dr=tmp_path,
        model="qwen3b", extracted_at="2026-06-17T00:00:00", evidence_chunk_id="p:notes:0:0",
    )
    entry = json.loads(path.read_text())["feedback"][0]
    assert entry["model"] == "qwen3b" and entry["evidence_chunk_id"] == "p:notes:0:0"


def test_append_is_atomic_no_residual_temp_files(tmp_path):
    # writes go through a unique temp + os.replace; no temp file is left behind.
    fc = _load_module()
    fc.write_correction(patient_id="p", variable="v", correct_value="a", run_id=RUN, dr=tmp_path)
    fc.write_confirmation(patient_id="p", variable="v", run_id=RUN, dr=tmp_path)
    out_dir = clinician_feedback_dir(RUN, tmp_path)
    assert json.loads((out_dir / "p__v.json").read_text())["feedback"]  # intact
    assert not list(out_dir.glob(".*tmp"))  # no leaked temp files


def test_invalid_reviewer_role_falls_back_to_unknown(tmp_path):
    fc = _load_module()
    fc.write_confirmation(patient_id="p", variable="v", run_id=RUN,
                          reviewer_role="wizard", dr=tmp_path)
    record = json.loads((clinician_feedback_dir(RUN, tmp_path) / "p__v.json").read_text())
    assert record["feedback"][0]["reviewer_role"] == "unknown"


def test_relevance_exhaust_twin_joins_the_selection_ranks(tmp_path):
    # The app's per-judgment NO_PHI twin (emit_relevance_label_exhaust) mints the
    # SAME chunk surrogate as the run's selection_judgment row for that chunk id,
    # so the human label joins the ranks it graded -- entirely in NO_PHI space.
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        record_selection_trace,
    )
    from jr_pipeline.runtime_infrastructure import (
        data_directory_layout_and_safe_writes as layout,
    )
    from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle

    secret_lifecycle._CACHE.clear()
    fc = _load_module()
    run, chunk = "20260101_000000", "Test_Patient:clinical_note:0:0"

    record_selection_trace(
        run_id=run, patient_id="Test_Patient", recipe_id="date_of_diagnosis",
        step_id="find_original", archetype=None, retriever_kind="bm25",
        total_evidence_tokens=10,
        ranked=[{"chunk_id": chunk, "doc_type": "clinical_note", "rank": 0, "in_bundle": True}],
        data_root=tmp_path,
    )
    fc.emit_relevance_label_exhaust(
        patient_id="Test_Patient", variable="date_of_diagnosis", chunk_id=chunk,
        relevant=True, run_id=run, reviewer_role="expert", dr=tmp_path,
    )

    def _records(record_type):
        d = layout.no_phi_exhaust_shards_dir(run, tmp_path) / record_type
        return [json.loads(line)
                for shard in d.glob("*.jsonl")
                for line in shard.read_text().splitlines() if line.strip()]

    (label,) = _records("relevance_label")
    (judgment,) = _records("selection_judgment")
    assert label["chunk_surrogate"] == judgment["rows"][0]["chunk_surrogate"]
    assert label["label"] == "relevant" and label["reviewer_tier"] == "expert"
    assert chunk not in json.dumps(label)  # raw chunk id never leaves the PHI side
