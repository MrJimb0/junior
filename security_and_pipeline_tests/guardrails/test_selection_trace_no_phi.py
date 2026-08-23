"""The NO_PHI selection_judgment exhaust carries the relevance signal with no chunk
text, no real chunk_id, and no patient_id.

The PHI-side trace keys on chunk_ids (which embed the patient id), so it can't leave
the institution. ``record_selection_trace`` routes through the writer-owned ``emit()``
gate: every id is a domain-separated HMAC surrogate (minted with the PHI-side run
secret), and the row lands in a process-unique JSONL shard.
"""
import json

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_selection_trace,
)
from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import is_surrogate

RUN = "20990101_000000"
PID = "Test_Patient"


def setup_function(_):
    sl._CACHE.clear()


def _ranked():
    return [
        {"chunk_id": f"{PID}:clinical_note:0:0", "doc_type": "clinical_note",
         "rank": 0, "prior_rank": 2, "score": 1.5, "token_count": 312, "in_bundle": True},
        {"chunk_id": f"{PID}:pathology_report:1:3", "doc_type": "pathology_report",
         "rank": 1, "prior_rank": 0, "score": 0.9, "token_count": 280, "in_bundle": False},
    ]


def _emit(tmp_path, patient_id=PID):
    return record_selection_trace(
        run_id=RUN, patient_id=patient_id, recipe_id="date_of_diagnosis",
        step_id="find_original", archetype="temporal_anchor", retriever_kind="bm25",
        total_evidence_tokens=4096, ranked=_ranked(), data_root=tmp_path,
        recipe_version="v1", encoder_fingerprint="enc123", reranker_fingerprint="rr456",
        chunking_config_id="chk789", cited_chunk_ids=[f"{patient_id}:clinical_note:0:0"],
    )


def _records(shard_path):
    return [json.loads(line) for line in shard_path.read_text().splitlines() if line.strip()]


def test_trace_is_text_free_and_id_free(tmp_path):
    shard = _emit(tmp_path)
    raw = shard.read_text(encoding="utf-8")

    assert PID not in raw  # no patient id
    assert "clinical_note:0:0" not in raw and "pathology_report:1:3" not in raw  # no raw chunk ids
    assert layout.NON_SENSITIVE_LABEL in shard.parts  # under NO_PHI
    assert layout.SENSITIVE_LABEL not in str(shard)
    assert "patients" not in shard.parts
    assert shard.parent == layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / "selection_judgment"

    data = _records(shard)[0]
    assert is_surrogate(data["rows"][0]["chunk_surrogate"]) and is_surrogate(data["group"])
    # the learnable, non-PHI signal IS preserved
    assert data["recipe_id"] == "date_of_diagnosis" and data["archetype"] == "temporal_anchor"
    assert data["retriever"] == "bm25" and data["rows"][0]["doc_type"] == "clinical_note"
    assert data["rows"][0]["rank"] == 0 and data["rows"][0]["prior_rank"] == 2
    assert data["rows"][0]["in_bundle"] is True
    assert data["n_candidates"] == 2 and data["n_in_bundle"] == 1


def test_common_header_present_and_nonempty(tmp_path):
    """Every record carries the frozen identity header."""
    data = _records(_emit(tmp_path))[0]
    required = {
        "schema_version", "vocab_version", "site_id", "study_id", "run_id", "recipe_id",
        "recipe_version", "variable_key", "archetype", "encoder_fingerprint",
        "reranker_fingerprint", "extractor_fingerprint", "code_lock_hash",
        "chunking_config_id", "emitted_month", "prompt_hash", "schema_hash",
    }
    assert required <= set(data), f"missing header fields: {required - set(data)}"
    assert data["recipe_version"] == "v1"
    assert data["encoder_fingerprint"] == "enc123"
    assert data["reranker_fingerprint"] == "rr456"
    assert data["chunking_config_id"] == "chk789"
    assert data["emitted_month"] == "2099-01"  # month-resolution from the run_id
    assert data["record_type"] == "selection_judgment"


def test_model_citations_captured_as_weak_label(tmp_path):
    data = _records(_emit(tmp_path))[0]
    cited = [r for r in data["rows"] if r["cited"]]
    assert len(cited) == 1 and data["n_cited"] == 1
    assert is_surrogate(cited[0]["chunk_surrogate"])  # the real chunk_id never appears


def test_surrogates_are_consistent_across_calls(tmp_path):
    # Same (secret, raw) -> same surrogate, so rows join across SLURM procs.
    rec_a = _records(_emit(tmp_path))[0]
    sl._CACHE.clear()  # force a fresh secret read (simulates a second process)
    shard = _emit(tmp_path)
    rec_b = _records(shard)[-1]
    assert rec_a["rows"][0]["chunk_surrogate"] == rec_b["rows"][0]["chunk_surrogate"]


def test_secret_is_phi_side(tmp_path):
    _emit(tmp_path)
    secret_path = sl._run_secret_path(RUN, tmp_path)
    assert secret_path.exists() and layout.SENSITIVE_LABEL in secret_path.parts


def test_reranker_features_ride_the_row(tmp_path):
    # The per-candidate reranker features (cross_encoder_logit / fusion vector)
    # travel as numbers on the selection_judgment row -- the reranker-optimization signal.
    ranked = [{
        "chunk_id": f"{PID}:clinical_note:0:0", "doc_type": "clinical_note", "rank": 0,
        "prior_rank": 1, "score": 2.5, "token_count": 10, "in_bundle": True,
        "features": {"cross_encoder_logit": 2.4567, "lexical_overlap": 0.5},
    }]
    shard = record_selection_trace(
        run_id=RUN, patient_id=PID, recipe_id="r", step_id="s", archetype="point_fact",
        retriever_kind="hybrid", total_evidence_tokens=100, ranked=ranked, data_root=tmp_path,
    )
    row = _records(shard)[0]["rows"][0]
    assert row["features"]["cross_encoder_logit"] == 2.4567
    assert row["features"]["lexical_overlap"] == 0.5


def test_distinct_patients_get_distinct_group_surrogates(tmp_path):
    _emit(tmp_path, patient_id="patient_a")
    shard = _emit(tmp_path, patient_id="patient_b")
    groups = {rec["group"] for rec in _records(shard)}
    assert len(groups) == 2  # different patients -> different group surrogates, no collision
