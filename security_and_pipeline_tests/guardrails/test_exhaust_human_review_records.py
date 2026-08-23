"""The human-review exhaust records (relevance_label, retriever_miss) are PHI-clean and
join back to the selection_judgment ranks they came from.

A reviewer's relevant/not_relevant/unsure verdict and a retriever-miss are emitted after
a run. They carry no patient id, no raw chunk id, and no manual-search text -- and the
chunk surrogate is minted in the shared evidence-chunk domain, so a label resolves to the
SAME stand-in as the chunk's selection_judgment row. That join is what makes the
retriever-vs-cross-encoder attribution computable in NO_PHI space; patient surrogates stay
per-record-type, so no cross-type patient join is created.
"""
import json

import pytest

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_relevance_label,
    record_retriever_miss,
    record_selection_trace,
)
from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import is_surrogate

RUN = "20990101_000000"
PID = "Test_Patient"
CHUNK = f"{PID}:clinical_note:0:0"
RECIPE = "date_of_diagnosis"
STEP = "find_original"


def setup_function(_):
    sl._CACHE.clear()


def _records(record_type, tmp_path):
    d = layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / record_type
    if not d.exists():
        return []
    out = []
    for shard in d.glob("*.jsonl"):
        out += [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    return out


def _label(tmp_path, *, chunk_id=CHUNK, label="relevant", reviewer="rev1", tier="expert"):
    return record_relevance_label(
        run_id=RUN, patient_id=PID, reviewer_id=reviewer, chunk_id=chunk_id, label=label,
        recipe_id=RECIPE, step_id=STEP, reviewer_tier=tier, recipe_version="v1",
        encoder_fingerprint="enc123", reranker_fingerprint="rr456", chunking_config_id="chk789",
        data_root=tmp_path,
    )


def _selection(tmp_path, *, chunk_id=CHUNK):
    ranked = [{
        "chunk_id": chunk_id, "doc_type": "clinical_note", "rank": 7, "prior_rank": 3,
        "score": 1.2, "token_count": 100, "in_bundle": False,
    }]
    return record_selection_trace(
        run_id=RUN, patient_id=PID, recipe_id=RECIPE, step_id=STEP, archetype="temporal_anchor",
        retriever_kind="bm25", total_evidence_tokens=4096, ranked=ranked, data_root=tmp_path,
        recipe_version="v1", encoder_fingerprint="enc123", reranker_fingerprint="rr456",
        chunking_config_id="chk789",
    )


def test_relevance_label_is_text_free_and_surrogated(tmp_path):
    shard = _label(tmp_path)
    raw = shard.read_text(encoding="utf-8")
    assert PID not in raw and "clinical_note:0:0" not in raw  # no patient id / raw chunk id
    assert layout.NON_SENSITIVE_LABEL in shard.parts and layout.SENSITIVE_LABEL not in str(shard)
    assert shard.parent == layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / "relevance_label"

    rec = _records("relevance_label", tmp_path)[0]
    assert rec["record_type"] == "relevance_label" and rec["label"] == "relevant"
    assert rec["reviewer_tier"] == "expert"
    assert is_surrogate(rec["chunk_surrogate"]) and is_surrogate(rec["patient_surrogate"])
    assert is_surrogate(rec["reviewer_surrogate"])
    assert rec["recipe_id"] == RECIPE and rec["step_id"] == STEP
    assert rec["reranker_fingerprint"] == "rr456"  # which cross-encoder version was judged


def test_label_joins_to_its_selection_judgment_chunk_surrogate(tmp_path):
    # THE join the whole flow depends on: same chunk_id + same run -> same chunk surrogate
    # on the human label and the selection_judgment row, so the label ties back to that
    # chunk's retriever_rank / cross_encoder_rank.
    _selection(tmp_path)
    sl._CACHE.clear()  # a separate review process re-derives the run secret from disk
    _label(tmp_path)
    sj_row = _records("selection_judgment", tmp_path)[0]["rows"][0]
    label = _records("relevance_label", tmp_path)[0]
    assert label["chunk_surrogate"] == sj_row["chunk_surrogate"]
    assert sj_row["prior_rank"] == 3 and sj_row["rank"] == 7  # the ranks the label attributes to


def test_retriever_miss_in_pool_joins_and_carries_ranks(tmp_path):
    _selection(tmp_path)
    record_retriever_miss(
        run_id=RUN, patient_id=PID, reviewer_id="rev1", selected_chunk_id=CHUNK,
        was_in_retriever_pool=True, recipe_id=RECIPE, step_id=STEP, reviewer_tier="trainee",
        retriever_rank_if_present=3, cross_encoder_rank_if_present=31, data_root=tmp_path,
    )
    sj_row = _records("selection_judgment", tmp_path)[0]["rows"][0]
    miss = _records("retriever_miss", tmp_path)[0]
    assert miss["was_in_retriever_pool"] is True
    assert miss["retriever_rank_if_present"] == 3 and miss["cross_encoder_rank_if_present"] == 31
    assert miss["manual_search_used"] is True
    assert miss["chunk_surrogate"] == sj_row["chunk_surrogate"]  # joins to the buried chunk


def test_retriever_miss_out_of_pool_is_clean_and_carries_no_join_ranks(tmp_path):
    shard = record_retriever_miss(
        run_id=RUN, patient_id=PID, reviewer_id="rev1", selected_chunk_id=f"{PID}:pathology:9:0",
        was_in_retriever_pool=False, recipe_id=RECIPE, step_id=STEP, data_root=tmp_path,
    )
    raw = shard.read_text(encoding="utf-8")
    assert PID not in raw and "pathology:9:0" not in raw
    miss = _records("retriever_miss", tmp_path)[0]
    assert miss["was_in_retriever_pool"] is False
    assert miss["retriever_rank_if_present"] is None and miss["cross_encoder_rank_if_present"] is None


def test_patient_surrogate_does_not_join_across_record_types(tmp_path):
    # Chunk linkage is shared; PATIENT linkage is not -- the same patient's stand-in
    # differs between a relevance_label and a retriever_miss, so no cross-type patient join.
    _label(tmp_path)
    record_retriever_miss(
        run_id=RUN, patient_id=PID, reviewer_id="rev1", selected_chunk_id=CHUNK,
        was_in_retriever_pool=True, recipe_id=RECIPE, step_id=STEP, data_root=tmp_path,
    )
    label = _records("relevance_label", tmp_path)[0]
    miss = _records("retriever_miss", tmp_path)[0]
    assert label["patient_surrogate"] != miss["patient_surrogate"]
    assert label["chunk_surrogate"] == miss["chunk_surrogate"]  # but the chunk DOES join


def test_unsure_label_round_trips(tmp_path):
    _label(tmp_path, label="unsure")
    assert _records("relevance_label", tmp_path)[0]["label"] == "unsure"


def test_unknown_reviewer_tier_degrades_to_unknown(tmp_path):
    _label(tmp_path, tier="attending_oncologist")
    assert _records("relevance_label", tmp_path)[0]["reviewer_tier"] == "unknown"


def test_invalid_label_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="relevance label must be one of"):
        _label(tmp_path, label="maybe")
    assert _records("relevance_label", tmp_path) == []  # nothing written


def test_label_surrogate_consistent_across_processes(tmp_path):
    # Same (secret, chunk_id) -> same surrogate, so two reviewers' labels point at the
    # same chunk and a re-emit joins identically.
    _label(tmp_path, reviewer="rev1")
    first = _records("relevance_label", tmp_path)[0]["chunk_surrogate"]
    sl._CACHE.clear()
    _label(tmp_path, reviewer="rev2")
    rows = _records("relevance_label", tmp_path)
    assert {r["chunk_surrogate"] for r in rows} == {first}
