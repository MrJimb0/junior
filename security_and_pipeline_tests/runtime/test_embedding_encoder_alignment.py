"""Retrieval verifies encoder alignment, not just dim.

Two different encoders can both produce dim=768; a dim-only check would let a corpus
embedded by encoder A be queried by encoder B, returning silently-wrong neighbors. The
retriever compares the query encoder fingerprint to the corpus's stored fingerprint.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.embedding.embedding_v1 import (
    verify_corpus_encoder_alignment,
)

_A = {"model_id": "modelA", "model_sha256": "sha-a", "pooling": "mean",
      "max_tokens": 512, "normalize": True, "dtype": None, "dim": 768}
_B = {"model_id": "modelB", "model_sha256": "sha-b", "pooling": "mean",
      "max_tokens": 512, "normalize": True, "dtype": None, "dim": 768}


def test_ok_when_fingerprints_match():
    assert verify_corpus_encoder_alignment(dict(_A), dict(_A), source_present=True) == "ok"


def test_mismatch_when_two_different_encoders_share_a_dimension():
    # both dim=768 but different model — exactly the silent-mismatch case
    assert verify_corpus_encoder_alignment(dict(_A), dict(_B), source_present=True) == "mismatch"


def test_legacy_sidecar_rebuilds_when_source_present():
    assert verify_corpus_encoder_alignment(dict(_A), None, source_present=True) == "rebuild"


def test_legacy_sidecar_fails_loud_when_source_gone():
    assert verify_corpus_encoder_alignment(dict(_A), None, source_present=False) == "fail"


# ── a cache-key field (tokenizer_hash) the stored corpus did not record must not
# falsely reject it. Only fields the STORED fingerprint recorded are compared;
# key-absent (legacy schema) != value=None (hub id).


def test_legacy_corpus_missing_new_cache_key_field_is_ok():
    # stored corpus predates tokenizer_hash (key ABSENT); live query computes it.
    # All recorded fields match -> aligned, must be 'ok'.
    stored_legacy = dict(_A)  # no tokenizer_hash key
    live_query = {**_A, "tokenizer_hash": "sha256:livehash"}
    assert verify_corpus_encoder_alignment(live_query, stored_legacy, source_present=True) == "ok"
    assert verify_corpus_encoder_alignment(live_query, stored_legacy, source_present=False) == "ok"


def test_real_tokenizer_swap_is_mismatch():
    # both recorded tokenizer_hash, values differ -> a real swap, must be caught
    stored = {**_A, "tokenizer_hash": "sha256:aaa"}
    query = {**_A, "tokenizer_hash": "sha256:bbb"}
    assert verify_corpus_encoder_alignment(query, stored, source_present=True) == "mismatch"


def test_hub_id_tokenizer_hash_none_both_sides_is_ok():
    # value=None (hub id, no local files) is a recorded value, compared, both None -> ok
    stored = {**_A, "tokenizer_hash": None}
    query = {**_A, "tokenizer_hash": None}
    assert verify_corpus_encoder_alignment(query, stored, source_present=True) == "ok"


def test_leniency_does_not_mask_a_recorded_field_mismatch():
    # skipping the absent tokenizer_hash must NOT hide a real difference on a field
    # the legacy corpus DID record (here model_sha256).
    stored_legacy = {**_A, "model_sha256": "sha-a"}  # no tokenizer_hash
    query = {**_A, "model_sha256": "sha-DIFFERENT", "tokenizer_hash": "sha256:livehash"}
    assert verify_corpus_encoder_alignment(query, stored_legacy, source_present=True) == "mismatch"


# ── cross-machine corpus portability: model_id (the encoder's load PATH) is EXCLUDED
# from the comparison, so a corpus built on a cluster and consumed on the laptop with
# byte-identical weights at a DIFFERENT path aligns; weight identity (model_sha256 +
# tokenizer_hash) still fully discriminates two different encoders.


def test_same_weights_different_model_id_path_is_ok():
    # Cluster build vs laptop consume: identical weights/tokenizer/config, different paths.
    cluster = {**_A, "tokenizer_hash": "sha256:same",
              "model_id": "/shared/lab/models/BioClinical-ModernBERT-base"}
    laptop = {**_A, "tokenizer_hash": "sha256:same",
              "model_id": "/Users/me/junior/models/embedding/BioClinical-ModernBERT-base"}
    # paths differ but weights/tokenizer match -> must be 'ok'.
    assert verify_corpus_encoder_alignment(laptop, cluster, source_present=True) == "ok"


def test_different_weights_mismatch_even_with_model_id_excluded():
    # Excluding the path must NOT let two genuinely different encoders align: a different
    # model_sha256 (even at the SAME model_id path) is still a mismatch.
    stored = {**_A, "model_id": "/same/path", "model_sha256": "sha-a"}
    query = {**_A, "model_id": "/same/path", "model_sha256": "sha-DIFFERENT"}
    assert verify_corpus_encoder_alignment(query, stored, source_present=True) == "mismatch"
