"""The writer-owned emit() gate is fail-closed on PHI.

emit() substitutes raw ids -> surrogates, pydantic-validates, then runs the
forbidden-content scan; an adversarial record carrying a raw id / full clinical date /
source filename must raise BEFORE any shard line is written.
"""
from __future__ import annotations

import json

import pytest

from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl
from jr_pipeline.runtime_infrastructure.exhaust.forbidden_content import (
    scan_record_for_forbidden_content,
)
from jr_pipeline.runtime_infrastructure.exhaust.writer import ExhaustWriteError, emit

RUN = "20260617_000000"


def setup_function(_):
    sl._CACHE.clear()


def _recipe_header() -> dict:
    return dict(
        site_id="site_a", study_id="study_a", run_id=RUN, emitted_month="2026-06",
        encoder_fingerprint="enc0", reranker_fingerprint="rer0", extractor_fingerprint="ext0",
        code_lock_hash="sha256:abc", chunking_config_id="chunk0",
        recipe_id="date_of_birth", recipe_version="v1", variable_key="date_of_birth",
        archetype="point_fact", prompt_hash="sha256:abcd", schema_hash="sha256:ef01",
    )


def _outcome(**over) -> dict:
    return {**_recipe_header(), "ok": True, "value_shape": "scalar", **over}


def _shards_for(record_type, tmp_path):
    d = layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / record_type
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def test_emit_substitutes_raw_id_with_a_surrogate(tmp_path):
    shard = emit(
        "extraction_outcome", _outcome(),
        run_id=RUN, phi_keys={"patient_surrogate": "Test_Patient"}, data_root=tmp_path,
    )
    raw = shard.read_text()
    assert "Test_Patient" not in raw  # raw id never reaches the shard
    assert layout.NON_SENSITIVE_LABEL in shard.parts  # NO_PHI tree
    assert "patients" not in shard.parts
    rec = json.loads(raw.strip())
    assert rec["record_type"] == "extraction_outcome"
    assert rec["patient_surrogate"].startswith("surr:v1:run:patient_surrogate:")


def test_shard_name_is_process_unique(tmp_path):
    # method_provenance carries only RunHeader fields (no recipe-level fields).
    recipe_only = ("recipe_id", "recipe_version", "variable_key", "archetype",
                   "prompt_hash", "schema_hash")
    run_header = {k: v for k, v in _recipe_header().items() if k not in recipe_only}
    shard = emit("method_provenance", run_header, run_id=RUN, data_root=tmp_path)
    assert shard.name.startswith(f"{RUN}.") and shard.name.endswith(".jsonl")
    assert shard.name.count(".") >= 4  # run_id token + host.pid.uuid + .jsonl


def test_emit_rejects_unknown_record_type(tmp_path):
    with pytest.raises(ExhaustWriteError):
        emit("not_a_record", {}, run_id=RUN, data_root=tmp_path)


def test_emit_rejects_out_of_vocab_value(tmp_path):
    with pytest.raises(ExhaustWriteError):
        emit("extraction_outcome", _outcome(value_shape="blob"),
             run_id=RUN, phi_keys={"patient_surrogate": "p"}, data_root=tmp_path)
    assert _shards_for("extraction_outcome", tmp_path) == []  # nothing written


@pytest.mark.parametrize("bad_field,bad_value", [
    ("error_code", "MRN 12345678"),            # MRN content
    ("error_code", "seen 2026-01-15 in note"), # full clinical date
    ("error_code", "Test_Patient:notes:0:0"),  # raw chunk id
    ("error_code", "demographics:0:TABLE"),    # raw table chunk id
    ("error_code", "clinical_note.csv"),       # source filename
])
def test_emit_fails_closed_on_adversarial_content(tmp_path, bad_field, bad_value):
    with pytest.raises(ExhaustWriteError):
        emit("extraction_outcome", _outcome(ok=False, **{bad_field: bad_value}),
             run_id=RUN, phi_keys={"patient_surrogate": "p"}, data_root=tmp_path)
    assert _shards_for("extraction_outcome", tmp_path) == []  # nothing written


def test_emit_fails_closed_when_scoped_secret_missing(tmp_path):
    # site-scope secret is never auto-created -> emit must fail closed, write nothing.
    with pytest.raises(sl.ExhaustSecretError):
        emit("extraction_outcome", _outcome(), run_id=RUN, scope="site",
             phi_keys={"patient_surrogate": "p"}, data_root=tmp_path)


def test_scanner_passes_a_clean_record_and_exempts_surrogates():
    clean = _outcome(patient_surrogate="surr:v1:run:patient_surrogate:" + "a" * 24)
    assert scan_record_for_forbidden_content(clean) == []


@pytest.mark.parametrize("value,reason", [
    ("2026-01-15", "ISO absolute date"),
    ("MRN: 0012345", "MRN"),
    ("Test_Patient:notes:3:1", "raw chunk id"),
    ("patient:STSS01:structured:abcd:row:ef01:column:dx", "structured pointer"),
    ("JaneDoe_notes.xlsx", "source filename"),
    # red-team bypasses the allow-list rejects:
    ("Bassett, Kitty", "bare patient name"),
    ("Patient Katherine Jones doing well", "raw EHR prose"),
    ("E1234567", "non-standard patient id"),
    ("12345678", "bare numeric MRN"),
    ("01/15/2026", "MM/DD/YYYY date"),
    ("January 15, 2026", "textual date"),
    ("20260115", "compact date"),
    ("2026/01/15", "slash-ISO date"),
    ("tumor measured 3.2 cm in the left upper lobe", "clinical narrative"),
    ("scan.dcm", "DICOM source filename"),
    # a BARE single-token surname (no comma/space) must not pass as a "controlled
    # token"; every legit token is lowercase/snake/hex, so an uppercase
    # purely-alphabetic word is name-shaped and must be rejected.
    ("BASSETT", "bare all-caps surname"),
    ("Bassett", "bare titlecase surname"),
    ("Smith", "bare titlecase surname"),
    # a pure-decimal run of exactly 16 or 64 chars must not pass as a hash
    # ([0-9] is a subset of [0-9a-f]); an MRN/account-shaped id must be rejected.
    ("1234567890123456", "16-digit numeric posing as a 16-hex fingerprint"),
    ("9999999999999999999999999999999999999999999999999999999999999999", "64-digit numeric posing as a digest"),
])
def test_scanner_flags_each_forbidden_shape(value, reason):
    assert scan_record_for_forbidden_content({"error_code": value}) != [], reason


@pytest.mark.parametrize("value", [
    "surr:v1:run:patient_surrogate:" + "a" * 24,  # surrogate
    "sha256:" + "a" * 64,                          # full digest
    "a1b2c3d4e5f6a7b8",                            # 16-hex config fingerprint
    "2026-06",                                      # month stamp
    "v1", "1.0.0", "exhaust/v1", "doc_type/v1",    # versions
    "date_of_birth", "point_fact", "bm25",          # controlled tokens
    "clinical_invariants/cancer_staging",           # labeler id
    "unset_site", "unknown", "",                    # defaults / empty
])
def test_scanner_accepts_every_legitimate_controlled_form(value):
    assert scan_record_for_forbidden_content({"error_code": value}) == [], value


def test_run_id_shape_is_field_aware():
    # The timestamped run-id shape is allowed ONLY in the run_id field; the same shape
    # in any other field could be a compact clinical datetime and must be rejected.
    assert scan_record_for_forbidden_content({"run_id": "20260617_000000"}) == []
    assert scan_record_for_forbidden_content({"run_id": "20260617_000000_aa"}) == []
    assert scan_record_for_forbidden_content({"error_code": "20260617_000000"}) != []


def test_uppercase_token_is_field_aware():
    # an uppercase site/study code is operator config (never patient data),
    # so it is allowed in those fields; the SAME name-shaped token in any other field
    # is rejected as a possible patient surname.
    assert scan_record_for_forbidden_content({"site_id": "STANFORD"}) == []
    assert scan_record_for_forbidden_content({"study_id": "SHC"}) == []
    assert scan_record_for_forbidden_content({"group": "STANFORD"}) != []
    assert scan_record_for_forbidden_content({"step_id": "Bassett"}) != []
    # lowercase/snake/hex controlled tokens still pass everywhere
    assert scan_record_for_forbidden_content({"group": "g0", "step_id": "retrieve_rerank"}) == []
