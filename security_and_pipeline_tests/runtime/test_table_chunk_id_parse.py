"""The :TABLE chunk_id parser must split from the RIGHT so a patient id
containing ':' cannot shift the (stem, row_id) fields.

Format: ``{patient_id}:{source_stem}:{row_id}:TABLE``. A positional left-split
breaks on colon-bearing ids and silently mis-resolves the source stem (which then
zeroes the reranker's source_priority feature).
"""
import json

import polars as pl
import pytest

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    PatientChunkStore,
    build_structured_evidence_pointer,
    parse_structured_evidence_pointer,
)

_parse = PatientChunkStore._parse_table_chunk_id


def test_normal_id():
    assert _parse("Test_Patient:diagnoses:3:TABLE") == ("diagnoses", 3)


def test_colon_bearing_patient_id_does_not_shift_fields():
    # A patient id with embedded colons must still resolve stem=clinical_note, row=5.
    assert _parse("weird:pt:id:clinical_note:5:TABLE") == ("clinical_note", 5)


def test_non_table_chunk_returns_none():
    assert _parse("Test_Patient:clinical_note:0:0") is None


def test_malformed_row_returns_none():
    assert _parse("p:stem:notanumber:TABLE") is None


# --- structured evidence pointer (build/parse) ---

def test_structured_evidence_pointer_round_trips():
    pointer = build_structured_evidence_pointer(
        patient_id="Test_Patient",
        file_hash="sha256:abc123",
        row_hash="sha256:def456",
        column="drug",
    )
    # hashes are stored bare so the pointer carries no stray ':'
    assert pointer == "patient:Test_Patient:structured:abc123:row:def456:column:drug"
    assert parse_structured_evidence_pointer(pointer) == {
        "patient_id": "Test_Patient",
        "file_hash": "abc123",
        "row_hash": "def456",
        "column": "drug",
    }


def test_structured_pointer_parse_rejects_non_pointers():
    assert parse_structured_evidence_pointer("Test_Patient:diagnoses:3:TABLE") is None
    assert parse_structured_evidence_pointer("patient:p:row:x:column:c") is None
    assert parse_structured_evidence_pointer("") is None


# --- :TABLE resolution verifies the structured parquet content hash ---

def _make_table_corpus(tmp_path, *, drug):
    root = tmp_path / "patient"
    structured = root / "structured"
    structured.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["seed"]}).write_parquet(root / "chunk_index.parquet")
    pq = structured / "diag.parquet"
    pl.DataFrame({"drug": [drug]}).write_parquet(pq)
    (structured / "diag.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": "diag.csv", "parquet_content_hash": hash_file(pq)}})
    )
    return root


def test_table_resolution_ok_when_parquet_matches_sidecar(tmp_path):
    store = PatientChunkStore(patient_root=_make_table_corpus(tmp_path, drug="aspirin"))
    assert store.text_for("p:diag:0:TABLE") == "drug: aspirin"


def test_table_resolution_raises_when_parquet_changed_since_ingest(tmp_path):
    root = _make_table_corpus(tmp_path, drug="aspirin")
    # tamper: rewrite the parquet so its content hash no longer matches the sidecar
    pl.DataFrame({"drug": ["tampered"]}).write_parquet(root / "structured" / "diag.parquet")
    store = PatientChunkStore(patient_root=root)
    with pytest.raises(RuntimeError, match="changed since ingest"):
        store.text_for("p:diag:0:TABLE")
