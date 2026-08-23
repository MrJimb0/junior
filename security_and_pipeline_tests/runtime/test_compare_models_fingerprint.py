"""compare-models reads a real (non-null) fingerprint.

_read_fingerprint reads the receipts at run_root/patients/... . Each compared model
runs under its own scratch run_id so it does not overwrite the sealed extract artifacts
in place.
"""
from __future__ import annotations

import json

from jr_pipeline.evaluating_pipeline_performance.compare_llm_models import (
    _materialize_scratch_corpus,
    _read_fingerprint,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


def test_read_fingerprint_finds_receipt_at_real_path(tmp_path):
    run_root = tmp_path / "run"
    steps = run_root / "patients" / "p1" / "extract" / "date_of_diagnosis" / "steps" / "s0"
    steps.mkdir(parents=True)
    (steps / "receipt.json").write_text(
        json.dumps({"payload": {"resolved_model": {"fingerprint": "fp-abc123"}}})
    )
    assert _read_fingerprint(run_root, "p1", "date_of_diagnosis") == "fp-abc123"


def test_read_fingerprint_none_when_no_receipts(tmp_path):
    assert _read_fingerprint(tmp_path / "empty", "p1", "v") is None


def test_materialize_scratch_corpus_links_inputs_and_spares_sealed_outputs(tmp_path):
    # Without materialization the scratch run has no patient dir and run_extract_one
    # raises FileNotFoundError. This proves the corpus is linked in (precondition met)
    # while the sealed extract/retrieve outputs are left untouched.
    src = phi_patient_run_dir("run0", "p1", tmp_path)
    (src / "structured").mkdir(parents=True)
    (src / "structured" / "notes.parquet").write_text("CORPUS")
    (src / "chunk_index.parquet").write_text("INDEX")
    sealed = src / "extract" / "date_of_diagnosis"
    sealed.mkdir(parents=True)
    (sealed / "result.json").write_text("SEALED")
    (src / "retrieve").mkdir()

    dst = _materialize_scratch_corpus(
        source_run_id="run0", scratch_run_id="run0__compare__m", patient_id="p1", dr=tmp_path
    )

    # scratch patient dir now exists (run_extract_one's precondition) with the corpus
    assert dst.is_dir()
    assert (dst / "chunk_index.parquet").read_text() == "INDEX"
    assert (dst / "structured" / "notes.parquet").read_text() == "CORPUS"
    # per-model outputs are NOT linked, so the sealed source artifacts can't be clobbered
    assert not (dst / "extract").exists()
    assert not (dst / "retrieve").exists()
    assert (sealed / "result.json").read_text() == "SEALED"
