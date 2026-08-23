"""The hnsw cache key includes the index config, not just the embeddings hash.

If it only compared embeddings_sha256, changing index.space/M/ef_construction would
serve a stale index. The decision checks kind/space/M/ef_construction/seed too; a legacy
sidecar missing a proof field rebuilds when the source embeddings are present, and fails
loud only when a rebuild is impossible.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_3_build_vector_index.build_index import (
    _index_cache_decision,
    run_index_one,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)

_SIDECAR = {"embeddings_sha256": "h", "index_kind": "hnswlib", "space": "ip",
            "M": 32, "ef_construction": 200, "random_seed": 100}
_RESOLVED = {"kind": "hnswlib", "space": "ip", "M": 32, "ef_construction": 200, "random_seed": 100}


def test_decision_skip_when_everything_matches():
    assert _index_cache_decision(sidecar_meta=dict(_SIDECAR), resolved=_RESOLVED,
                                 current_emb_hash="h", source_present=True) == "skip"


def test_decision_rebuild_when_M_changed():
    assert _index_cache_decision(sidecar_meta=dict(_SIDECAR), resolved={**_RESOLVED, "M": 64},
                                 current_emb_hash="h", source_present=True) == "rebuild"


def test_decision_rebuild_when_embeddings_changed():
    assert _index_cache_decision(sidecar_meta=dict(_SIDECAR), resolved=_RESOLVED,
                                 current_emb_hash="DIFFERENT", source_present=True) == "rebuild"


def test_decision_legacy_sidecar_missing_seed_rebuilds_when_source_present():
    legacy = {k: v for k, v in _SIDECAR.items() if k != "random_seed"}
    assert _index_cache_decision(sidecar_meta=legacy, resolved=_RESOLVED,
                                 current_emb_hash="h", source_present=True) == "rebuild"


def test_decision_legacy_sidecar_fails_loud_when_source_gone():
    legacy = {k: v for k, v in _SIDECAR.items() if k != "random_seed"}
    assert _index_cache_decision(sidecar_meta=legacy, resolved=_RESOLVED,
                                 current_emb_hash="h", source_present=False) == "fail"


def _seed_patient(tmp_path, monkeypatch, n=6, dim=8):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    run_id, patient_id = "20260101_index", "Test_Patient"
    out = phi_patient_run_dir(run_id, patient_id)
    out.mkdir(parents=True, exist_ok=True)
    mat = np.random.default_rng(0).standard_normal((n, dim)).astype("float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    np.save(out / "embeddings.npy", mat)
    pl.DataFrame({"chunk_id": [f"c{i}" for i in range(n)]}).write_parquet(out / "chunk_index.parquet")
    return run_id, patient_id


def test_run_index_one_rebuilds_when_M_changes(tmp_path, monkeypatch):
    run_id, patient_id = _seed_patient(tmp_path, monkeypatch)
    cfg16 = {"run_id": run_id, "index": {"M": 16, "ef_construction": 200, "space": "ip"}}
    run_index_one(cfg=cfg16, patient_id=patient_id)
    assert run_index_one(cfg=cfg16, patient_id=patient_id).get("cached") is True
    cfg32 = {"run_id": run_id, "index": {"M": 32, "ef_construction": 200, "space": "ip"}}
    rebuilt = run_index_one(cfg=cfg32, patient_id=patient_id)
    assert rebuilt.get("cached") is not True
    assert "size" in rebuilt
