"""Builds the vector-search index for one patient's embedded text chunks.

This step reads the embeddings.npy file produced by step 2 and writes hnsw.bin —
the HNSW index (via hnswlib) that step 4 queries. Each patient gets their own
independent index.

Within a single patient, vectors are inserted on one thread with a fixed random seed.
This makes the saved index bytes identical from run to run, which the audit trail
needs in order to cross-link correctly. The per-patient chunk count is small enough
that single-threaded insertion costs less than a second.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.config_loading import validate_index_config
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    ensure_layout,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_log = get_logger("index")

# ── index config defaults ─────────────────────────────────────────────────────
# All keys below sit under the "index" key in the project cfg dict.
# Set them explicitly in your deployment config to override.
#
#   index.M                number of bidirectional links per vector in the graph;
#                          higher = better recall, more memory, slower build;
#                          hnswlib recommends 16-64, 32 is a good middle ground
#   index.ef_construction  how thoroughly the graph is explored when inserting
#                          each vector during build; higher = better quality,
#                          slower build; 200 is the hnswlib recommended default
#   index.space            similarity metric: "ip" (inner product) works because
#                          all vectors are length-normalized at embed time;
#                          use "cosine" or "l2" if you switch to a non-normalized encoder
#   index.random_seed      fixed integer passed to hnswlib — the value doesn't
#                          matter, only that it never changes between runs
_DEFAULT_M = 32
_DEFAULT_EF_CONSTRUCTION = 200
_DEFAULT_SPACE = "ip"
_DEFAULT_RANDOM_SEED = 100

class AnnIndex(Protocol):
    """Interface any index backend must satisfy — build, save, and report size."""

    kind: str

    def build(self, vectors: np.ndarray, *, space: str, m: int, ef_construction: int) -> None: ...
    def save(self, path: Path) -> None: ...
    def size(self) -> int: ...

@dataclass
class HnswlibIndex:
    """HNSW index backed by the hnswlib library. Each vector's ID is simply its
    row position in embeddings.npy (0, 1, 2 ...)."""

    kind: str = "hnswlib"
    # arbitrary fixed value — the specific number doesn't matter, only that it never changes.
    random_seed: int = _DEFAULT_RANDOM_SEED
    _index: Any = None

    def build(self, vectors: np.ndarray, *, space: str, m: int, ef_construction: int) -> None:
        import hnswlib

        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32, copy=False)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D; got shape {vectors.shape}")
        n, dim = vectors.shape
        idx = hnswlib.Index(space=space, dim=dim)
        # set_num_threads(1) before init_index pins both build and query to a single
        # thread -- with multiple threads hnswlib inserts vectors in an unpredictable
        # order, which would make the saved index bytes differ from run to run.
        idx.set_num_threads(1)
        idx.init_index(
            max_elements=max(n, 1),
            ef_construction=ef_construction,
            M=m,
            random_seed=int(self.random_seed),
        )
        if n > 0:
            idx.add_items(vectors, np.arange(n, dtype=np.int64), num_threads=1)
        self._index = idx

    def save(self, path: Path) -> None:
        if self._index is None:
            raise RuntimeError("Index has not been built")
        tmp = path.with_suffix(path.suffix + ".tmp")
        self._index.save_index(str(tmp))
        tmp.rename(path)

    def size(self) -> int:
        return int(self._index.get_current_count()) if self._index is not None else 0

def build_index(cfg: dict) -> AnnIndex:
    """construct an AnnIndex from cfg; only hnswlib is wired up today."""
    kind = cfg.get("kind", "hnswlib")
    if kind == "hnswlib":
        return HnswlibIndex(random_seed=int(cfg.get("random_seed", _DEFAULT_RANDOM_SEED)))
    raise ValueError(f"Unknown index.kind: {kind!r}")

def _cached_index_meta(hnsw_meta_path: Path) -> dict | None:
    """The index_meta payload from the hnsw companion metadata file (sidecar), or None
    if it predates this format or is unreadable."""
    try:
        env = json.loads(hnsw_meta_path.read_text(encoding="utf-8"))
        return env["payload"]["index_meta"]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


# The hnsw.bin sidecar records what the index was built from. These fields must
# all be present AND match to reuse a cached index. Missing any one means we cannot
# prove the cached hnsw.bin was built from the current embeddings and the current
# index config.
_INDEX_CACHE_KEY_FIELDS = (
    "embeddings_sha256", "index_kind", "space", "M", "ef_construction", "random_seed",
)


def _index_cache_decision(
    *,
    sidecar_meta: dict | None,
    resolved: dict,
    current_emb_hash: str,
    source_present: bool,
) -> str:
    """Decide whether to reuse, rebuild, or refuse: returns 'skip' | 'rebuild' | 'fail'.

    'skip' (reuse the cached index) only when the sidecar proves the cached index was
    built from the current embeddings AND the current index config
    (kind/space/M/ef_construction/seed). Otherwise -- or if the sidecar predates this
    format / is missing any proof field -- rebuild from the source embeddings when they
    are present. Only when a rebuild is impossible (the source embeddings are gone) do
    we 'fail' loudly rather than serve an index we cannot vouch for.
    """
    if not sidecar_meta or any(f not in sidecar_meta for f in _INDEX_CACHE_KEY_FIELDS):
        return "rebuild" if source_present else "fail"
    matches = (
        sidecar_meta["embeddings_sha256"] == current_emb_hash
        and str(sidecar_meta["index_kind"]) == str(resolved["kind"])
        and str(sidecar_meta["space"]) == str(resolved["space"])
        and int(sidecar_meta["M"]) == int(resolved["M"])
        and int(sidecar_meta["ef_construction"]) == int(resolved["ef_construction"])
        and int(sidecar_meta["random_seed"]) == int(resolved["random_seed"])
    )
    if matches:
        return "skip"
    return "rebuild" if source_present else "fail"


def run_index_one(
    *,
    cfg: dict,
    patient_id: str,
    code_lock_hash: str | None = None,
    force: bool = False,
) -> dict:
    """Build (or reuse) the HNSW search index for one patient and write its companion
    metadata file (sidecar)."""
    validate_index_config(cfg)

    run_id = cfg["run_id"]
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)

    patient_out = phi_patient_run_dir(run_id, patient_id)
    emb_path = patient_out / "embeddings.npy"
    if not emb_path.is_file():
        raise FileNotFoundError(f"Missing embeddings for {patient_id!r}; run embed first.")
    chunk_index_path = patient_out / "chunk_index.parquet"
    if not chunk_index_path.is_file():
        raise FileNotFoundError(
            f"Missing chunk index for {patient_id!r}; run embed first."
        )

    hnsw_path = patient_out / "hnsw.bin"
    hnsw_meta_path = patient_out / "hnsw.bin.meta.json"

    log = _log.bind(run_id=run_id, patient_id=patient_id)

    vector_shape = np.load(emb_path, allow_pickle=False, mmap_mode="r").shape
    if len(vector_shape) != 2:
        raise ValueError(f"embeddings.npy must be 2-D; got shape {vector_shape}")
    chunk_rows = int(pl.read_parquet(chunk_index_path).height)
    if int(vector_shape[0]) != chunk_rows:
        raise RuntimeError(
            f"Embedding/chunk-index row mismatch for {patient_id!r}: "
            f"embeddings.npy has {vector_shape[0]} rows, "
            f"chunk_index.parquet has {chunk_rows} rows. Re-run embed."
        )

    # Reuse the cached index only when the sidecar proves it was built from the current
    # embeddings AND the current index config (space/M/ef_construction/seed); otherwise
    # rebuild. The content hash below detects any change to the embeddings, which forces
    # a full rebuild.
    current_emb_hash = hash_file(emb_path)
    idx_cfg = cfg.get("index") or {}
    space = idx_cfg.get("space", _DEFAULT_SPACE)
    m = int(idx_cfg.get("M", _DEFAULT_M))
    ef_construction = int(idx_cfg.get("ef_construction", _DEFAULT_EF_CONSTRUCTION))
    resolved = {
        "kind": idx_cfg.get("kind", "hnswlib"),
        "space": space,
        "M": m,
        "ef_construction": ef_construction,
        "random_seed": int(idx_cfg.get("random_seed", _DEFAULT_RANDOM_SEED)),
    }
    if not force and hnsw_path.is_file() and hnsw_meta_path.is_file():
        decision = _index_cache_decision(
            sidecar_meta=_cached_index_meta(hnsw_meta_path),
            resolved=resolved,
            current_emb_hash=current_emb_hash,
            source_present=emb_path.is_file(),
        )
        if decision == "skip":
            log.info("index_skip_cached")
            record_transition(
                run_root,
                entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="index"),
                from_state=None,
                to_state="completed",
                reason="cached: hnsw.bin built from current embeddings and index config",
                step_context="index",
                code_lock_hash=code_lock_hash,
            )
            return {"patient_id": patient_id, "cached": True}
        if decision == "fail":
            raise RuntimeError(
                f"Cannot reuse or rebuild the hnsw index for {patient_id!r}: sidecar "
                f"{hnsw_meta_path.name} lacks fields to prove compatibility and the source "
                "embeddings are unavailable to rebuild from."
            )
        log.info("index_cache_busted", extra_={"reason": "embeddings or index config changed"})

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="index"),
        from_state=None,
        to_state="running",
        reason="index_start",
        step_context="index",
        code_lock_hash=code_lock_hash,
    )

    vectors = np.load(emb_path, allow_pickle=False)

    idx = build_index(idx_cfg)
    try:
        idx.build(vectors, space=space, m=m, ef_construction=ef_construction)
        # the index must hold exactly one vector per embeddings row, or step 4's
        # mapping from an index hit back to the right chunk would be off by rows.
        # Verify before finalizing.
        if idx.size() != int(vectors.shape[0]):
            raise RuntimeError(
                f"hnsw index built {idx.size()} vectors but embeddings.npy has "
                f"{int(vectors.shape[0])} rows for patient {patient_id!r}; refusing to "
                "write a misaligned index."
            )
    except Exception as exc:
        # without this, a crash leaves the entity stuck in "running" forever.
        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="index"),
            from_state="running",
            to_state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            step_context="index",
            code_lock_hash=code_lock_hash,
        )
        raise

    # hnswlib refuses to overwrite; clear any stale file first.
    if hnsw_path.is_file():
        hnsw_path.unlink()
    idx.save(hnsw_path)

    payload = {
        "patient_id": patient_id,
        "index_kind": idx.kind,
        "space": space,
        "M": m,
        "ef_construction": ef_construction,
        "random_seed": resolved["random_seed"],
        "size": idx.size(),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "content_hash": hash_file(hnsw_path),
        "embeddings_sha256": current_emb_hash,  # content hash; the next run compares it to decide reuse vs rebuild
    }
    env = envelope_for(
        artifact_type="index_meta",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="index",
        patient_id=patient_id,
        payload={"index_meta": payload},
        code_lock_hash=code_lock_hash,
    )
    write_artifact(env, path=hnsw_meta_path)  # data-co-located sidecar (next to the index)

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="index"),
        from_state="running",
        to_state="completed",
        reason=f"hnsw built (n={idx.size()})",
        step_context="index",
        code_lock_hash=code_lock_hash,
    )
    log.info("index_done", extra_={"size": idx.size()})
    return {"patient_id": patient_id, "size": idx.size()}
