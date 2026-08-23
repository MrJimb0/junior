"""Retrieves the most relevant chunks for a query from one patient's chart.

How it searches is set in config:
  - "bm25" = classic keyword / term-frequency search
  - "embedding" = similarity search over the meaning vectors (embeddings)
  - "hybrid" = run both and merge their ranked lists
  - "exact" = literal substring match
  - "direct_parquet" = look the value up straight from a structured table
Results are written to a small retrieval record alongside the patient's other
files so the extract step can read the selected chunks.
"""
from __future__ import annotations

import time

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.bm25.bm25_v2 import BM25Retriever
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.embedding.embedding_v1 import (
    EmbeddingRetriever,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.exact_text.exact_text_v1 import (
    ExactRetriever,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.hybrid.hybrid_v1 import (
    HybridRetriever,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.nonembedded_structured_parquet_table_looker_uper.direct_parquet_retriever_v1 import (
    DirectParquetRetriever,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)
from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    ensure_layout,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    Retriever,
)

_log = get_logger("retrieve")

# ── retrieve config defaults ──────────────────────────────────────────────────
# All keys below come from the project cfg dict. Set explicitly to override.
#
#   k                      number of chunks to return per query
#   retriever.kind         which search to use:
#                          "bm25" (keyword/term-frequency search) | "embedding"
#                          (meaning-vector similarity) | "exact" (literal substring
#                          match) | "hybrid" (run bm25 + embedding and merge) |
#                          "direct_parquet" (look the value up from a structured table)
#   retriever.source_file  restrict the text search to one source file
#                          (bm25/embedding/exact only); e.g. "clinical_note.csv" —
#                          omit to search every embedded file
#   retriever (bm25):
#     k1                   how quickly repeated terms stop adding weight; higher =
#                          more weight to terms that appear many times
#     b                    how much to penalize long documents; 0 = ignore length,
#                          1 = fully normalize for length
#   retriever (embedding):
#     space                the similarity measure — must match the one used when the
#                          index was built
#     ef_search            how hard the fast vector index (hnswlib) searches at query
#                          time; higher = finds more of the true closest matches but
#                          is slower
#     oversample           pull back k × oversample candidates first, then narrow to
#                          source_file — so filtering doesn't leave fewer than k
#   retriever (hybrid):
#     fusion               how to merge the two ranked lists: "reciprocal_rank" =
#                          reciprocal rank fusion (RRF), which merges by rank
#                          position and so ignores each method's raw score scale
#                          (default) | "weighted_sum" = add the raw scores with weights
#     k0                   the RRF smoothing constant; the original paper recommends 60
_DEFAULT_K = 10
_DEFAULT_RETRIEVER_KIND = "bm25"
_DEFAULT_BM25_K1 = 1.5
_DEFAULT_BM25_B = 0.75
_DEFAULT_EMBED_SPACE = "ip"
_DEFAULT_EMBED_EF_SEARCH = 80
_DEFAULT_EMBED_OVERSAMPLE = 4
_DEFAULT_HYBRID_FUSION = "reciprocal_rank"
_DEFAULT_HYBRID_K0 = 60
_DEFAULT_EXACT_CASE_INSENSITIVE = True


def build_retriever(cfg: dict, *, encoder_cfg_fallback: dict | None = None) -> Retriever:
    """Build the retriever named in the config; for "hybrid", build each member
    retriever it combines."""
    kind = cfg["kind"]
    if kind == "exact":
        return ExactRetriever(
            case_insensitive=bool(cfg.get("case_insensitive", _DEFAULT_EXACT_CASE_INSENSITIVE)),
            source_file=cfg.get("source_file"),
        )
    if kind == "bm25":
        return BM25Retriever(
            k1=float(cfg.get("k1", _DEFAULT_BM25_K1)),
            b=float(cfg.get("b", _DEFAULT_BM25_B)),
            source_file=cfg.get("source_file"),
        )
    if kind == "embedding":
        enc_cfg = cfg.get("encoder") or encoder_cfg_fallback
        if enc_cfg is None:
            raise ValueError("embedding retriever needs 'encoder' in its config")
        return EmbeddingRetriever(
            encoder_cfg=enc_cfg,
            space=cfg.get("space", _DEFAULT_EMBED_SPACE),
            ef_search=int(cfg.get("ef_search", _DEFAULT_EMBED_EF_SEARCH)),
            oversample=int(cfg.get("oversample", _DEFAULT_EMBED_OVERSAMPLE)),
            source_file=cfg.get("source_file"),
        )
    if kind == "direct_parquet":
        return DirectParquetRetriever(
            table=cfg["table"],
            filter_expr=cfg.get("filter"),
        )
    if kind == "hybrid":
        # Convenience: a recipe can write `kind: hybrid` with the same
        # query/k/source_file it used for bm25 and automatically get bm25 +
        # embedding searched over that source and their results merged — without
        # having to list the two `members` itself. An explicit `members` list, if
        # given, still takes precedence.
        member_cfgs = cfg.get("members")
        if not member_cfgs:
            sf = cfg.get("source_file")
            member_cfgs = [
                {"kind": "bm25", "source_file": sf},
                {"kind": "embedding", "source_file": sf},
            ]
        members = [
            build_retriever(m, encoder_cfg_fallback=encoder_cfg_fallback)
            for m in member_cfgs
        ]
        return HybridRetriever(
            members=members,
            fusion=cfg.get("fusion", _DEFAULT_HYBRID_FUSION),
            k0=int(cfg.get("k0", _DEFAULT_HYBRID_K0)),
            weights=cfg.get("weights"),
        )
    raise ValueError(f"Unknown retriever kind: {kind!r}")


def _retrieval_payload(
    retriever: Retriever,
    candidates: list[Candidate],
    *,
    patient_id: str,
    text: str,
    query_id: str | None,
    variable: str | None,
    elapsed: float,
) -> dict:
    """Build the contents of the retrieval record. This record is written to the
    NO_PHI stream (no protected health information), so it holds chunk ids, ranks,
    and scores — never chunk text. Split out into its own function so its contents —
    especially the scrubbed per-member hybrid outcomes — can be unit-tested without
    building a full patient corpus."""
    return {
        "patient_id": patient_id,
        "query": {"text": text, "query_id": query_id, "variable": variable},
        "retriever": {
            "kind": retriever.info.kind,
            "version": retriever.info.version,
            "config": retriever.info.config,
        },
        "candidates": [
            {
                "chunk_id": c.chunk_id,
                "rank": c.rank,
                "score": c.score,
                "selected": True,
                "retriever": c.retriever,
            }
            for c in candidates
        ],
        "selected_chunk_ids": [c.chunk_id for c in candidates],
        "fusion_strategy": (
            retriever.info.config.get("fusion") if retriever.info.kind == "hybrid" else None
        ),
        "score_normalization": retriever.score_normalization,
        # record the scrubbed per-member outcomes (only an error category and a HASH
        # of any message — never raw text that might contain protected health
        # information) into the NO_PHI retrieval record. For hybrid search,
        # HybridRetriever fills .last_member_outcomes during query(); the other
        # retrievers have no members, so this stays None.
        "member_outcomes": getattr(retriever, "last_member_outcomes", None),
        "timings": {"query_s": round(elapsed, 6)},
    }


def retrieve_one(
    *,
    cfg: dict,
    patient_id: str,
    text: str,
    query_id: str | None = None,
    variable: str | None = None,
    code_lock_hash: str | None = None,
) -> dict:
    """Run one retrieval query for one patient; return and save the retrieval record."""
    run_id = cfg["run_id"]
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)

    patient_out = phi_patient_run_dir(run_id, patient_id)
    if not patient_out.is_dir():
        raise FileNotFoundError(f"No patient dir for {patient_id!r}; run ingest + embed first")

    corpus = PatientChunkStore(patient_root=patient_out)
    # bm25 keyword search is the default fallback — it needs no model and no vector index.
    retriever = build_retriever(
        cfg.get("retriever") or {"kind": _DEFAULT_RETRIEVER_KIND},
        encoder_cfg_fallback=cfg.get("encoder"),
    )

    k = int(cfg.get("k", _DEFAULT_K))

    log = _log.bind(run_id=run_id, patient_id=patient_id)
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="retrieve"),
        from_state=None,
        to_state="running",
        reason=f"retrieve kind={retriever.info.kind} k={k}",
        step_context="retrieve",
        code_lock_hash=code_lock_hash,
    )
    t0 = time.perf_counter()
    try:
        candidates: list[Candidate] = retriever.query(corpus, text=text, k=k)
    except Exception as exc:
        # without this, a crash leaves the step stuck in the "running" state forever.
        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="retrieve"),
            from_state="running",
            to_state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            step_context="retrieve",
            code_lock_hash=code_lock_hash,
        )
        raise
    elapsed = time.perf_counter() - t0

    payload = _retrieval_payload(
        retriever,
        candidates,
        patient_id=patient_id,
        text=text,
        query_id=query_id,
        variable=variable,
        elapsed=elapsed,
    )
    env = envelope_for(
        artifact_type="retrieval",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="retrieve",
        patient_id=patient_id,
        variable=variable,
        payload=payload,
        code_lock_hash=code_lock_hash,
    )
    env["content_hash"] = hash_artifact_payload(env)
    validate_artifact(env, "retrieval")

    if cfg.get("write", True):  # always validate above; only the file path is decided here (and the path comes from the artifact registry)
        atomic_write_json(artifact_path("retrieval", patient_root=patient_out, query_id=query_id), env)

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="retrieve"),
        from_state="running",
        to_state="completed",
        reason=f"candidates={len(candidates)}",
        step_context="retrieve",
        code_lock_hash=code_lock_hash,
    )
    log.info(
        "retrieve_done",
        extra_={
            "kind": retriever.info.kind,
            "k": k,
            "hits": len(candidates),
            "seconds": round(elapsed, 4),
        },
    )
    return env

def run_retrieve_one(
    *,
    cfg: dict,
    patient_id: str,
    code_lock_hash: str | None = None,
    force: bool = False,  # noqa: ARG001 — kept for STAGES-dispatch uniformity
) -> dict:
    """Command-line entry point for the retrieve step. `force` is ignored here; it
    only exists so every step's run function takes the same arguments (the STAGES
    dispatch contract)."""
    q = cfg.get("query") or {}
    text = q.get("text")
    if not text:
        raise ValueError("retrieve config must include query.text")
    return retrieve_one(
        cfg=cfg,
        patient_id=patient_id,
        text=text,
        query_id=q.get("query_id"),
        variable=q.get("variable"),
        code_lock_hash=code_lock_hash,
    )
