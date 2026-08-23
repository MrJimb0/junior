"""Builders for this run's NO_PHI exhaust records — the de-identified training signal
the pipeline throws off as a byproduct, safe to pool across hospitals.

For each retrieve_and_prompt step the spine knows, per candidate chunk, *why* it was
(or was not) put in the evidence packet: which retriever surfaced it, its rank
before/after rerank, its score, its size, its document type, whether it made the
bundle, and — once the extractor answers — whether the model *cited* it. That is a
learnable relevance signal, but the PHI-side trace keys on chunk_ids (which embed the
patient id), so it cannot leave the institution.

This module builds the parallel NO_PHI records — the per-chunk ``selection_judgment``
ranked list, the model-free ``clinical_invariant`` weak labels, the run's single
``method_provenance`` identity record, and the human-review records a reviewer's
workbench feeds in later (``relevance_label`` = a relevant/not_relevant/unsure verdict on
a candidate chunk; ``retriever_miss`` = relevant evidence the retriever missed). The
human records mint the chunk surrogate in the shared evidence-chunk domain so a label
joins back to that chunk's ``selection_judgment`` ranks. Every identifier is replaced by a
**domain-separated HMAC surrogate** (``surrogates.make_surrogate`` with the PHI-side
run secret), every categorical is gated through a controlled vocabulary, and the whole
record is handed to ``emit()`` — which validates it against its schema, runs the
forbidden-content scan, and appends it to a process-unique shard. Nothing here writes
chunk text, real chunk ids, absolute dates, or patient ids.

Every record carries the **frozen common header**: schema/vocab/site/study/run identity
+ method fingerprints, so rows stay attributable and poolable across runs, recipe edits,
encoder changes, and sites.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.exhaust.schema import SCHEMA_VERSION
from jr_pipeline.runtime_infrastructure.exhaust.secret_lifecycle import run_secret
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import (
    EVIDENCE_CHUNK_LINKAGE_DOMAIN,
    LinkageScope,
    make_surrogate,
)
from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    VOCAB_VERSION,
    HumanTier,
    RelevanceLabel,
    canonicalize_doc_type,
    is_unmapped,
)
from jr_pipeline.runtime_infrastructure.exhaust.writer import emit

_RUN = LinkageScope.RUN.value
_UNKNOWN = "unknown"


def site_id() -> str:
    """Opaque per-institution id assigned at deployment (config, never derived from
    PHI). Identifies an institution, never a patient."""
    return os.environ.get("JR_SITE_ID", "unset_site")


def study_id() -> str:
    """Opaque per-study id (config, never derived from PHI). Lets rows pool within a
    study; defaults to ``unset_study`` for a dev run."""
    return os.environ.get("JR_STUDY_ID", "unset_study")


def fingerprint(obj: Any) -> str:
    """Stable short fingerprint of a config/info object (canonical JSON -> sha256)."""
    canonical = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _model_identity(model_id: Any) -> str | None:
    """Name the model in a way that survives leaving this machine.

    A local checkout says "./models/embedding/thomas-sounack:BioClinical-ModernBERT-base23APR2026";
    a cluster says "/shared/<lab>/models/thomas-sounack:BioClinical-ModernBERT-base23APR2026".
    The directory differs, the name does not, and the name is what the HuggingFace repo
    was called — so for a path, the last component IS the model identity.

    A hub id ("NeuML/bioclinical-modernbert-base-embeddings") is already a name rather
    than a location and is kept whole; taking its last component would drop the
    publisher and let two organisations' models collide."""
    if not isinstance(model_id, str) or not model_id.strip():
        return None
    value = model_id.strip()
    looks_like_a_path = value.startswith(("/", "./", "../", "~")) or "\\" in value
    return value.rsplit("/", 1)[-1] if looks_like_a_path else value


def encoder_fingerprint(encoder_cfg: dict | None) -> str:
    """Identify the encoder for the shareable exhaust, portably.

    This value exists to be POOLED across institutions -- that is the whole point of the
    NO_PHI stream -- so it must be equal for the same encoder wherever it ran, and
    different for a different one. Hashing the config verbatim failed both halves.

    Equal-wherever-it-ran: `junior run` resolves model_id to an absolute path before the
    config reaches here while the per-stage commands hash the raw relative string, so one
    laptop produced two identities for one settings file depending on which command was
    typed -- and the cohort one wrote a home directory into a value meant to be shared.
    A verbatim hash also folds in `device`, so the same encoder on a laptop and on a
    cluster could never agree.

    Different-for-a-different-one: dropping the path outright is not enough either. Two
    unrelated models sharing pooling/max_tokens/dtype would then fingerprint the same,
    and pooled data from two encoders would merge silently -- worse than the problem.

    So this is an ALLOW-list of the fields the encoder itself declares vector-affecting
    (VECTOR_AFFECTING_FINGERPRINT_FIELDS), with model_id reduced to a portable name.
    Anything else in the config -- device, local_files_only, cache directories -- is a
    fact about a machine and is left out on purpose."""
    from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
        VECTOR_AFFECTING_FINGERPRINT_FIELDS,
    )

    cfg = encoder_cfg or {}
    identity: dict[str, Any] = {}
    for field in VECTOR_AFFECTING_FINGERPRINT_FIELDS:
        if field == "model_id":
            named = _model_identity(cfg.get("model_id"))
            if named is not None:
                identity["model"] = named
            continue
        if cfg.get(field) is not None:
            identity[field] = cfg[field]
    # A pinned weight hash is the strongest identity there is, and fully portable.
    pins = cfg.get("expected_file_sha256")
    if isinstance(pins, dict) and pins.get("model.safetensors"):
        identity["model_sha256"] = pins["model.safetensors"]
    return fingerprint(identity)


def _emitted_month(run_id: str) -> str:
    """Month-resolution stamp (e.g. ``2026-06``) — never finer than month, to avoid
    leaking a full date. Derived from the timestamped run_id so it is deterministic
    and offline-safe."""
    digits = "".join(c for c in run_id if c.isdigit())
    if len(digits) >= 6:
        return f"{digits[:4]}-{digits[4:6]}"
    return _UNKNOWN


def _run_header(
    *,
    run_id: str,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
) -> dict[str, Any]:
    """The run-level identity every exhaust record carries (the RunHeader fields)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "vocab_version": VOCAB_VERSION,
        "site_id": site_id(),
        "study_id": study_id(),
        "run_id": run_id,
        "emitted_month": _emitted_month(run_id),
        "encoder_fingerprint": encoder_fingerprint,
        "reranker_fingerprint": reranker_fingerprint,
        "extractor_fingerprint": extractor_fingerprint,
        "code_lock_hash": code_lock_hash,
        "chunking_config_id": chunking_config_id,
    }


def common_header(
    *,
    run_id: str,
    recipe_id: str,
    recipe_version: str,
    archetype: str | None,
    encoder_fingerprint: str,
    reranker_fingerprint: str,
    chunking_config_id: str,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    prompt_hash: str = _UNKNOWN,
    schema_hash: str = _UNKNOWN,
) -> dict[str, Any]:
    """The recipe-level identity header (RunHeader + the recipe the row answers)."""
    return {
        **_run_header(
            run_id=run_id,
            encoder_fingerprint=encoder_fingerprint,
            reranker_fingerprint=reranker_fingerprint,
            extractor_fingerprint=extractor_fingerprint,
            code_lock_hash=code_lock_hash,
            chunking_config_id=chunking_config_id,
        ),
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "variable_key": recipe_id,  # the question id; recipe_id is the variable name today
        "archetype": archetype or _UNKNOWN,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
    }


def record_selection_trace(
    *,
    run_id: str,
    patient_id: str,
    recipe_id: str,
    step_id: str,
    archetype: str | None,
    retriever_kind: str,
    total_evidence_tokens: int | None,
    ranked: list[dict[str, Any]],
    recipe_version: str = _UNKNOWN,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    prompt_hash: str = _UNKNOWN,
    schema_hash: str = _UNKNOWN,
    cited_chunk_ids: list[str] | None = None,
    data_root: Path | None = None,
) -> Path:
    """Build + emit one NO_PHI ranked-list record for this (patient, recipe, step).

    ``ranked`` is the post-rerank candidate list; each item carries the real
    ``chunk_id`` (replaced by an HMAC surrogate here, never written), plus
    ``doc_type``, ``rank``, ``prior_rank``, ``score``, ``token_count`` and
    ``in_bundle``. ``cited_chunk_ids`` are the chunks the extractor named as evidence —
    an independent relevance weak label. Returns the shard path ``emit`` wrote to.
    """
    secret = run_secret(run_id, data_root)

    def _surrogate(domain: str, output_field: str, raw: str) -> str:
        return make_surrogate(
            secret, scope=_RUN, record_type=domain, output_field=output_field, raw=raw
        )

    # ``group`` stays selection_judgment-scoped (it only groups this trace's rows). The
    # per-chunk surrogate is minted in the SHARED evidence-chunk domain so a later human
    # relevance_label / retriever_miss for the same chunk_id resolves to the SAME
    # surrogate and joins back to this row's retriever/cross-encoder ranks.
    group = _surrogate("selection_judgment", "group", f"{patient_id}|{recipe_id}|{step_id}")
    cited = {str(c) for c in (cited_chunk_ids or [])}

    rows: list[dict[str, Any]] = []
    n_doc_type_unmapped = 0
    for item in ranked:
        chunk_id = str(item["chunk_id"])
        raw_doc_type = item.get("doc_type")
        if is_unmapped(raw_doc_type):
            n_doc_type_unmapped += 1
        rows.append({
            "chunk_surrogate": _surrogate(EVIDENCE_CHUNK_LINKAGE_DOMAIN, "chunk_surrogate", chunk_id),
            "doc_type": canonicalize_doc_type(raw_doc_type),  # controlled vocab, never raw
            "rank": item.get("rank"),
            "prior_rank": item.get("prior_rank"),
            "score": item.get("score"),
            "token_count": item.get("token_count"),
            "in_bundle": bool(item.get("in_bundle")),
            "cited": chunk_id in cited,  # model named it as evidence (weak relevance +)
            # reranker per-candidate features (combined-score vector /
            # cross_encoder_logit) -- the reranker-optimization signal. Numbers only.
            "features": {k: float(v) for k, v in (item.get("features") or {}).items()},
        })

    record = {
        **common_header(
            run_id=run_id, recipe_id=recipe_id, recipe_version=recipe_version, archetype=archetype,
            encoder_fingerprint=encoder_fingerprint, reranker_fingerprint=reranker_fingerprint,
            chunking_config_id=chunking_config_id, extractor_fingerprint=extractor_fingerprint,
            code_lock_hash=code_lock_hash, prompt_hash=prompt_hash, schema_hash=schema_hash,
        ),
        "record_type": "selection_judgment",
        "group": group,
        "step_id": step_id,
        "retriever": retriever_kind,
        "total_evidence_tokens": total_evidence_tokens,
        "n_candidates": len(rows),
        "n_in_bundle": sum(1 for r in rows if r["in_bundle"]),
        "n_cited": sum(1 for r in rows if r["cited"]),
        # durable counter so a flood of unrecognised doc_types is visible, not hidden.
        "n_doc_type_unmapped": n_doc_type_unmapped,
        "rows": rows,
    }
    return emit("selection_judgment", record, run_id=run_id, data_root=data_root)


def record_invariant_outcomes(
    *,
    run_id: str,
    patient_id: str,
    outcomes: list[dict[str, Any]],
    code_lock_hash: str = _UNKNOWN,
    data_root: Path | None = None,
) -> Path:
    """Build + emit the clinical-invariant outcomes as a text-free NO_PHI weak-label
    record: an independent, model-free labeler for the Snorkel layer.

    Only ``rule_id`` + ``ok`` + ``severity`` (all controlled / boolean) travel — the
    rule ``context`` carries dates (HIPAA identifiers) and is NEVER emitted. The
    patient id becomes an HMAC surrogate. Returns the shard path.
    """
    record = {
        **_run_header(run_id=run_id, code_lock_hash=code_lock_hash),
        "record_type": "clinical_invariant",
        "labeler_id": "clinical_invariants/cancer_staging",
        "labeler_version": "v1",
        "outcomes": [
            {"rule_id": o.get("rule_id"), "ok": bool(o.get("ok")), "severity": o.get("severity")}
            for o in outcomes
        ],
        "n_violations": sum(1 for o in outcomes if not o.get("ok")),
    }
    return emit(
        "clinical_invariant", record, run_id=run_id,
        phi_keys={"patient_surrogate": patient_id}, data_root=data_root,
    )


def record_method_provenance(
    run_id: str,
    *,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    data_root: Path | None = None,
) -> Path:
    """Emit the run's single method_provenance record: the RunHeader identity
    (no patient data) that every other exhaust record's fingerprints tie back to -- the
    flywheel's join key for "which method produced these rows". Returns the shard path."""
    record = {
        **_run_header(
            run_id=run_id,
            encoder_fingerprint=encoder_fingerprint,
            reranker_fingerprint=reranker_fingerprint,
            extractor_fingerprint=extractor_fingerprint,
            code_lock_hash=code_lock_hash,
            chunking_config_id=chunking_config_id,
        ),
        "record_type": "method_provenance",
    }
    return emit("method_provenance", record, run_id=run_id, data_root=data_root)


# ── Human-review records (the reviewer-workbench signal) ─────────────────────
# A reviewer labels candidate chunks AFTER a run, so these are emitted by a later
# review process that pins the same run_id + data_root; run_secret(run_id) is
# re-derivable PHI-side, so the surrogates match the run's selection_judgment rows and
# finalize (idempotent) folds them into the same parquet on a re-run.
_VALID_TIERS = frozenset(t.value for t in HumanTier)
_VALID_RELEVANCE_LABELS = frozenset(m.value for m in RelevanceLabel)


def _reviewer_tier(raw: str | None) -> str:
    """Coerce a reviewer tier to a HumanTier member; an unrecognized one is ``unknown``
    (mirrors the Shiny reviewer-tier resolution -- a bad tier degrades, never raises)."""
    tier = (raw or "unknown").strip().lower()
    return tier if tier in _VALID_TIERS else HumanTier.UNKNOWN.value


def _relevance_label(raw: str) -> str:
    """Normalize + validate a relevance label, failing closed on an unknown value (a
    silently-defaulted clinical label would be a wrong training target)."""
    label = (raw or "").strip().lower()
    if label not in _VALID_RELEVANCE_LABELS:
        raise ValueError(
            f"relevance label must be one of {sorted(_VALID_RELEVANCE_LABELS)}, got {raw!r}"
        )
    return label


def _review_header(
    *,
    run_id: str,
    recipe_id: str,
    recipe_version: str,
    step_id: str,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
) -> dict[str, Any]:
    """The ReviewHeader identity (RunHeader + the recipe/step the human reviewed)."""
    return {
        **_run_header(
            run_id=run_id,
            encoder_fingerprint=encoder_fingerprint,
            reranker_fingerprint=reranker_fingerprint,
            extractor_fingerprint=extractor_fingerprint,
            code_lock_hash=code_lock_hash,
            chunking_config_id=chunking_config_id,
        ),
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
        "step_id": step_id,
    }


def record_relevance_label(
    *,
    run_id: str,
    patient_id: str,
    reviewer_id: str,
    chunk_id: str,
    label: str,
    recipe_id: str,
    step_id: str,
    reviewer_tier: str | None = None,
    recipe_version: str = _UNKNOWN,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    data_root: Path | None = None,
) -> Path:
    """Build + emit one NO_PHI ``relevance_label`` record: a reviewer's relevant /
    not_relevant / unsure verdict on one candidate chunk, de-identified.

    ``chunk_id`` MUST be the same id the ``selection_judgment`` trace recorded for this
    (run, recipe, step); its surrogate is minted in the shared evidence-chunk domain so
    the label joins back to that chunk's retriever/cross-encoder ranks.
    ``patient_id``/``reviewer_id`` become per-record-type surrogates (no cross-type
    patient join). ``label`` must be a RelevanceLabel member; an out-of-vocab tier
    degrades to ``unknown``. Returns the shard path ``emit`` wrote to."""
    record = {
        **_review_header(
            run_id=run_id, recipe_id=recipe_id, recipe_version=recipe_version, step_id=step_id,
            encoder_fingerprint=encoder_fingerprint, reranker_fingerprint=reranker_fingerprint,
            extractor_fingerprint=extractor_fingerprint, code_lock_hash=code_lock_hash,
            chunking_config_id=chunking_config_id,
        ),
        "record_type": "relevance_label",
        "reviewer_tier": _reviewer_tier(reviewer_tier),
        "label": _relevance_label(label),
    }
    return emit(
        "relevance_label", record, run_id=run_id,
        phi_keys={
            "patient_surrogate": patient_id,
            "reviewer_surrogate": reviewer_id,
            "chunk_surrogate": chunk_id,
        },
        linkage_domains={"chunk_surrogate": EVIDENCE_CHUNK_LINKAGE_DOMAIN},
        data_root=data_root,
    )


def record_retriever_miss(
    *,
    run_id: str,
    patient_id: str,
    reviewer_id: str,
    selected_chunk_id: str,
    was_in_retriever_pool: bool,
    recipe_id: str,
    step_id: str,
    reviewer_tier: str | None = None,
    retriever_rank_if_present: int | None = None,
    cross_encoder_rank_if_present: int | None = None,
    manual_search_used: bool = True,
    recipe_version: str = _UNKNOWN,
    encoder_fingerprint: str = _UNKNOWN,
    reranker_fingerprint: str = _UNKNOWN,
    chunking_config_id: str = _UNKNOWN,
    extractor_fingerprint: str = _UNKNOWN,
    code_lock_hash: str = _UNKNOWN,
    data_root: Path | None = None,
) -> Path:
    """Build + emit one NO_PHI ``retriever_miss`` record: a reviewer selected relevant
    evidence the retriever missed or buried.

    Only the structural outcome travels -- the manual search *query* is free text and
    stays PHI-side, never here. ``selected_chunk_id``'s surrogate uses the shared
    evidence-chunk domain, so when ``was_in_retriever_pool`` is True it joins to that
    chunk's ``selection_judgment`` row and the recorded ranks attribute the miss to the
    cross-encoder; when False, retrieval/chunking is the bottleneck. Returns the shard
    path."""
    record = {
        **_review_header(
            run_id=run_id, recipe_id=recipe_id, recipe_version=recipe_version, step_id=step_id,
            encoder_fingerprint=encoder_fingerprint, reranker_fingerprint=reranker_fingerprint,
            extractor_fingerprint=extractor_fingerprint, code_lock_hash=code_lock_hash,
            chunking_config_id=chunking_config_id,
        ),
        "record_type": "retriever_miss",
        "reviewer_tier": _reviewer_tier(reviewer_tier),
        "was_in_retriever_pool": bool(was_in_retriever_pool),
        "retriever_rank_if_present": retriever_rank_if_present,
        "cross_encoder_rank_if_present": cross_encoder_rank_if_present,
        "manual_search_used": bool(manual_search_used),
    }
    return emit(
        "retriever_miss", record, run_id=run_id,
        phi_keys={
            "patient_surrogate": patient_id,
            "reviewer_surrogate": reviewer_id,
            "chunk_surrogate": selected_chunk_id,
        },
        linkage_domains={"chunk_surrogate": EVIDENCE_CHUNK_LINKAGE_DOMAIN},
        data_root=data_root,
    )
