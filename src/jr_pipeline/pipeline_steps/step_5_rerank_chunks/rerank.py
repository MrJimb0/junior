"""Builds the step-5 reranker from a recipe's reranking config.

Reranking is the second-pass step that re-orders the chunks the search step
returned, so the most relevant evidence ends up on top before it is handed to the
language model. Step 5 is one candidate ranker with two parts:

  * a cross-encoder (a small model that scores each question-and-chunk pair),
    turned on or off per recipe with ``cross_encoder: true|false``;
  * everything else — metadata filtering, duplicate-text removal, the rule-based
    combined score, date-ordered selection, and the optional re-sort into reading
    order — in ``CandidateRanker``.

This file is just the shell: it reads the config, resolves the cross-encoder
model from the registry when it is switched on, and hands both to the candidate
ranker. The reranking itself runs inside step 7 (retrieve_and_prompt), which
searches -> reranks -> fits the evidence in one pass.

Config keys (all optional unless noted):
  cross_encoder    true to build the cross-encoder scorer; false (default) for no
                   model at all.
  model_id         models_registry key for the cross-encoder; required when
                   cross_encoder is true.
  rank_by          which scorer SELECTS the top_n: "cross_encoder",
                   "combined_score", "newest_documents" or "oldest_documents".
                   Unset means the cross-encoder when one was built, else the
                   combined score.
  dedup_identical_text  drop candidates repeating text already kept (default true).
  resort_by_date   true to lay the chosen chunks out newest-document-first.
  chronological_order  explicit "newest_first" or "oldest_first" reading order;
                   supersedes resort_by_date when set. Independent of rank_by —
                   selection order and reading order are separate decisions.
  weights          combined-score signal weights (combined score only).
  source_priority  per-source-file weight (combined score only).
  k0               rank-smoothing constant for the combined score (default 60).
  max_length / batch_size / device   cross-encoder runtime knobs.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import (
    CandidateRanker,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    CandidateScorer,
)

# k0 is the smoothing constant used when turning a search rank into a score
# (score = 1 / (k0 + rank)). A larger k0 flattens the gap between ranks. 60 is
# the conventional default for reciprocal rank fusion (RRF = merging ranked lists
# by their rank position rather than their raw scores).
_DEFAULT_K0 = 60


def build_reranker(cfg: dict) -> CandidateRanker:
    """Return a CandidateRanker configured by ``cfg`` (the recipe's reranking block)."""
    cross_encoder_scorer = None
    if cfg.get("cross_encoder"):
        cross_encoder_scorer = _build_cross_encoder(cfg)

    return CandidateRanker(
        cross_encoder_scorer=cross_encoder_scorer,
        rank_by=cfg.get("rank_by"),
        dedup_identical_text=bool(cfg.get("dedup_identical_text", True)),
        weights=cfg.get("weights"),
        source_priority=cfg.get("source_priority"),
        k0=int(cfg.get("k0", _DEFAULT_K0)),
        resort_by_date=bool(cfg.get("resort_by_date", False)),
        chronological_order=cfg.get("chronological_order"),
    )


def _build_cross_encoder(cfg: dict) -> CandidateScorer:
    """Resolve the cross-encoder model from the registry and construct its scorer.

    Raises a clear error (never a silent fall back to the model-free score) when
    the recipe asks for the cross-encoder but names no model, or the registry
    entry has no path on disk.
    """
    from jr_pipeline.pipeline_steps.step_5_rerank_chunks.cross_encoder import (
        CrossEncoderReranker,
    )
    from jr_pipeline.runtime_infrastructure.models_registry import resolve_model

    model_id = cfg.get("model_id")
    if not model_id:
        raise ValueError(
            "cross_encoder reranking requires reranking.model_id (a models_registry key)"
        )
    entry = resolve_model(model_id)
    model_path = entry.get("resolved_path")
    if not model_path:
        raise ValueError(
            f"models_registry entry {model_id!r} declares no 'path' for the cross-encoder"
        )
    return CrossEncoderReranker(
        model_path=model_path,
        max_length=cfg.get("max_length"),
        batch_size=int(cfg.get("batch_size", 16)),
        device_preference=cfg.get("device", "auto"),
    )
