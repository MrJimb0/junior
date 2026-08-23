"""Models-registry loader + cross_encoder build dispatch + loud errors.

The cross-encoder's logit-normalization / determinism (with fake tokenizer/model
fixtures, no weights) is covered separately. These tests need no torch/weights:
build_reranker only resolves the registry and constructs the adapter (the model loads
lazily in rerank()).
"""
from __future__ import annotations

import textwrap

import pytest

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import build_reranker
from jr_pipeline.runtime_infrastructure.models_registry import resolve_model


def _registry(tmp_path):
    (tmp_path / "models" / "reranking" / "gte_x").mkdir(parents=True)
    reg = tmp_path / "models" / "models_registry.yaml"
    reg.write_text(textwrap.dedent("""
        models:
          gte_x:
            type: reranker
            provider: cross_encoder
            kind: cross_encoder
            path: models/reranking/gte_x
    """))
    return reg


def test_resolve_model_returns_absolute_path(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_MODELS_REGISTRY", str(_registry(tmp_path)))
    entry = resolve_model("gte_x")
    assert entry["kind"] == "cross_encoder"
    assert entry["resolved_path"] == tmp_path / "models" / "reranking" / "gte_x"


def test_resolve_model_unknown_id_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_MODELS_REGISTRY", str(_registry(tmp_path)))
    with pytest.raises(KeyError):
        resolve_model("not_registered")


def test_build_cross_encoder_via_registry_without_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_MODELS_REGISTRY", str(_registry(tmp_path)))
    r = build_reranker({"cross_encoder": True, "model_id": "gte_x"})
    # the candidate ranker wraps a cross-encoder scorer when the toggle is on
    assert r.info.kind == "composed"
    # the ranker's config folds in the wrapped scorer's identity (its model_path,
    # max_length, batch_size) so the reranker fingerprint tracks the cross-encoder used
    assert r.info.config["cross_encoder"]["model_path"] == (
        r._cross_encoder_scorer.info.config["model_path"]
    )
    # __init__ records the resolved path on the inner scorer but must not load the
    # model (no weights here)
    assert str(tmp_path) in r._cross_encoder_scorer.info.config["model_path"]


def test_cross_encoder_requires_a_model_id(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_MODELS_REGISTRY", str(_registry(tmp_path)))
    with pytest.raises(ValueError, match="model_id"):
        build_reranker({"cross_encoder": True})
