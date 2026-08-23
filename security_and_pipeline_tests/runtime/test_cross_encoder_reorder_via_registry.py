"""Controlled-reorder fixture: the registry dispatch path reorders.

build_reranker resolves the gte registry entry and constructs the cross-encoder
adapter; with fake tokenizer/model fixtures (no weights) whose logits intentionally
reverse the retrieval order, the registry-built reranker must flip the pair. This is
the deterministic proof of reordering that the real slow spine deliberately does NOT
have to demonstrate (a real model may or may not reorder on toy data).
"""
from __future__ import annotations

import textwrap
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import build_reranker  # noqa: E402
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (  # noqa: E402
    RerankInput,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate  # noqa: E402


class _FakeCorpus:
    def text_for(self, chunk_id: str) -> str:
        return f"text::{chunk_id}"


class _FakeEnc(dict):
    def __init__(self, n):
        super().__init__(input_ids=torch.zeros((n, 4), dtype=torch.long))

    def to(self, device):
        return self


class _FakeTokenizer:
    model_max_length = 512

    def __call__(self, queries, chunks, **kw):
        return _FakeEnc(len(chunks))


class _FakeModel:
    def __init__(self, logits_fn):
        self._logits_fn = logits_fn

    def __call__(self, **enc):
        return SimpleNamespace(logits=self._logits_fn(enc["input_ids"].shape[0]))


def _registry(tmp_path):
    (tmp_path / "models" / "reranking" / "gte_x").mkdir(parents=True)
    reg = tmp_path / "models" / "models_registry.yaml"
    reg.write_text(textwrap.dedent("""
        models:
          gte_reranker_modernbert:
            type: reranker
            provider: cross_encoder
            kind: cross_encoder
            path: models/reranking/gte_x
    """))
    return reg


def _cands(*specs):
    return [Candidate(chunk_id=cid, rank=rank, score=0.0, retriever="bm25") for cid, rank in specs]


def test_registry_path_reorders_with_fake_logits(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_MODELS_REGISTRY", str(_registry(tmp_path)))
    # Dispatch through the registry exactly like the recipe path does.
    reranker = build_reranker({"cross_encoder": True, "model_id": "gte_reranker_modernbert"})
    assert reranker.info.kind == "composed"

    # Inject fakes onto the inner cross-encoder scorer so no weights load; logits
    # REVERSE the retrieval order: retrieval gives a(rank 1) before b(rank 2), but
    # the cross-encoder scores b higher (0.1 vs 0.9) -> reranked order must be [b, a].
    scorer = reranker._cross_encoder_scorer
    scorer._tokenizer = _FakeTokenizer()
    scorer._model = _FakeModel(lambda n: torch.tensor([0.1, 0.9][:n], dtype=torch.float32))
    scorer._device = "cpu"
    scorer._max_length = 128

    out = reranker.rerank(
        _FakeCorpus(),
        RerankInput(candidates=_cands(("a", 1), ("b", 2)), query_text="q"),
        top_n=10,
    ).candidates
    assert [c.chunk_id for c in out] == ["b", "a"], "registry cross-encoder path did not reorder"
    assert out[0].features["cross_encoder_logit"] == 0.9
    # the reranker never silently became a model-free fallback
    assert reranker._cross_encoder_scorer is not None
