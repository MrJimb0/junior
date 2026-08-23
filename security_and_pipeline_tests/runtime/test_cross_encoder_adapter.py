"""Cross-encoder adapter behavior with FAKE tokenizer/model fixtures.

No local weights: the adapter's loaded state is injected, so _ensure_loaded early-returns
and rerank() exercises the real batching / device-transfer / logit-normalization / sort.
torch is used only to build fake logit tensors. It is the `torch` extra rather than a
core dependency, so this module skips when it is absent instead of failing collection.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.cross_encoder import (  # noqa: E402
    CrossEncoderReranker,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (  # noqa: E402
    RerankInput,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate  # noqa: E402

_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


class _FakeCorpus:
    def text_for(self, chunk_id: str) -> str:
        return f"text::{chunk_id}"


class _FakeEnc(dict):
    def __init__(self, n, recorder):
        super().__init__(input_ids=torch.zeros((n, 4), dtype=torch.long))
        self._recorder = recorder

    def to(self, device):
        self._recorder["to_device"] = device
        return self


class _FakeTokenizer:
    model_max_length = 512

    def __init__(self, recorder):
        self._recorder = recorder

    def __call__(self, queries, chunks, **kw):
        self._recorder.setdefault("calls", []).append(
            {"n": len(chunks), "kw": kw, "queries": list(queries), "chunks": list(chunks)}
        )
        return _FakeEnc(len(chunks), self._recorder)


class _FakeModel:
    def __init__(self, logits_fn, recorder):
        self._logits_fn = logits_fn
        self._recorder = recorder

    def __call__(self, **enc):
        self._recorder["grad_enabled"] = torch.is_grad_enabled()
        return SimpleNamespace(logits=self._logits_fn(enc["input_ids"].shape[0]))


def _loaded(logits_fn, recorder, *, batch_size=16, device="cpu", max_length=512):
    r = CrossEncoderReranker(model_path="/does/not/exist", batch_size=batch_size)
    r._tokenizer = _FakeTokenizer(recorder)
    r._model = _FakeModel(logits_fn, recorder)
    r._device = device
    r._max_length = max_length
    return r


def _cands(*specs):
    return [Candidate(chunk_id=cid, rank=rank, score=0.0, retriever="bm25") for cid, rank in specs]


# ── logit normalization (the static helper) ────────────────────────────────────
def test_flatten_logits_accepts_n_n1_and_singleton():
    assert CrossEncoderReranker._flatten_logits(torch.tensor([1.0, 2.0, 3.0]), 3) == [1.0, 2.0, 3.0]
    assert CrossEncoderReranker._flatten_logits(torch.tensor([[1.0], [2.0]]), 2) == [1.0, 2.0]
    assert CrossEncoderReranker._flatten_logits(torch.tensor(5.0), 1) == [5.0]


def test_flatten_logits_rejects_bad_shape():
    with pytest.raises(ValueError, match="one score per"):
        CrossEncoderReranker._flatten_logits(torch.zeros((2, 2)), 2)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_flatten_logits_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="non-finite"):
        CrossEncoderReranker._flatten_logits(torch.tensor([1.0, bad]), 2)


# ── rerank() with fakes ─────────────────────────────────────────────────────────
def test_empty_candidates_returns_without_loading():
    # _model is NOT injected; if rerank tried to load /does/not/exist it would raise.
    r = CrossEncoderReranker(model_path="/does/not/exist")
    assert r.rerank(_FakeCorpus(), RerankInput(candidates=[], query_text="q"), top_n=5) == []
    assert r._model is None  # never loaded


def test_tokenizer_receives_pairs_truncation_and_max_length():
    rec: dict = {}
    r = _loaded(lambda n: torch.arange(n, dtype=torch.float32), rec, max_length=128)
    r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 1), ("b", 2)), query_text="Q"), top_n=10)
    call = rec["calls"][0]
    assert call["kw"]["truncation"] is True and call["kw"]["padding"] is True
    assert call["kw"]["max_length"] == 128
    assert call["queries"] == ["Q", "Q"]          # query paired with each chunk
    assert call["chunks"] == ["text::a", "text::b"]
    assert rec["to_device"] == "cpu"               # batch moved to the model device
    assert rec["grad_enabled"] is False            # forward ran under inference_mode


def test_batches_respect_batch_size():
    rec: dict = {}
    r = _loaded(lambda n: torch.zeros(n, dtype=torch.float32), rec, batch_size=1)
    r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 1), ("b", 2), ("c", 3)), query_text="q"), top_n=10)
    assert len(rec["calls"]) == 3 and all(c["n"] == 1 for c in rec["calls"])


def test_sorts_by_logit_desc_and_writes_rounded_feature():
    rec: dict = {}
    # logits by input order a,b,c -> 0.1, 0.9, 0.5  => order b, c, a
    r = _loaded(lambda n: torch.tensor([0.1, 0.9, 0.5][:n], dtype=torch.float32), rec)
    out = r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 1), ("b", 2), ("c", 3)), query_text="q"), top_n=10)
    assert [c.chunk_id for c in out] == ["b", "c", "a"]
    assert [c.rank for c in out] == [1, 2, 3]
    assert out[0].features["cross_encoder_logit"] == pytest.approx(0.9)


def test_tie_break_prior_rank_then_chunk_id():
    rec: dict = {}
    # equal logits -> tie broken by prior rank (1 < 9), then chunk_id
    r = _loaded(lambda n: torch.zeros(n, dtype=torch.float32), rec)
    out = r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 9), ("z", 1)), query_text="q"), top_n=10)
    assert [c.chunk_id for c in out] == ["z", "a"]  # z has the better prior rank


@pytest.mark.parametrize("device", _DEVICES)
def test_deterministic_order_per_device(device):
    # device list is built dynamically (cpu always; mps only when available) — no skip.
    runs = []
    for _ in range(2):
        rec: dict = {}
        r = _loaded(lambda n: torch.tensor([0.2, 0.8, 0.5][:n], dtype=torch.float32), rec, device=device)
        out = r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 1), ("b", 2), ("c", 3)), query_text="q"), top_n=10)
        runs.append([c.chunk_id for c in out])
    assert runs[0] == runs[1] == ["b", "c", "a"]


def test_no_torch_raises_clearly_and_never_falls_back(monkeypatch):
    # simulate torch absent: importing it inside _ensure_loaded must raise a clear error,
    # not silently degrade to a model-free reranker.
    monkeypatch.setitem(sys.modules, "torch", None)
    r = CrossEncoderReranker(model_path="/does/not/exist")
    with pytest.raises(RuntimeError, match="requires torch"):
        r.rerank(_FakeCorpus(), RerankInput(candidates=_cands(("a", 1)), query_text="q"), top_n=5)


def test_deterministic_cublas_is_requested_before_anything_touches_a_device(monkeypatch):
    """CUDA reads CUBLAS_WORKSPACE_CONFIG only when its runtime initializes, so asking for
    deterministic cuBLAS after a model is already on the GPU asks for nothing. select_device
    is the choke point every torch model passes through on its way to a device, which makes
    it the last point still early enough — the cross-encoder itself runs far too late."""
    import os

    from jr_pipeline.runtime_infrastructure.torch_device_selection import select_device

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    select_device("cpu")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_an_operators_own_cublas_setting_is_not_overwritten(monkeypatch):
    import os

    from jr_pipeline.runtime_infrastructure.torch_device_selection import select_device

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    select_device("cpu")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
