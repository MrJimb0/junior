"""Freed GPU memory has to be handed back, and the asking has a sharp edge.

Torch's caching allocator keeps freed blocks in a pool rather than returning them to the
OS. Measured on this pipeline, one extraction left 11.8 GiB of live tensors against 47
GiB held by the allocator — transient attention and generation buffers, freed but never
returned. A few patients of that reaches the MPS high-watermark, and then EVERY
allocation in the process fails, including the small one the query encoder needs. That is
what "hybrid member failed / RuntimeError" was: retrieval blamed for a chart it had
nothing to do with, 48 times over, while the stage still reported success.

These tests never invoke a real backend. `torch.mps.empty_cache()` in a process that
never initialized MPS segfaults — which is not catchable, and is exactly what happened
when this reclaim was first called speculatively between patients. The dispatch is
checked against a stub instead, and the contract that the caller must have just used the
device is asserted at its one call site.
"""
from __future__ import annotations

import gc
import sys
import types
import weakref

import pytest

from jr_pipeline.runtime_infrastructure.torch_device_selection import (
    release_cached_device_memory,
)


class _Backend:
    def __init__(self):
        self.emptied = 0
        self.synchronized = 0
        self.events = []

    def synchronize(self):
        self.synchronized += 1
        self.events.append("synchronize")

    def empty_cache(self):
        self.emptied += 1
        self.events.append("empty_cache")


@pytest.fixture
def fake_torch(monkeypatch):
    """A torch whose backends record rather than allocate."""
    torch = types.ModuleType("torch")
    torch.mps, torch.cuda = _Backend(), _Backend()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_it_reclaims_the_named_device_and_only_that_one(fake_torch):
    release_cached_device_memory("mps")

    assert fake_torch.mps.emptied == 1
    assert fake_torch.mps.synchronized == 1
    assert fake_torch.mps.events == ["synchronize", "empty_cache"]
    assert fake_torch.cuda.emptied == 0, "freed a device the caller was not using"


@pytest.mark.parametrize("device", ["cpu", "", "meta", "MPS", None])
def test_anything_that_is_not_a_gpu_is_left_alone(fake_torch, device):
    """The guard is the whole safety story, so it refuses everything it does not know —
    including a capitalised spelling, rather than normalising and hoping."""
    release_cached_device_memory(device)  # type: ignore[arg-type]

    assert fake_torch.mps.emptied == 0
    assert fake_torch.cuda.emptied == 0


def test_a_backend_without_empty_cache_is_not_an_error(monkeypatch):
    torch = types.ModuleType("torch")
    torch.cuda = object()  # an older torch, or a stripped build
    monkeypatch.setitem(sys.modules, "torch", torch)

    release_cached_device_memory("cuda")


class _Encoded(dict):
    def to(self, device):
        return self


class _TrackingTokenizer:
    pad_token_id = 0

    def __init__(self, torch):
        self._torch = torch
        self.prompt_ref = None

    def apply_chat_template(self, messages, **kwargs):
        prompt = self._torch.zeros((1, 5), dtype=self._torch.long)
        self.prompt_ref = weakref.ref(prompt)
        return _Encoded({"input_ids": prompt})

    def decode(self, tokens, skip_special_tokens=True):
        return "output text"


class _TrackingModel:
    def __init__(self, torch, *, fail=False):
        self._torch = torch
        self._fail = fail
        self.output_ref = None

    def generate(self, prompt_ids, attention_mask=None, **kwargs):
        if self._fail:
            raise RuntimeError("MPS backend out of memory")
        output = self._torch.zeros((1, 7), dtype=self._torch.long)
        self.output_ref = weakref.ref(output)
        return output


def _request():
    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
        LLMRequest,
    )

    return LLMRequest(
        endpoint_name="dev",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )


def _tracking_provider(torch, *, fail=False):
    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (
        LocalHFProvider,
    )

    provider = LocalHFProvider(endpoint_name="dev", model_id="fake")
    provider._tokenizer = _TrackingTokenizer(torch)
    provider._model = _TrackingModel(torch, fail=fail)
    provider._device = "mps"
    return provider


def test_generation_drops_tensor_references_before_reclaim(monkeypatch):
    """empty_cache only releases unoccupied blocks, so it must run after all local
    prompt/output aliases are gone rather than merely somewhere near the return."""
    torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers import (
        local_huggingface_provider,
    )

    provider = _tracking_provider(torch)
    reclaimed = []

    def record_reclaim(device):
        gc.collect()
        reclaimed.append(device)
        assert provider._tokenizer.prompt_ref() is None
        assert provider._model.output_ref() is None

    monkeypatch.setattr(
        local_huggingface_provider, "release_cached_device_memory", record_reclaim
    )

    provider._raw_chat(_request())

    assert reclaimed == ["mps"]


def test_generation_reclaims_after_oom(monkeypatch):
    """An OOM is precisely when cleanup matters most; the next retrieval/generation
    must not inherit all of the failed call's now-unreferenced allocator blocks."""
    torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers import (
        local_huggingface_provider,
    )

    provider = _tracking_provider(torch, fail=True)
    reclaimed = []

    def record_reclaim(device):
        reclaimed.append(device)

    monkeypatch.setattr(
        local_huggingface_provider, "release_cached_device_memory", record_reclaim
    )

    with pytest.raises(RuntimeError, match="out of memory"):
        provider._raw_chat(_request())

    assert reclaimed == ["mps"]
