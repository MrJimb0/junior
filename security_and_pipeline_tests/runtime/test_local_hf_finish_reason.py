"""local_hf reports finish_reason='length' on truncation.

When a generation hits the token cap the provider reports finish_reason='length'
(not 'stop'), so the resilience layer's truncation-boost retry can fire instead of
treating the truncated output as complete.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (  # noqa: E402
    LLMRequest,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (  # noqa: E402
    LocalHFProvider,
)


class _Encoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return _Encoded({"input_ids": torch.zeros((1, 5), dtype=torch.long)})

    def decode(self, tokens, skip_special_tokens=True):
        return "output text"


class _FakeModel:
    def __init__(self, n_generated):
        self._n = n_generated

    def generate(self, prompt_ids, attention_mask=None, **kwargs):
        prompt_len = prompt_ids.shape[-1]
        return torch.zeros((1, prompt_len + self._n), dtype=torch.long)


def _provider(n_generated):
    provider = LocalHFProvider(endpoint_name="dev", model_id="fake", max_new_tokens_cap=16000)
    provider._tokenizer = _FakeTokenizer()
    provider._model = _FakeModel(n_generated)
    provider._device = "cpu"
    return provider


def _req(max_tokens):
    return LLMRequest(endpoint_name="dev", messages=[{"role": "user", "content": "hi"}],
                      max_tokens=max_tokens)


def test_finish_reason_length_when_generation_hits_cap():
    # max_new = min(max_tokens=10, cap=16000) = 10; model emits exactly 10 -> truncated
    resp = _provider(n_generated=10).chat(_req(max_tokens=10))
    assert resp.response_raw["choices"][0]["finish_reason"] == "length"
    assert resp.response_raw["usage"]["completion_tokens"] == 10


def test_finish_reason_stop_when_generation_under_cap():
    resp = _provider(n_generated=4).chat(_req(max_tokens=10))
    assert resp.response_raw["choices"][0]["finish_reason"] == "stop"
