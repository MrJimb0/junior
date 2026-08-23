"""local_hf raises loudly when the prompt would overflow the model's context window.

A remote API 400s on overflow; a local model would instead silently generate degraded
output that still parses as JSON. The provider guards on the model's usable context and
raises a descriptive error naming the recipe and the sizes, rather than truncating the
evidence.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (  # noqa: E402
    LLMRequest,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (  # noqa: E402
    LocalHFProvider,
)


class _Encoded(dict):
    moved = False

    def to(self, device):
        self.moved = True
        return self


class _FakeTokenizer:
    pad_token_id = 0
    model_max_length = 1_000_000  # sentinel -> ignored, so the config window is authoritative
    last_encoded = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_encoded = _Encoded(
            {"input_ids": torch.zeros((1, 6), dtype=torch.long)}
        )
        return self.last_encoded

    def decode(self, tokens, skip_special_tokens=True):
        return "unused"


class _SmallWindowModel:
    config = SimpleNamespace(max_position_embeddings=8)

    def generate(self, *args, **kwargs):
        raise AssertionError("generate must not run when the context guard trips")


def test_oversized_prompt_is_rejected_with_a_descriptive_error():
    provider = LocalHFProvider(endpoint_name="dev", model_id="fake", max_new_tokens_cap=16000)
    provider._tokenizer = _FakeTokenizer()
    provider._model = _SmallWindowModel()
    provider._device = "cpu"
    # prompt is 6 tokens + max_new 10 = 16, over the model's 8-token window
    req = LLMRequest(
        endpoint_name="dev",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=10,
        task_name="stage",
    )
    with pytest.raises(ValueError, match="exceeds the model's usable context"):
        provider.chat(req)


def test_hardware_prompt_ceiling_is_checked_before_copying_to_device():
    provider = LocalHFProvider(
        endpoint_name="dev",
        model_id="fake",
        max_prompt_tokens_cap=5,
    )
    provider._tokenizer = _FakeTokenizer()
    provider._model = _SmallWindowModel()
    provider._device = "cpu"
    req = LLMRequest(
        endpoint_name="dev",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        task_name="stage",
    )

    with pytest.raises(ValueError, match="safe prompt ceiling of 5"):
        provider.chat(req)
    assert provider._tokenizer.last_encoded.moved is False
