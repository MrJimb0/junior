"""local_hf truncated generations flow through the shared retry/quarantine layer.

chat() delegates to chat_with_retries, so a cut-off local generation gets the same
bigger-budget content retry and truncation quarantine as a remote call, instead of
being returned unguarded. The bigger-budget retry is gated by the recipe's
llm.retry_cut_off_answers flag carried on the request: an opted-in recipe retries,
one that did not opt in quarantines the truncation without retry.
"""
from __future__ import annotations

import json

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
        return "x"  # short content -> looks truncated when finish_reason is 'length'


class _RecordingModel:
    """Always emits n_generated new tokens and records the max_new_tokens of each call."""

    def __init__(self, n_generated):
        self._n = n_generated
        self.max_new_tokens_seen = []

    def generate(self, prompt_ids, attention_mask=None, **kwargs):
        self.max_new_tokens_seen.append(kwargs.get("max_new_tokens"))
        prompt_len = prompt_ids.shape[-1]
        return torch.zeros((1, prompt_len + self._n), dtype=torch.long)


def _provider(model):
    provider = LocalHFProvider(endpoint_name="dev", model_id="fake", max_new_tokens_cap=16000)
    provider._tokenizer = _FakeTokenizer()
    provider._model = model
    provider._device = "cpu"
    return provider


def _req(quarantine_path, retry_cut_off_answers=False, max_tokens=10):
    return LLMRequest(
        endpoint_name="dev",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
        task_name="any_variable",
        quarantine_path=quarantine_path,
        retry_cut_off_answers=retry_cut_off_answers,
    )


def test_opted_in_recipe_retries_truncated_generation_with_a_bigger_budget(tmp_path):
    # First call: max_new = min(10, cap) = 10; the model emits 10 -> truncated. The
    # recipe opted in (llm.retry_cut_off_answers), so a second call runs with 15 and,
    # now under the cap, finishes normally.
    model = _RecordingModel(n_generated=10)
    _provider(model).chat(
        _req(quarantine_path=tmp_path / "q.jsonl", retry_cut_off_answers=True)
    )
    assert len(model.max_new_tokens_seen) == 2
    assert model.max_new_tokens_seen[1] > model.max_new_tokens_seen[0]


def test_recipe_that_did_not_opt_in_quarantines_truncation_without_retry(tmp_path):
    model = _RecordingModel(n_generated=10)
    quarantine = tmp_path / "q.jsonl"
    _provider(model).chat(_req(quarantine_path=quarantine))
    assert len(model.max_new_tokens_seen) == 1  # no bigger-budget retry
    records = [json.loads(line) for line in quarantine.read_text().splitlines()]
    assert any(r["kind"] == "truncation" for r in records)
