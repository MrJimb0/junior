"""A local-model weight swap busts the LLM response cache and changes the fingerprint.

The response cache is keyed by provider_config (make_key deliberately excludes the model
fingerprint), so the on-disk weight hash must live in provider_config for a same-path
weight swap to invalidate stale answers for every run, pinned or not. The weight hash is
also part of the model fingerprint that a pinned run verifies.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_7_extract_variables.llm_response_cache import (  # noqa: E402
    LLMCache,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (  # noqa: E402
    LLMRequest,
    LLMResponse,
    ResolvedModel,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (  # noqa: E402
    LocalHFProvider,
)


def _model_dir(path, weight_bytes):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(weight_bytes)
    (path / "config.json").write_text("{}")
    return path


def _req():
    return LLMRequest(
        endpoint_name="dev", messages=[{"role": "user", "content": "hi"}], max_tokens=64
    )


def test_weight_swap_changes_provider_config_and_cache_key(tmp_path):
    model_dir = _model_dir(tmp_path / "m", b"OLD-WEIGHTS")
    before_cfg = LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config()
    before_key = LLMCache.make_key(req=_req(), provider_config=before_cfg)

    # different length so the content-hash's (path, size, mtime) cache can't serve stale
    (model_dir / "model.safetensors").write_bytes(b"NEW-WEIGHTS!!")
    after_cfg = LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config()
    after_key = LLMCache.make_key(req=_req(), provider_config=after_cfg)

    assert before_cfg["model_weight_sha256"] != after_cfg["model_weight_sha256"]
    assert before_key != after_key


def test_weight_sha_is_cached_across_provider_instances(tmp_path, monkeypatch):
    # provider_config() runs once per (patient, recipe) to form the LLM cache key; the
    # multi-GB weight file must be hashed once per run, not once per provider instance.
    import jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider as prov

    model_dir = _model_dir(tmp_path / "m", b"WEIGHTS-A")
    prov._WEIGHT_SHA_CACHE.clear()  # isolate the count from other tests sharing the cache
    calls = {"n": 0}
    real_hash_file = prov.hash_file

    def counting_hash_file(path):
        calls["n"] += 1
        return real_hash_file(path)

    monkeypatch.setattr(prov, "hash_file", counting_hash_file)

    prov.LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config()
    after_first = calls["n"]
    assert after_first >= 1  # the weight file was hashed on the first instance

    prov.LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config()
    assert calls["n"] == after_first  # a second instance on the same weights re-uses the cache


def test_cache_entry_written_under_one_weight_misses_after_swap(tmp_path):
    model_dir = _model_dir(tmp_path / "m", b"WEIGHTS-A")
    key_a = LLMCache.make_key(
        req=_req(),
        provider_config=LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config(),
    )
    cache = LLMCache(path=tmp_path / "cache.sqlite")
    cache.put(
        key_a,
        LLMResponse(
            content="{}",
            response_raw={"ok": True},
            resolved_model=ResolvedModel(
                endpoint_name="dev", api_model_id=str(model_dir), api_version=None,
                deployment_name="local", fingerprint="fp",
            ),
            usage={},
            latency_s=0.0,
        ),
    )
    assert cache.get(key_a) is not None  # same weight -> same key -> hit

    (model_dir / "model.safetensors").write_bytes(b"WEIGHTS-BBBB")
    key_b = LLMCache.make_key(
        req=_req(),
        provider_config=LocalHFProvider(endpoint_name="dev", model_id=str(model_dir)).provider_config(),
    )
    assert cache.get(key_b) is None  # swapped weight -> different key -> miss, never a stale hit
    cache.close()


class _Encoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return _Encoded({"input_ids": torch.zeros((1, 5), dtype=torch.long)})

    def decode(self, tokens, skip_special_tokens=True):
        return "{}"


class _FakeModel:
    def generate(self, prompt_ids, attention_mask=None, **kwargs):
        prompt_len = prompt_ids.shape[-1]
        return torch.zeros((1, prompt_len + 2), dtype=torch.long)  # 2 new tokens, under cap


def _fake_provider(model_dir):
    provider = LocalHFProvider(endpoint_name="dev", model_id=str(model_dir))
    provider._tokenizer = _FakeTokenizer()
    provider._model = _FakeModel()
    provider._device = "cpu"
    return provider


def test_weight_swap_changes_model_fingerprint(tmp_path):
    model_dir = _model_dir(tmp_path / "m", b"WEIGHTS-A")
    fp_a = _fake_provider(model_dir).chat(_req()).resolved_model.fingerprint
    (model_dir / "model.safetensors").write_bytes(b"WEIGHTS-BBBB")
    fp_b = _fake_provider(model_dir).chat(_req()).resolved_model.fingerprint
    assert fp_a.startswith("local_hf|")
    assert fp_a != fp_b
