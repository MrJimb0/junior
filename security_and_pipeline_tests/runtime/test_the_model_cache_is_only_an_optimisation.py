"""The model-response cache speeds things up. It is not what makes resume correct.

Those are easy to confuse, and confusing them is expensive. One real run
re-extracted a finished patient in twenty minutes with 29 rows in its cache and zero
hits — because the prompts genuinely differed (a config bug meant one pass capped
evidence at 7 chunks and the other did not). The cache behaved correctly throughout.
What was missing was any record-based reason to skip work that was already done.

So resume is decided by recorded extraction state, and holds with the cache empty; the
cache only ever saves a call within work that was going to run anyway.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_7_extract_variables.llm_response_cache import LLMCache
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
)


def _a_request(**overrides) -> LLMRequest:
    base = dict(
        endpoint_name="local_qwen",
        messages=[{"role": "user", "content": "what stage is this"}],
        temperature=0.0,
        max_tokens=512,
        response_format="json_object",
        seed=None,
        task_name="stage",
    )
    return LLMRequest(**{**base, **overrides})


_A_PROVIDER = {"kind": "local_hf", "model_id": "/models/qwen", "model_weight_sha256": "sha256:w"}


def test_the_same_request_always_gets_the_same_key():
    """Cheap to break and hard to notice: anything time-varying or object-identity based
    folded into the key turns every run into a cold cache."""
    first = LLMCache.make_key(req=_a_request(), provider_config=dict(_A_PROVIDER))
    second = LLMCache.make_key(req=_a_request(), provider_config=dict(_A_PROVIDER))

    assert first == second


def test_different_evidence_is_a_different_key():
    """This is the behaviour that produced 0 hits on that run, and it was right to. The
    prompts differed because one pass saw 7 chunks and the other 16; a cache that
    matched them would have served an answer to a different question."""
    seven = _a_request(messages=[{"role": "user", "content": "evidence: c1 c2 c3 c4 c5 c6 c7"}])
    sixteen = _a_request(messages=[{"role": "user", "content": "evidence: c1 .. c16"}])

    assert LLMCache.make_key(req=seven, provider_config=dict(_A_PROVIDER)) != LLMCache.make_key(
        req=sixteen, provider_config=dict(_A_PROVIDER)
    )


def test_swapping_the_weights_at_the_same_path_is_a_different_key():
    """model_id is a path and says nothing about what is at it. The weight hash is what
    stops a rebuilt model at the same location serving the previous model's answers."""
    original = LLMCache.make_key(req=_a_request(), provider_config=dict(_A_PROVIDER))
    rebuilt = LLMCache.make_key(
        req=_a_request(), provider_config={**_A_PROVIDER, "model_weight_sha256": "sha256:other"}
    )

    assert original != rebuilt


def test_a_stored_answer_from_a_different_model_is_a_miss(tmp_path):
    """The fingerprint is checked at read time rather than folded into the key, so a
    gateway quietly repointing an endpoint reads as a miss instead of as a hit (ADR
    0028)."""
    cache = LLMCache(path=tmp_path / "llm_cache.db")
    key = LLMCache.make_key(req=_a_request(), provider_config=dict(_A_PROVIDER))

    assert cache.get(key, expected_fingerprint="model-a") is None
