"""[PHI]: deterministic LLM failures are not blindly retried.

Retrying a deterministic local error (parse/validation) just re-sends the PHI prompt to
no effect. Only transport-transient (connection/timeout) errors are retried; deterministic
exceptions fail closed with a sanitized quarantine receipt.
"""
from __future__ import annotations

import pytest
import requests

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_call_retry_logic import (
    ResilienceContext,
    RetryPolicy,
    chat_with_retries,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
)


def _req():
    return LLMRequest(endpoint_name="e", messages=[{"role": "user", "content": "hi"}])


def test_deterministic_exception_is_not_retried(tmp_path):
    calls = {"n": 0}

    def inner(req):
        calls["n"] += 1
        raise ValueError("parse failed for patient MRN 9999999")  # deterministic + PHI-ish

    ctx = ResilienceContext(endpoint_name="e", task_name="t", quarantine_path=tmp_path / "q.jsonl")
    policy = RetryPolicy(max_retries=4, base_delay_s=0.0, jitter=0.0)
    with pytest.raises(ValueError):
        chat_with_retries(inner=inner, req=_req(), ctx=ctx, policy=policy)
    assert calls["n"] == 1, "a deterministic exception must not be retried"
    quarantine = (tmp_path / "q.jsonl").read_text()
    assert "deterministic_failure_no_retry" in quarantine
    assert "9999999" not in quarantine  # sanitized receipt (error_kind only, no raw message)


def test_transport_exception_is_retried(tmp_path):
    calls = {"n": 0}

    def inner(req):
        calls["n"] += 1
        raise requests.ConnectionError("connection refused")

    ctx = ResilienceContext(endpoint_name="e", task_name="t", quarantine_path=tmp_path / "q.jsonl")
    policy = RetryPolicy(max_retries=3, base_delay_s=0.0, jitter=0.0)
    with pytest.raises(requests.ConnectionError):
        chat_with_retries(inner=inner, req=_req(), ctx=ctx, policy=policy)
    assert calls["n"] == 3, "a transport error must be retried up to max_retries"
