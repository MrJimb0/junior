"""retry/recovery layer wrapped around a single LLM chat call (ADR 0030).

A network call to an LLM can fail for transient reasons (server busy, brief
outage) or come back cut off mid-answer. Providers call chat_with_retries
instead of sending the request directly, and this wrapper does four things:

- on retryable HTTP statuses (408 request-timeout, 429 rate-limited, 5xx
  server errors) it waits and tries again, with the wait roughly doubling
  each attempt ("exponential backoff") plus a little randomness ("jitter")
  so many workers don't all retry in lockstep; it honors the server's
  Retry-After hint up to a safety cap.
- on final failure it appends a record of the failed call — endpoint, task,
  error kind/status, never the prompt text itself — to a JSONL "quarantine"
  file, so a batch run doesn't lose track of which calls blew up. (The full
  prompt already sits in the step's receipt on the PHI side.)
- content-retry loop: if the answer looks cut off (the model reports it
  stopped because it hit the length limit, with suspiciously short content,
  or the JSON brackets don't balance), it raises max_tokens and asks again —
  but only when the recipe opted in via ``llm.retry_cut_off_answers`` (the
  retry re-spends tokens on the same question, so the recipe decides; the
  pipeline holds no list of qualifying variables).
- a non-retryable client error (4xx other than the ones above) means trying
  again is pointless: write one quarantine record, then raise immediately.

tests that mock the underlying chat bypass this wrapper.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
    LLMResponse,
)

try:
    import requests
except ImportError:  # pragma: no cover — requests is a dep of the openai/gemini providers
    requests = None  # type: ignore[assignment]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: float = 0.2
    sleep_cap_s: float = 300.0
    retry_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    max_content_retries: int = 2
    content_retry_delay_s: float = 1.0
    token_boost_multiplier: float = 1.5
    hard_max_tokens: int = 32000


@dataclass
class ResilienceContext:
    endpoint_name: str
    task_name: str | None = None
    quarantine_path: Path | None = None


def _sleep_jittered(delay: float, jitter: float) -> None:
    scale = 1.0 + random.uniform(-jitter, jitter)
    time.sleep(max(0.0, delay * scale))


def _parse_retry_after(header: str | None) -> float | None:
    if not header:
        return None
    h = header.strip()
    try:
        return float(h)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(h)
        from datetime import datetime
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return max(0.0, (dt - datetime.now(UTC)).total_seconds())
    except Exception:
        return None


def _quarantine(path: Path | None, entry: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


_TRUNCATION_CONTENT_LEN = 32


def _finish_reason(response: LLMResponse) -> str | None:
    raw = response.response_raw or {}
    try:
        return raw.get("choices", [{}])[0].get("finish_reason")
    except Exception:
        return None


def _detect_truncation(response: LLMResponse) -> bool:
    """best-effort guess: does this response look cut off mid-output (truncated)?"""
    fr = _finish_reason(response)
    content = response.content or ""
    if fr == "length" and len(content) < _TRUNCATION_CONTENT_LEN:
        return True
    if response.usage and response.usage.get("completion_tokens") == 0 and fr == "length":
        return True
    # unclosed JSON: more opening { or [ than closing ones means the answer was cut
    # off before it finished. allow a one-bracket slack for a stray top-level brace.
    s = content.strip()
    if s and s[0] in "{[":
        openers = s.count("{") + s.count("[")
        closers = s.count("}") + s.count("]")
        if openers > closers + 1:
            return True
    return False


def chat_with_retries(
    *,
    inner: Callable[[LLMRequest], LLMResponse],
    req: LLMRequest,
    ctx: ResilienceContext,
    policy: RetryPolicy | None = None,
) -> LLMResponse:
    """wrap inner(req) with retry-on-transient-failure plus retry-on-cut-off-answer."""
    policy = policy or RetryPolicy()
    attempt = 0
    content_retry_count = 0
    current_req = req

    while True:
        attempt += 1
        try:
            resp = inner(current_req)
        except Exception as e:  # noqa: BLE001 — we dispatch below on type
            http_error = requests is not None and isinstance(e, requests.HTTPError)
            transport_error = requests is not None and isinstance(
                e, (requests.ConnectionError, requests.Timeout)
            )
            if not http_error:
                # only retry errors that are inherently transient (connection dropped,
                # timed out). A repeatable local error (JSON parse, validation, a code
                # bug) would just re-send the PHI prompt to no effect — so refuse rather
                # than re-send when unsure (fail closed), leaving a durable quarantine record.
                if not transport_error:
                    _quarantine(ctx.quarantine_path, {
                        "ts": time.time(),
                        "endpoint": ctx.endpoint_name,
                        "task": ctx.task_name,
                        "error_kind": type(e).__name__,
                        "kind": "deterministic_failure_no_retry",
                        "attempts": attempt,
                    })
                    raise
                if attempt >= policy.max_retries:
                    _quarantine(ctx.quarantine_path, {
                        "ts": time.time(),
                        "endpoint": ctx.endpoint_name,
                        "task": ctx.task_name,
                        "error_kind": type(e).__name__,
                        "kind": "transport_retries_exhausted",
                        "attempts": attempt,
                    })
                    raise
                _sleep_jittered(
                    min(policy.max_delay_s, policy.base_delay_s * (2 ** max(0, attempt - 1))),
                    policy.jitter,
                )
                continue
            status = _status_of(e)
            if status is not None and 400 <= status < 500 and status not in policy.retry_statuses:
                _quarantine(ctx.quarantine_path, {
                    "ts": time.time(),
                    "endpoint": ctx.endpoint_name,
                    "task": ctx.task_name,
                    "status": status,
                    "error": repr(e),
                    "kind": "non_retryable",
                })
                raise
            if attempt >= policy.max_retries:
                _quarantine(ctx.quarantine_path, {
                    "ts": time.time(),
                    "endpoint": ctx.endpoint_name,
                    "task": ctx.task_name,
                    "status": status,
                    "error": repr(e),
                    "kind": "retries_exhausted",
                    "attempts": attempt,
                })
                raise
            ra = _parse_retry_after(_header_of(e, "Retry-After"))
            base = min(policy.max_delay_s, policy.base_delay_s * (2 ** max(0, attempt - 1)))
            delay = min(policy.sleep_cap_s, max(base, ra or 0.0))
            _sleep_jittered(delay, policy.jitter)
            continue

        if _detect_truncation(resp):
            # the bigger-budget retry runs only when the recipe opted in
            # (llm.retry_cut_off_answers, carried on the request).
            if (
                content_retry_count < policy.max_content_retries
                and current_req.retry_cut_off_answers
            ):
                boosted = min(
                    policy.hard_max_tokens,
                    int(current_req.max_tokens * policy.token_boost_multiplier),
                )
                if boosted > current_req.max_tokens:
                    current_req = _with_max_tokens(current_req, boosted)
                    content_retry_count += 1
                    time.sleep(policy.content_retry_delay_s)
                    # reset the transient-error attempt counter so the bigger-budget call
                    # gets a fresh, full set of network retries.
                    attempt = 0
                    continue
            _quarantine(ctx.quarantine_path, {
                "ts": time.time(),
                "endpoint": ctx.endpoint_name,
                "task": ctx.task_name,
                "kind": "truncation",
                "finish_reason": _finish_reason(resp),
                "content_len": len(resp.content or ""),
            })
        meta = dict(resp.usage or {})
        if attempt > 1:
            meta["retry_attempts"] = attempt
        if content_retry_count:
            meta["content_retries"] = content_retry_count
            meta["adjusted_max_tokens"] = current_req.max_tokens
        return LLMResponse(
            content=resp.content,
            response_raw=resp.response_raw,
            resolved_model=resp.resolved_model,
            usage=meta,
            latency_s=resp.latency_s,
        )


def _with_max_tokens(req: LLMRequest, new_max: int) -> LLMRequest:
    # copy the request but with a new max_tokens; keep task_name and quarantine_path
    # so later code reading them off the current request still sees the right values.
    return LLMRequest(
        endpoint_name=req.endpoint_name,
        messages=req.messages,
        temperature=req.temperature,
        max_tokens=new_max,
        response_format=req.response_format,
        seed=req.seed,
        timeout_s=req.timeout_s,
        task_name=req.task_name,
        quarantine_path=req.quarantine_path,
        retry_cut_off_answers=req.retry_cut_off_answers,
    )


def _status_of(e: Exception) -> int | None:
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    try:
        return int(resp.status_code)
    except Exception:
        return None


def _header_of(e: Exception, name: str) -> str | None:
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    try:
        return resp.headers.get(name)
    except Exception:
        return None
