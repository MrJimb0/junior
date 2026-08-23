"""Talk to a Gemini-style language-model endpoint hosted by the institution.

This is the connector that lets an extraction step send a prompt to a Google
Gemini model and read its answer back, when that model is served from an
approved institutional endpoint (for example a Vertex AI deployment) rather
than the public internet.

It is a thin wrapper over the model's `:generateContent` HTTP endpoint. The
endpoint URL and the authentication token come from the institutional
allowlist (the vetted list of endpoints we are permitted to call). The public
consumer Google AI Studio host (`generativelanguage.googleapis.com`) is on the
denylist — the hardcoded list of forbidden public hosts — and can never be
added to the allowlist. This keeps PHI (protected health information) off
shared public infrastructure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
    check_url,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
    LLMResponse,
    ResolvedModel,
    build_auth_header,
)


@dataclass
class GeminiCompatProvider:
    """An HTTPS connector that speaks the Gemini generateContent request/response shape."""

    endpoint_name: str
    url: str
    auth: str | None = None
    default_model: str | None = None
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        check_url(self.url)

    def provider_config(self) -> dict[str, Any]:
        """The provider settings that go into the response cache key.

        The URL and auth token are deliberately left out: they are secrets/
        infrastructure details, and including them would needlessly change the
        cache key (so an identical question would miss the cache after a URL
        change).
        """
        return {
            "kind": "gemini_compat",
            "endpoint_name": self.endpoint_name,
            "default_model": self.default_model,
        }

    def chat(self, req: LLMRequest) -> LLMResponse:
        """Send the prompt to the endpoint, retrying transient failures.

        Wraps the actual HTTP call so that temporary network/server errors are
        retried with a growing wait between attempts (backoff), handled by the
        shared resilience layer.
        """
        from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_call_retry_logic import (
            ResilienceContext,
            chat_with_retries,
        )

        ctx = ResilienceContext(
            endpoint_name=self.endpoint_name,
            task_name=req.task_name,
            quarantine_path=req.quarantine_path,
        )
        return chat_with_retries(inner=self._raw_chat, req=req, ctx=ctx)

    def _raw_chat(self, req: LLMRequest) -> LLMResponse:
        import requests

        t0 = time.perf_counter()

        # Gemini wants the prompt in its own format: a separate "system
        # instruction" plus one "user" turn. Most of the codebase carries
        # prompts in the OpenAI chat format (a list of role/content messages),
        # so here we split that list into the system text and the user text.
        system = ""
        user_parts: list[dict[str, str]] = []
        for m in req.messages:
            role = m.get("role") or ""
            content = m.get("content") or ""
            if role == "system":
                system = content
            elif role == "user":
                user_parts.append({"text": content})

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]} if system else None,
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": {
                "temperature": req.temperature,
                "maxOutputTokens": req.max_tokens,
            },
        }
        if payload["systemInstruction"] is None:
            payload.pop("systemInstruction")
        if req.response_format == "json_object":
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {"Content-Type": "application/json", **build_auth_header(self.auth, self.endpoint_name)}
        # allow_redirects=False for the reason spelled out in openai_compatible_provider:
        # nothing re-checks a redirect hop against the allowlist, so following one would
        # resend the chart passages to a host no one approved.
        resp = requests.post(
            self.url,
            json=payload,
            headers=headers,
            timeout=req.timeout_s or self.timeout_s,
            allow_redirects=False,
        )
        resp.raise_for_status()
        data = resp.json()

        # Gemini may split a long answer into several pieces ("parts"); stitch
        # them back together into one string. Only the first candidate is read:
        # "candidates" are alternative answers to the same question, and
        # concatenating them would interleave two different answers into one
        # corrupted string.
        content = ""
        try:
            candidates = data.get("candidates") or []
            parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
            for p in parts:
                if "text" in p:
                    content += p["text"]
        except Exception as e:
            raise RuntimeError(f"Unexpected gemini response shape: keys={list(data.keys())}") from e

        model_id = data.get("modelVersion") or self.default_model or "unknown"
        resolved = ResolvedModel(
            endpoint_name=self.endpoint_name,
            api_model_id=str(model_id),
            api_version=str(data.get("modelVersion") or ""),
            deployment_name=None,
            fingerprint=f"{model_id}|{data.get('modelVersion', '')}",
        )
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            content=content,
            response_raw=data,
            resolved_model=resolved,
            usage={
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
            latency_s=round(time.perf_counter() - t0, 6),
        )
