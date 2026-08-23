"""provider for any endpoint that speaks the OpenAI /chat/completions request shape.

Many model servers copy OpenAI's request/response format, so this one adapter
covers a lot: an institutional API gateway (APIM-style — the gateway
handles authentication), Azure OpenAI, and self-hosted servers
(vLLM / TGI / LM Studio / Ollama). The denylist (forbidden hosts) is checked
both when the allowlist loads and again in __post_init__, so building this
object directly can't bypass it.
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
class OpenAICompatProvider:
    """HTTPS provider that speaks the OpenAI chat-completions JSON format."""

    endpoint_name: str
    url: str
    # how to authenticate: "scheme:value" or "scheme:$ENV_NAME" (the $ form reads the
    # secret from an environment variable). e.g. "bearer:$HF_TOKEN", "apim:$APIM_KEY".
    auth: str | None = None
    default_model: str | None = None
    timeout_s: float = 60.0
    # Endpoint-specific quirks carried over from the allowlist `extras` (e.g. Azure / GPT-5):
    #   api_version             -> appended as ?api-version=... (Azure deployments require it)
    #   use_max_completion_tokens -> send the parameter named max_completion_tokens instead of max_tokens (GPT-5)
    #   skip_temperature        -> omit the temperature parameter entirely (GPT-5 rejects it)
    api_version: str | None = None
    use_max_completion_tokens: bool = False
    skip_temperature: bool = False

    def __post_init__(self) -> None:
        check_url(self.url)  # direct construction must not bypass the denylist

    def _request_url(self) -> str:
        """append the Azure-style api-version query param when the endpoint declares one."""
        if not self.api_version:
            return self.url
        sep = "&" if "?" in self.url else "?"
        return f"{self.url}{sep}api-version={self.api_version}"

    def provider_config(self) -> dict[str, Any]:
        """The provider identity used in the result-cache key. url/auth are excluded
        on purpose, so an endpoint is identified by its name (and so secrets never
        end up in cached metadata)."""
        return {
            "kind": "openai_compat",
            "endpoint_name": self.endpoint_name,
            "default_model": self.default_model,
        }

    def chat(self, req: LLMRequest) -> LLMResponse:
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
        model = self.default_model or "unknown"

        payload: dict[str, Any] = {
            "model": model,
            "messages": req.messages,
        }
        # GPT-5 rejects `temperature`; skip it when the endpoint says so.
        if not self.skip_temperature:
            payload["temperature"] = req.temperature
        # GPT-5 / newer Azure deployments use `max_completion_tokens`, not `max_tokens`.
        payload["max_completion_tokens" if self.use_max_completion_tokens else "max_tokens"] = req.max_tokens
        if req.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        if req.seed is not None:
            payload["seed"] = req.seed

        headers = {"Content-Type": "application/json", **build_auth_header(self.auth, self.endpoint_name)}
        resp = requests.post(
            self._request_url(),
            json=payload,
            headers=headers,
            timeout=req.timeout_s or self.timeout_s,
            # Redirects are not followed, because the allowlist and the denylist are
            # checked against the CONFIGURED url and nothing re-checks a hop. requests
            # follows a 307 by resending this body — the chart passages — to whatever
            # host the response names. It drops the Authorization header across hosts,
            # so the credential survives that and the patient data does not. A gateway
            # that answers with a redirect should be corrected in the allowlist rather
            # than followed here.
            allow_redirects=False,
        )
        resp.raise_for_status()
        data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            # the server returned JSON that doesn't match the expected OpenAI shape.
            raise RuntimeError(f"Unexpected openai-compat response shape: keys={list(data.keys())}") from e

        resolved = ResolvedModel(
            endpoint_name=self.endpoint_name,
            api_model_id=str(data.get("model", model)),
            api_version=data.get("system_fingerprint") or None,
            deployment_name=None,
            fingerprint=f"{data.get('model', model)}|{data.get('system_fingerprint', '')}",
        )
        return LLMResponse(
            content=content,
            response_raw=data,
            resolved_model=resolved,
            usage=data.get("usage") or {},
            latency_s=round(time.perf_counter() - t0, 6),
        )
