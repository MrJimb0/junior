"""shared types for the llm (large language model) provider subsystem.

every provider turns an ``LLMRequest`` (endpoint name + chat messages +
generation settings) into an ``LLMResponse`` (the parsed answer + the raw
payload + a ``ResolvedModel`` fingerprint captured from the api response).
A fingerprint here is a short identifying value computed from the response
that tells us exactly which model/backend answered. it is the load-bearing
bit: if the institution silently swaps which model sits behind an endpoint
name, the fingerprint changes across runs even though the allowlist name
stayed the same — so we can catch the swap instead of trusting it blindly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


def build_auth_header(auth: str | None, endpoint_name: str | None = None) -> dict[str, str]:
    """Build the HTTP authentication header for an allowlist ``auth`` spec
    ("scheme:value" or "scheme:$ENV_NAME", where $ENV_NAME reads the secret
    from an environment variable instead of hardcoding it; a bare value
    defaults to the "bearer" token scheme). "apim" is the institution's API
    gateway, which authenticates with a subscription-key header. Shared by the
    openai- and gemini-shaped HTTP providers. Empty/None auth yields no header.

    A referenced env var that is unset or empty is a loud error: sending a request
    with an empty key otherwise 401s at call time with no hint about the cause."""
    if not auth:
        return {}
    if ":" in auth:
        scheme, rhs = auth.split(":", 1)
    else:
        scheme, rhs = "bearer", auth
    if rhs.startswith("$"):
        env_name = rhs.lstrip("$")
        rhs = os.environ.get(env_name, "")
        if not rhs:
            where = f" for endpoint {endpoint_name!r}" if endpoint_name else ""
            raise ValueError(
                f"auth references environment variable ${env_name}{where}, but it is unset "
                "or empty; export it before running so requests carry a real key rather "
                "than an empty one that 401s at call time."
            )
    scheme = scheme.lower().strip()
    if scheme == "bearer":
        return {"Authorization": f"Bearer {rhs}"}
    if scheme in {"apim", "ocp-apim-subscription-key"}:
        return {"Ocp-Apim-Subscription-Key": rhs}
    return {"Authorization": f"{scheme} {rhs}"}


@dataclass(frozen=True)
class ResolvedModel:
    """which model/backend actually answered, captured from the api response.

    ``fingerprint`` is a short identifying value computed from the response.
    It is required on every llm-call receipt (ADR 0017). If the institution
    silently re-points an endpoint at a different model, the ``fingerprint``
    differs across runs even when the allowlist name didn't change — letting
    us detect the swap.
    """

    endpoint_name: str
    api_model_id: str
    api_version: str | None
    deployment_name: str | None
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_name": self.endpoint_name,
            "api_model_id": self.api_model_id,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class LLMRequest:
    endpoint_name: str
    messages: list[dict[str, str]]   # the chat turns: [{"role": "system", "content": "..."}, ...]
    temperature: float = 0.0          # 0.0 = deterministic / least random output
    max_tokens: int = 1024            # ceiling on how many tokens the model may generate
    response_format: str | None = None   # "json_object" forces strict JSON, for providers that support it
    seed: int | None = None
    timeout_s: float | None = None
    task_name: str | None = None       # which extraction task; labels quarantine records and receipts
    quarantine_path: Path | None = None  # if the call ultimately fails, a record of the failure (not the prompt) is appended here for review
    # The recipe's llm.retry_cut_off_answers flag, carried on the request so the
    # retry layer can see it: when true, an answer that comes back cut off is
    # retried with a bigger max_tokens budget instead of shipped truncated.
    retry_cut_off_answers: bool = False


@dataclass(frozen=True)
class LLMResponse:
    content: str
    response_raw: dict[str, Any]
    resolved_model: ResolvedModel
    usage: dict[str, Any] = field(default_factory=dict)   # token counts: prompt_tokens, completion_tokens, total
    latency_s: float = 0.0


class LLMProvider(Protocol):
    """minimal contract for llm providers used by the extract runner."""

    def chat(self, req: LLMRequest) -> LLMResponse: ...

    def provider_config(self) -> dict[str, Any]:
        """The provider identity used as part of the result-cache key, so a config
        change re-runs extraction. MUST NOT include urls, keys, or deployment
        secrets (those must never land in cached metadata)."""
