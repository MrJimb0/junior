"""provider registry and factory.

A recipe (the extraction config) names an LLM endpoint by NAME only. The
institutional allowlist resolves that name into a provider kind, URL, and
authentication, and the "factory" functions here turn that resolved entry
into a live, ready-to-call provider object.

raw URLs from recipe config are never accepted — the only legitimate
URL source is the institutional allowlist file, which is already
checked against the denylist (the list of forbidden hosts) at load time.
Forcing every provider to be built through this single path means there is
no way to slip an unvetted endpoint into the pipeline.
"""
from __future__ import annotations

from collections.abc import Callable

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.gemini_compatible_provider import (
    GeminiCompatProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_allowlist import (
    AllowedEndpoint,
    AllowlistError,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (
    LocalHFProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.openai_compatible_provider import (
    OpenAICompatProvider,
)

_FACTORIES: dict[str, Callable[[AllowedEndpoint], LLMProvider]] = {}


def register_provider(kind: str, factory: Callable[[AllowedEndpoint], LLMProvider]) -> None:
    """register a factory that turns an allowed endpoint into a provider."""
    _FACTORIES[kind] = factory


_DEFAULT_LOCAL_HF_MAX_NEW_TOKENS = 16000


def _openai_factory(e: AllowedEndpoint) -> LLMProvider:
    extras = e.extras or {}
    provider = OpenAICompatProvider(
        endpoint_name=e.name,
        url=e.url,
        auth=e.auth,
        default_model=e.default_model,
        # endpoint-specific quirks (e.g. Azure / GPT-5) from the allowlist's `extras`.
        api_version=extras.get("api_version"),
        use_max_completion_tokens=bool(extras.get("use_max_completion_tokens", False)),
        skip_temperature=bool(extras.get("skip_temperature", False)),
    )
    # slow endpoints (e.g. a reasoning model) may raise the HTTP timeout per entry.
    if extras.get("default_timeout_seconds"):
        provider.timeout_s = float(extras["default_timeout_seconds"])
    return provider


def _gemini_factory(e: AllowedEndpoint) -> LLMProvider:
    extras = e.extras or {}
    provider = GeminiCompatProvider(
        endpoint_name=e.name,
        url=e.url,
        auth=e.auth,
        default_model=e.default_model,
    )
    # slow endpoints (e.g. a reasoning model) may raise the HTTP timeout per entry,
    # the same as the openai factory — otherwise the gemini provider stays on 60s.
    if extras.get("default_timeout_seconds"):
        provider.timeout_s = float(extras["default_timeout_seconds"])
    return provider


# Local providers already built in this process, keyed by everything that decides
# which weights get loaded and how. A LocalHFProvider loads its model lazily on first
# use and holds it on the instance, so a fresh instance per call means reloading a
# multi-gigabyte model for every recipe of every patient. Extraction over a cohort
# runs in one process, which makes that the difference between loading the model once
# and loading it hundreds of times. Mirrors the embed encoder's _ENCODER_CACHE.
_LOCAL_PROVIDER_CACHE: dict[tuple, LLMProvider] = {}


def _local_hf_factory(e: AllowedEndpoint) -> LLMProvider:
    # for the local in-process model, the allowlist's `url` field instead carries the
    # HuggingFace model id (the on-disk path or hub name of the model); default_model is unused.
    extras = e.extras or {}
    settings = {
        "endpoint_name": e.name,
        "model_id": e.url,
        "device_preference": extras.get("device", "auto"),
        "dtype": extras.get("dtype"),
        "max_new_tokens_cap": int(
            extras.get("max_new_tokens_cap", _DEFAULT_LOCAL_HF_MAX_NEW_TOKENS)
        ),
        "max_prompt_tokens_cap": (
            int(extras["max_prompt_tokens_cap"])
            if extras.get("max_prompt_tokens_cap") is not None
            else None
        ),
        # Free text describing the hardware the ceiling was measured on. Travels with
        # the cap so the screen that shows the number can also say whose number it is.
        "measured_on": extras.get("measured_on"),
        # offline by default; an endpoint may opt into auto-downloading the model from the
        # HuggingFace Hub (dev/demo only — a patient-data run must never reach the network).
        "allow_hub_download": bool(extras.get("allow_download", False)),
    }
    # Every setting is in the key, so two endpoints that differ in any of them get
    # their own provider rather than silently sharing one model.
    key = tuple(sorted(settings.items()))
    if key not in _LOCAL_PROVIDER_CACHE:
        _LOCAL_PROVIDER_CACHE[key] = LocalHFProvider(**settings)
    return _LOCAL_PROVIDER_CACHE[key]


register_provider("openai_compat", _openai_factory)
register_provider("gemini_compat", _gemini_factory)
register_provider("local_hf", _local_hf_factory)


def build_provider(endpoint: AllowedEndpoint) -> LLMProvider:
    """construct the provider instance for an endpoint via its registered factory."""
    if endpoint.provider not in _FACTORIES:
        raise AllowlistError(
            f"No provider factory registered for kind {endpoint.provider!r}. "
            f"Known: {sorted(_FACTORIES)}"
        )
    return _FACTORIES[endpoint.provider](endpoint)
