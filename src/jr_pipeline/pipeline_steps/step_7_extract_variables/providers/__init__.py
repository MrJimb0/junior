"""LLM provider subsystem.

A "provider" is the adapter that actually sends a prompt to a large
language model and returns its answer. Core ships three:
``openai_compatible_provider`` (any endpoint that speaks the OpenAI
chat-completions request shape), ``gemini_compatible_provider``
(Gemini-shaped), and ``local_huggingface_provider`` (a model run
in-process on this machine for development — HF means the HuggingFace
model hub/library; nothing leaves the machine, i.e. no network egress).

The pipeline refuses public consumer APIs by hardcoded denylist; only
endpoints present in an institutional allowlist — loaded from a path
*outside* this repo — are usable. This keeps patient data from ever
being sent to an unapproved host.
"""
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (  # noqa: F401
    LLMProvider,
    LLMRequest,
    LLMResponse,
    ResolvedModel,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_lookup import (  # noqa: F401
    build_provider,
    register_provider,
)
