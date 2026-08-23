"""A recipe step that just asks the model a question — no chart search.

Most steps first pull relevant chart text (retrieval) and feed it to the model.
This one skips that: it fills the prompt template using only the outputs of
earlier steps in the same recipe, sends it to the model, and then tries to read
the model's reply as JSON. "Best-effort JSON" means: if the reply isn't clean
JSON we record the parse error and return no data, rather than crashing.

Use it for steps that reason over already-extracted values (e.g. derive a stage
group from TNM fields another step produced) instead of over raw chart text.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.evidence_grounding import (
    ask_until_grounded,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.llm_response_cache import LLMCache
from jr_pipeline.pipeline_steps.step_7_extract_variables.prompt_template_rendering import (
    load_prompt,
    render,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_type_lookup import (
    register_step,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.retrieve_and_prompt_step import (
    _best_effort_json,
    _read_text_or_empty,
)


@register_step("llm_only")
class LLMOnlyHandler:
    """Runs one model call with no chart search, and writes a receipt of what
    prompt was sent and what came back (for auditability)."""

    kind = "llm_only"

    def execute(self, ctx: StepContext) -> StepResult:
        step = ctx.step
        if not step.prompt:
            raise ValueError(f"step {step.id}: 'prompt' path required")

        template = load_prompt(step.prompt)
        tpl_context = {
            "patient_id": ctx.patient_id,
            "evidence_json": "",
            "evidence_text": "",
            "OUTPUT_SCHEMA": _read_text_or_empty(ctx.recipe.output_schema_path),
            "vars": ctx.upstream_vars,
            "steps": ctx.step_outputs,
        }
        sys_rendered, usr_rendered = render(template, tpl_context)
        messages = [
            {"role": "system", "content": sys_rendered},
            {"role": "user", "content": usr_rendered},
        ]

        provider = ctx.provider
        if provider is None:
            raise RuntimeError(f"step {step.id}: no provider available")

        req = LLMRequest(
            endpoint_name=provider.provider_config().get("endpoint_name", "unknown"),
            messages=messages,
            temperature=ctx.recipe.llm.temperature,
            max_tokens=ctx.recipe.llm.max_tokens,
            response_format=ctx.recipe.llm.response_format,
            seed=ctx.recipe.llm.seed,
            task_name=ctx.recipe.name,
            quarantine_path=(ctx.quarantine_dir / "quarantine.jsonl") if ctx.quarantine_dir else None,
            retry_cut_off_answers=ctx.recipe.llm.retry_cut_off_answers,
        )

        def _call_once(conversation):
            call_req = replace(req, messages=conversation)
            if ctx.llm_cache is not None:
                key = LLMCache.make_key(
                    req=call_req, provider_config=provider.provider_config()
                )
                expected_fp = getattr(ctx.recipe.llm, "expected_fingerprint", None)
                hit = ctx.llm_cache.get(key, expected_fingerprint=expected_fp)
                if hit is not None:
                    return hit, True
                fresh = provider.chat(call_req)
                ctx.llm_cache.put(key, fresh)
                return fresh, False
            return provider.chat(call_req), False

        # This step retrieves nothing of its own -- it reasons over what earlier steps
        # established -- so the passages it may cite are the ones those steps showed.
        # Same set the recipe-level guard uses, which is the point: a step must not be
        # refused here for a citation that would be accepted there.
        shown_from_earlier_steps: set[str] = set()
        for earlier in (ctx.step_outputs or {}).values():
            if isinstance(earlier, dict):
                shown_from_earlier_steps.update(earlier.get("evidence_chunk_ids") or [])

        outcome = ask_until_grounded(
            call_once=_call_once,
            messages=messages,
            shown_chunk_ids=shown_from_earlier_steps,
            # With nothing established upstream there is no list to correct toward, so
            # the correction would name no valid ids and the retry would be theatre.
            max_retries=int(ctx.max_provenance_retries or 0) if shown_from_earlier_steps else 0,
            parse=_best_effort_json,
        )
        response = outcome["response"]
        parsed, parse_error = outcome["parsed"], outcome["parse_error"]
        provenance_attempts = outcome["attempts"]

        receipt_payload: dict[str, Any] = {
            "prompt": {
                "template_name": template.name,
                "template_hash": template.template_hash,
                "context_vars": {"patient_id": ctx.patient_id},
            },
            "messages_sent": messages,
            "resolved_model": response.resolved_model.to_dict(),
            "response_raw": response.response_raw,
            "response_parsed": parsed if parsed is not None else None,
            "validation": {"parse_ok": parse_error is None, "parse_error": parse_error},
            "timings": {"llm_s": response.latency_s},
            # Attempts whose citations named passages no earlier step showed. Empty is
            # the ordinary case, and always empty when nothing was established upstream.
            "provenance_retries": provenance_attempts,
        }
        data = parsed if isinstance(parsed, dict) else None
        if data is None and parse_error is None:
            parse_error = "parsed response was not a JSON object"
        return StepResult(
            data=data,
            receipt_payload=receipt_payload,
            error=parse_error if data is None else None,
        )
