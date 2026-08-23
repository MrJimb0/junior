"""Find the relevant chart text, then ask a language model to read it and answer.

This is the main extraction step — the one most chart variables use. For a
question like "what stage was this cancer?", it gathers the passages of the
patient's record most likely to contain the answer, hands that evidence to a
language model along with the question, and reads back the model's answer as
structured data (JSON).

The work happens in four stages, which map onto pipeline steps 4 through 7:

  1. Retrieve candidate passages (step 4): pull the most relevant short text
     spans ("chunks") for this question out of the patient's record.
  2. Rerank them (step 5): filter by the recipe's metadata conditions, then
     re-sort the survivors so the best ones come first. With the cross-encoder off
     (the default) this is the deterministic rule-based combined score — it blends
     simple signals (no model, so the order is fully reproducible) — and keeps the
     top few.
  3. Prepare the evidence packet (step 6): consolidate the selected passages into
     one labeled package, count its token size, and format it for the prompt.
  4. Prompt and parse (step 7): fill the prompt template with that evidence,
     call the model (reusing a cached answer if the identical call was made
     before), and do a best-effort parse of the JSON the model returns.

Everything is recorded in a "receipt": the candidate passages, the rerank with
its per-signal scores and the variable's declared archetype (its extraction
pattern, e.g. a single value vs. a list), the prepared packet, the exact
messages sent, the raw model response, and the parsed result — so an auditor can
replay the whole call later.
"""
from __future__ import annotations

import json
import time
from typing import Any

from jinja2 import UndefinedError

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.builder_table_normalization import (
    normalize_builder_source,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import build_retriever
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.nonembedded_structured_parquet_table_looker_uper.direct_parquet_retriever_v1 import (  # noqa: E501
    _ROW_ID_COLUMN,
    read_filtered_table_rows,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rerank import build_reranker
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    EvidenceFilter,
    RerankInput,
)
from jr_pipeline.pipeline_steps.step_6_prepare_evidence_for_extraction.prepare_evidence import (
    prepare_evidence,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.evidence_grounding import (
    ask_until_grounded,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.llm_response_cache import LLMCache
from jr_pipeline.pipeline_steps.step_7_extract_variables.prompt_template_rendering import (
    load_prompt,
    render,
    render_inline,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recent_retrieval_coverage import (
    append_recent_candidates,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_type_lookup import (
    register_step,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    encoder_fingerprint,
    fingerprint,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    builder_intermediates_dir,
    derived_tables_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import render_table_row
from jr_pipeline.runtime_infrastructure.source_file_discovery import discover_source_files

_log = get_logger("rerank")


def chunks_that_may_reach_one_prompt(recipe_top_n: int, machine_ceiling) -> int:
    """The lower of what the recipe asks for and what this machine can execute.

    A recipe's top_n is a research judgement about how much evidence the question
    needs; the ceiling is a fact about the hardware in front of it, set once per
    machine (max_chunks_per_prompt) rather than edited into shared recipes. No
    ceiling means the recipe decides, which is what a cluster wants. Always the
    minimum, never a raise: hardware may cut evidence short, it may not add any.

    Absent and 0 both mean "no ceiling", which is how an unset key arrives. A NEGATIVE
    ceiling is refused instead: it used to reach the reranker, which returns no
    candidates at all for a non-positive top_n, so every prompt in the cohort ran on
    empty evidence and answered null with nothing raised anywhere along the way."""
    if not machine_ceiling:
        return int(recipe_top_n)
    ceiling = int(machine_ceiling)
    if ceiling < 0:
        raise ValueError(
            f"max_chunks_per_prompt is {ceiling}; it is how many chunks of chart text "
            "may reach one prompt, so it cannot be negative. Remove the key, or set it "
            "to 0, to let each recipe decide for itself."
        )
    return min(int(recipe_top_n), ceiling)


@register_step("retrieve_and_prompt")
class RetrieveAndPromptHandler:
    """Retrieve the most relevant passages, rerank them, package the evidence, and ask the model."""

    kind = "retrieve_and_prompt"

    def execute(self, ctx: StepContext) -> StepResult:
        step = ctx.step
        cfg = step.config

        retrieval_cfg = cfg.get("retrieval")
        if not retrieval_cfg:
            raise ValueError(f"step {step.id}: 'retrieval' block required")
        if not step.prompt:
            raise ValueError(f"step {step.id}: 'prompt' path required")

        retriever = build_retriever(retrieval_cfg, encoder_cfg_fallback=ctx.encoder_cfg)
        query_text = retrieval_cfg.get("query", "")
        k = int(retrieval_cfg.get("k", 10))

        # Step 4: pull the k most relevant candidate passages for this question
        # (k = how many to retrieve, from the recipe; default 10).
        t_ret = time.perf_counter()
        candidates = retriever.query(ctx.corpus, text=query_text, k=k)
        candidates, recent_coverage_ids = append_recent_candidates(
            candidates, ctx.corpus, retrieval_cfg
        )
        retrieval_elapsed = time.perf_counter() - t_ret

        # Step 5: filter the candidates by the recipe's metadata conditions, then
        # re-sort what survives so the strongest come first and keep the top_n.
        # The cross_encoder toggle, model_id, filters, resort_by_date and top_n all
        # come from the recipe's reranking spec. With the cross-encoder off (the
        # default) scoring uses no model and is fully reproducible: it blends a few
        # simple signals (the combined score), which a recipe can tune via weights
        # or a source-priority preference. This step 5 is what actually SELECTS the
        # top_n passages to keep; the later step-6 budget trim is only a safety net
        # for the rare case where they still overflow. The variable's declared
        # archetype (its extraction pattern, taken from the recipe's front-matter
        # header) is passed along as extraction_type.
        archetype = ctx.recipe.front_matter.get("archetype")
        rr = ctx.recipe.reranking
        step_rr = cfg.get("reranking") or {}
        chronological_order = step_rr.get(
            "chronological_order", rr.chronological_order
        )
        top_n = int(step_rr.get("top_n", rr.top_n))
        # A machine ceiling on how many chunks may reach one prompt, capping whatever
        # the recipe asked for. A recipe's top_n is a research judgement about how much
        # evidence the question needs; this is a fact about the hardware in front of it,
        # and the two do not belong in the same file. Chunks run to 512 tokens, so 20 of
        # them is a ~10k-token prompt — fine on a hosted endpoint, and on a 48 GiB laptop
        # it peaks at 64 GiB, swaps, and the run is killed partway through a cohort.
        #
        # Set once per machine in the project's settings (max_chunks_per_prompt) rather
        # than edited into every recipe, because the recipes are shared and the machines
        # are not. Unset means the recipe decides, which is what a cluster wants.
        top_n = chunks_that_may_reach_one_prompt(top_n, ctx.max_chunks_per_prompt)
        reranker = build_reranker({
            "cross_encoder": rr.cross_encoder,
            "model_id": rr.model_id,
            "rank_by": rr.rank_by,
            "dedup_identical_text": rr.dedup_identical_text,
            "resort_by_date": rr.resort_by_date,
            "chronological_order": chronological_order,
            **(rr.scorer_config or {}),
        })
        # Resolve any {{ vars.* }} reference in a filter value (e.g. a date compared
        # against the upstream diagnosis date) against this patient's upstream
        # results — here, where that context is in hand, so the reranker itself does
        # no templating. A filter whose value can't be resolved (the upstream
        # variable is absent or null for this patient) is skipped, not fatal, and
        # recorded so the skip stays visible in the receipt.
        filter_context = {
            "vars": ctx.upstream_vars,
            "steps": ctx.step_outputs,
            "patient_id": ctx.patient_id,
        }
        resolved_filters, skipped_filters = _resolve_filters(rr.filters, filter_context)
        t_rr = time.perf_counter()
        rerank_result = reranker.rerank(
            ctx.corpus,
            RerankInput(
                candidates=candidates,
                query_text=query_text,
                filters=resolved_filters,
            ),
            top_n=top_n,
        )
        reranked = rerank_result.candidates
        # The filter stats always come from THIS pass: they are the record of what the
        # recipe's filters did, including emptying the pool and triggering the fallback.
        filter_drop_stats = rerank_result.filter_stats
        dedup_stats = rerank_result.dedup_stats
        filter_fallback_used = False
        if (
            not reranked
            and resolved_filters
            and rr.filters_fallback_to_unfiltered
        ):
            fallback_result = reranker.rerank(
                ctx.corpus,
                RerankInput(candidates=candidates, query_text=query_text, filters=()),
                top_n=top_n,
            )
            reranked = fallback_result.candidates
            filter_fallback_used = bool(reranked)
            # The fallback re-ran on the unfiltered pool, so ITS dedup is the one that
            # shaped what the model actually sees.
            dedup_stats = fallback_result.dedup_stats
        rerank_elapsed = time.perf_counter() - t_rr

        # Build the template context now; it is used both to render the final
        # prompt below and (with empty evidence here) as the base everything shares.
        # Every template variable is filled even when empty so the template engine
        # doesn't error on a missing key (StrictUndefined).
        template = load_prompt(step.prompt)
        schema_text = _read_text_or_empty(ctx.recipe.output_schema_path)
        base_context: dict[str, Any] = {
            "patient_id": ctx.patient_id,
            "evidence_json": "[]",
            "evidence_text": "",
            "OUTPUT_SCHEMA": schema_text,
            "vars": ctx.upstream_vars,
            "steps": ctx.step_outputs,
        }

        # Step 6: consolidate the selected passages into one labeled evidence
        # package. Step 5 already chose which passages and in what order; step 6
        # includes them all (it does NOT trim to a budget — the prior steps keep the
        # set small enough for a frontier model's context window) and measures the
        # result, failing loudly if it somehow exceeds the model's context ceiling.
        # Optionally pull whole rows from an already-processed builder table (e.g. a
        # pathology CAP table the builder produced as a CSV for this patient). These
        # are normalized to the same format as any source and handed to step 6 as
        # pre-resolved evidence, so they sit in the SAME package as the retrieved
        # free text. See _gather_builder_table_evidence.
        builder_items, builder_records = _gather_builder_table_evidence(ctx)

        t_pe = time.perf_counter()
        packet = prepare_evidence(
            corpus=ctx.corpus,
            reranked_candidates=[
                {"chunk_id": r.chunk_id, "rank": r.rank, "score": r.score}
                for r in reranked
            ],
            variable=f"{ctx.recipe.name}/{step.id}",
            patient_id=ctx.patient_id,
            max_context_tokens=ctx.recipe.evidence.max_context_tokens,
            direct_evidence_items=builder_items,
        )
        prepare_elapsed = time.perf_counter() - t_pe
        evidence_blocks = packet.get("blocks") or []

        # If no evidence text survived, there is nothing for the model to base an
        # answer on — calling it anyway would just invite a made-up ("hallucinated")
        # answer. So skip the model call entirely and record why.
        empty_evidence = len(evidence_blocks) == 0

        # Build the rows for the selection trace here, while the passage metadata
        # is still in hand. The selection trace is a PHI-free (NO_PHI) audit
        # record of which passages were considered and how they ranked — it
        # carries no patient text. We assemble the rows now but write the trace
        # only after the model answers, so it can also include which passages the
        # model cited (a rough, "weak" signal of what was actually relevant).
        # Passage ids are read here in the clear and converted to non-identifying
        # surrogate ids at write time.
        included_ids = set(packet.get("included_chunk_ids") or [])
        trace_rows = []
        for r in reranked:
            row = ctx.corpus.metadata_for(r.chunk_id) or {}
            trace_rows.append({
                "chunk_id": r.chunk_id,
                "doc_type": row.get("doc_type"),
                "rank": r.rank,
                "prior_rank": r.prior_rank,
                "score": r.score,
                "token_count": row.get("token_count"),
                "in_bundle": r.chunk_id in included_ids,
                "features": dict(getattr(r, "features", {}) or {}),  # the per-signal scores the reranker used
            })
        # Builder-table rows were never reranked (they aren't retrieved chunks), so
        # they have no prior_rank/features; record them in the trace anyway, with
        # whether each made it into the bundle. doc_type is left None -> canonicalized
        # to "unknown" (a table row is not a clinical document type).
        for item in builder_items:
            trace_rows.append({
                "chunk_id": item["chunk_id"],
                "doc_type": None,
                "rank": item.get("rank"),
                "prior_rank": None,
                "score": item.get("score"),
                "token_count": None,
                "in_bundle": item["chunk_id"] in included_ids,
                "features": {},
            })

        # Now render the prompt with the prepared evidence filled in (base_context
        # above carried empty placeholders only to satisfy StrictUndefined).
        tpl_context = dict(base_context)
        tpl_context["evidence_json"] = json.dumps(
            [{"chunk_id": b["chunk_id"], "text": b["text"]} for b in evidence_blocks],
            ensure_ascii=False,
        )
        tpl_context["evidence_text"] = packet.get("formatted_evidence_text", "")
        sys_rendered, usr_rendered = render(template, tpl_context)
        messages = [
            {"role": "system", "content": sys_rendered},
            {"role": "user", "content": usr_rendered},
        ]

        provider = ctx.provider
        if provider is None:
            raise RuntimeError(
                f"step {step.id}: no provider available; check allowlist + "
                "recipe.llm.model name"
            )

        response = None
        was_cached = False
        provenance_attempts: list[dict[str, Any]] = []
        parsed: Any = None
        parse_error: str | None = None
        if empty_evidence:
            parse_error = "empty_evidence: no chunk text retrieved for this step"
            # This is the commonest way a variable comes back empty, and the receipt is
            # the only place it was recorded -- so the answer read downstream as "the
            # chart does not say" when it means nothing was ever put in front of the
            # model. Say it on screen, with the counts that tell the operator where the
            # passages were lost.
            _log.warning("empty_evidence", extra_={
                "variable": ctx.recipe.name,
                "step_id": step.id,
                "n_candidates_retrieved": len(candidates),
                "n_after_reranking": len(reranked),
                "n_builder_table_rows": len(builder_items),
                "hint": f"{ctx.recipe.name}/{step.id}: no chart text reached the model, "
                        "so an empty answer here means we never looked, not that the "
                        "chart is silent. Check this step's reranking filters, and that "
                        "the source_file its retrieval names exists for this patient.",
            })
        else:
            def _call_once(conversation):
                req = LLMRequest(
                    endpoint_name=provider.provider_config().get("endpoint_name", "unknown"),
                    messages=conversation,
                    temperature=ctx.recipe.llm.temperature,
                    max_tokens=ctx.recipe.llm.max_tokens,
                    response_format=ctx.recipe.llm.response_format,
                    seed=ctx.recipe.llm.seed,
                    task_name=ctx.recipe.name,
                    quarantine_path=(ctx.quarantine_dir / "quarantine.jsonl") if ctx.quarantine_dir else None,
                    retry_cut_off_answers=ctx.recipe.llm.retry_cut_off_answers,
                )
                if ctx.llm_cache is not None:
                    key = LLMCache.make_key(req=req, provider_config=provider.provider_config())
                    hit = ctx.llm_cache.get(
                        key, expected_fingerprint=getattr(ctx.recipe.llm, "expected_fingerprint", None)
                    )
                    if hit is not None:
                        return hit, True
                    fresh = provider.chat(req)
                    ctx.llm_cache.put(key, fresh)
                    return fresh, False
                return provider.chat(req), False

            # A citation to a passage this step never showed gets one correction and
            # another ask. Re-sending the SAME request could not help: decoding is
            # greedy and the cache is keyed on the prompt, so the answer would be
            # identical. The shown set here is this step's own evidence packet.
            outcome = ask_until_grounded(
                call_once=_call_once,
                messages=messages,
                shown_chunk_ids=included_ids,
                max_retries=int(ctx.max_provenance_retries or 0),
                parse=_best_effort_json,
            )
            response = outcome["response"]
            parsed, parse_error = outcome["parsed"], outcome["parse_error"]
            provenance_attempts = outcome["attempts"]
            was_cached = outcome["was_cached"]

        receipt_payload: dict[str, Any] = {
            "retrieval": {
                "kind": retriever.info.kind,
                "version": retriever.info.version,
                "config": retriever.info.config,
                "candidates": [
                    {"chunk_id": c.chunk_id, "rank": c.rank, "score": c.score}
                    for c in candidates
                ],
                "score_normalization": retriever.score_normalization,
                "recent_coverage": {
                    "requested": retrieval_cfg.get("include_recent"),
                    "added_chunk_ids": recent_coverage_ids,
                },
                "elapsed_s": round(retrieval_elapsed, 6),
            },
            "rerank": {
                "kind": reranker.info.kind,
                "version": reranker.info.version,
                "config": reranker.info.config,
                "extraction_type": archetype,
                # The metadata filters this call ran: as the recipe declared them
                # (values may still hold {{ vars.* }}), as actually applied (resolved
                # to this patient's values), which were skipped and why, and how many
                # candidates each dropped.
                "filters": {
                    "declared": [_filter_to_dict(f) for f in rr.filters],
                    "applied": [_filter_to_dict(f) for f in resolved_filters],
                    "skipped": skipped_filters,
                    "drop_stats": filter_drop_stats,
                    "fallback_to_unfiltered": {
                        "enabled": rr.filters_fallback_to_unfiltered,
                        "used": filter_fallback_used,
                    },
                },
                # Candidates dropped for repeating text already in the pool: how many,
                # in how many groups, and which chunk each duplicate deferred to.
                "dedup": dedup_stats,
                "candidates": [
                    {
                        "chunk_id": r.chunk_id,
                        "rank": r.rank,
                        "prior_rank": r.prior_rank,
                        "score": r.score,
                        "features": dict(r.features),
                    }
                    for r in reranked
                ],
                "elapsed_s": round(rerank_elapsed, 6),
            },
            "evidence_packet": {
                "block_count": packet.get("block_count"),
                # Estimated (chars/4) — burn awareness, not billing; the model API's
                # real token usage is in the economics block below.
                "total_evidence_tokens": packet.get("total_evidence_tokens"),
                "evidence_tokens_by_doc_type": packet.get("evidence_tokens_by_doc_type"),
                "max_context_tokens": packet.get("max_context_tokens"),
                "included_chunk_ids": packet.get("included_chunk_ids"),
                "empty_evidence": empty_evidence,
                "formatted_evidence_text": packet.get("formatted_evidence_text", ""),
                # Per builder table drawn into the bundle: name, which run's
                # intermediate it came from, the normalized parquet's content hash,
                # and how many rows were considered vs. emitted (or missing=True).
                "builder_tables": builder_records,
                "elapsed_s": round(prepare_elapsed, 6),
            },
            "prompt": {
                "template_name": template.name,
                "template_hash": template.template_hash,
                "context_vars": {
                    k: v for k, v in tpl_context.items()
                    if k in {"patient_id"}
                },
            },
            "messages_sent": messages,
            "resolved_model": response.resolved_model.to_dict() if response is not None else None,
            "response_raw": response.response_raw if response is not None else None,
            "response_parsed": parsed if parsed is not None else None,
            "validation": {
                "parse_ok": parse_error is None,
                "parse_error": parse_error,
            },
            "timings": {
                "retrieval_s": round(retrieval_elapsed, 6),
                "rerank_s": round(rerank_elapsed, 6),
                "prepare_evidence_s": round(prepare_elapsed, 6),
                "llm_s": response.latency_s if response is not None else 0.0,
                "cache": response.usage.get("cache") if (response is not None and isinstance(response.usage, dict)) else None,
            },
            # Per-call cost accounting: how many tokens went in and came out, how
            # long it took, and whether the answer came from cache. Every model
            # connector reports these the same way, so we can compare the cost and
            # accuracy of cheaper vs. pricier models. These numbers flow into the
            # run's overall economics summary downstream.
            "economics": {
                "prompt_tokens": _usage_get(response, "prompt_tokens"),
                "completion_tokens": _usage_get(response, "completion_tokens"),
                "total_tokens": _usage_get(response, "total_tokens"),
                "latency_s": response.latency_s if response is not None else None,
                "cached": was_cached,
                # The token counts above are the FINAL call's. When a citation had to be
                # corrected there were earlier calls too, so this says how many, rather
                # than letting the totals read as the whole spend.
                "llm_calls": len(provenance_attempts) + 1,
            },
            # Attempts whose citations named passages this step never showed, each with
            # the ids it invented. Empty is the ordinary case. Kept even when the next
            # attempt succeeded: an answer corrected into citing its source is weaker
            # evidence than one that cited it first time, and nothing else records that.
            "provenance_retries": provenance_attempts,
        }

        data = parsed if isinstance(parsed, dict) else None
        # If the model's output couldn't be parsed into a JSON object, that's a
        # genuine failure. Record it as an error on the StepResult so it shows up
        # in result.json's errors rather than silently looking successful.
        if data is None and parse_error is None:
            parse_error = "parsed response was not a JSON object"
        error = parse_error if data is None else None

        # Assemble the inputs for the PHI-free (NO_PHI) selection trace: a fixed
        # header identifying the recipe and the exact methods used (each method's
        # "fingerprint" is a short hash of its config, so the trace records which
        # encoder/reranker/chunker version produced it), plus the passage ids the
        # model itself cited as an independent, rough signal of relevance. We hand
        # these to step 8, which is the one that actually writes the trace. Step 7
        # never writes across the PHI -> NO_PHI boundary in the middle of an
        # extraction — that separation keeps patient text out of the PHI-free
        # artifacts.
        trace = {
            "recipe_id": ctx.recipe.name,
            "step_id": step.id,
            "archetype": archetype,
            "retriever_kind": retriever.info.kind,
            "total_evidence_tokens": packet.get("total_evidence_tokens"),
            "ranked": trace_rows,
            "recipe_version": ctx.recipe.version,
            "encoder_fingerprint": encoder_fingerprint(ctx.encoder_cfg or {}),
            # Fingerprint the reranker over its static config AND the recipe-declared
            # filter specs (the template form, recipe text — never the per-patient
            # resolved values), so this PHI-free record captures the filtering policy
            # without leaking a patient's resolved dates.
            "reranker_fingerprint": fingerprint({
                "kind": reranker.info.kind,
                "version": reranker.info.version,
                "config": reranker.info.config,
                "filters": [_filter_to_dict(f) for f in ctx.recipe.reranking.filters],
                "filters_fallback_to_unfiltered": (
                    ctx.recipe.reranking.filters_fallback_to_unfiltered
                ),
            }),
            "chunking_config_id": fingerprint(ctx.chunker_cfg or {}),
            "cited_chunk_ids": _cited_chunk_ids(data),
        }

        # Pass the evidence packet and the selection-trace inputs to step 8
        # (organize_output), which writes the PHI-bearing evidence files and the
        # PHI-free trace. (prepare_evidence above has no side effects, so handing
        # its output on is safe.)
        return StepResult(
            data=data,
            receipt_payload=receipt_payload,
            error=error,
            step8_payload={
                "variable": f"{ctx.recipe.name}/{step.id}",
                "evidence_packet": packet,
                "trace": trace,
            },
        )


# Filter values that, after resolving {{ vars.* }}, mean "the upstream variable had
# no value for this patient" — so the filter can't be applied and is skipped.
_UNRESOLVED_FILTER_VALUES = {"", "none", "null", "unknown", "nan", "n/a"}


def _filter_to_dict(f) -> dict[str, Any]:
    """A reranking filter (declared FilterSpec or resolved EvidenceFilter) as a plain dict."""
    return {
        "field": f.field,
        "op": f.op,
        "value": f.value,
        "keep_if_missing": f.keep_if_missing,
    }


def _resolve_filter_value(value: Any, context: dict[str, Any]) -> tuple[Any, bool]:
    """Render one filter value against the patient's upstream context.

    Returns ``(resolved, ok)``. ``ok`` is False only when a value that referenced an
    upstream variable (a ``{{ ... }}`` template) rendered to an empty/unknown sentinel —
    the variable it referenced had no value for this patient. A literal value is always
    used verbatim, including the literal ``'unknown'`` an operator may legitimately want
    to filter on (e.g. a None doc_type is canonicalized to 'unknown'); only an
    unresolved template is skippable. Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value, True
    rendered = render_inline(value, context).strip()
    is_template = "{{" in value
    if is_template and rendered.lower() in _UNRESOLVED_FILTER_VALUES:
        return rendered, False
    return rendered, True


def _resolve_filters(
    filter_specs, context: dict[str, Any]
) -> tuple[tuple[EvidenceFilter, ...], list[dict[str, Any]]]:
    """Resolve ``{{ vars.* }}`` references in each declared filter into concrete values.

    Returns the filters to apply plus a note for each filter skipped because its
    value could not be resolved (the upstream variable was absent or null for this
    patient). Skipping — rather than crashing, or dropping every candidate — keeps
    a missing optional upstream value from silently emptying the evidence.
    """
    applied: list[EvidenceFilter] = []
    skipped: list[dict[str, Any]] = []
    for f in filter_specs:
        note: dict[str, Any] = {"field": f.field, "op": f.op, "value": f.value}
        try:
            if isinstance(f.value, (list, tuple)):
                resolved_items = []
                for item in f.value:
                    item_value, ok = _resolve_filter_value(item, context)
                    if ok:
                        resolved_items.append(item_value)
                if not resolved_items:
                    note["reason"] = "no list value resolved"
                    skipped.append(note)
                    continue
                resolved_value: Any = resolved_items
            else:
                resolved_value, ok = _resolve_filter_value(f.value, context)
                if not ok:
                    note["reason"] = "value resolved to an empty/unknown upstream variable"
                    skipped.append(note)
                    continue
        except UndefinedError as e:
            note["reason"] = f"undefined upstream variable ({e})"
            skipped.append(note)
            continue
        applied.append(
            EvidenceFilter(
                field=f.field,
                op=f.op,
                value=resolved_value,
                keep_if_missing=f.keep_if_missing,
            )
        )
    if skipped:
        _log.warning("rerank_filters_skipped", extra_={"skipped": skipped})
    return tuple(applied), skipped


def _gather_builder_table_evidence(
    ctx: StepContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve each ``also_include_tables`` recipe entry into pre-resolved evidence.

    Per entry ``{name, filter?, columns?, max_rows?, source_run_id?}``: discover the
    builder's ``<name>.<ext>`` for this patient under the named run (``source_run_id``
    or this run) the SAME way ingest discovers sources (``discover_source_files`` —
    any supported format, case-insensitive), normalize it to a cached derived parquet
    with the same step-1 pipeline, then filter + project the rows with the SAME
    primitive the direct_parquet retriever uses. Returns ``(direct_evidence_items,
    receipt_records)``. A missing table is skipped and recorded, never raised — not
    every patient has every builder table.
    """
    specs = ctx.step.config.get("also_include_tables") or []
    items: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for spec in specs:
        name = spec.get("name")
        if not name:
            raise ValueError(
                f"step {ctx.step.id}: each also_include_tables entry needs a 'name'"
            )
        source_run_id = spec.get("source_run_id") or ctx.run_id
        builder_dir = builder_intermediates_dir(source_run_id, ctx.patient_id)
        # Same discovery as ingest: any supported extension, matched case-insensitively,
        # returning the real on-disk path. So a builder may drop path_cap.csv/.tsv/.xlsx.
        discovered = discover_source_files(builder_dir) if builder_dir.is_dir() else {}
        src_path = discovered.get(name)
        record: dict[str, Any] = {"name": name, "source_run_id": source_run_id}
        if src_path is None:
            record["missing"] = True
            records.append(record)
            continue

        dest = derived_tables_dir(ctx.corpus.patient_root) / f"{name}.parquet"
        sidecar = normalize_builder_source(
            src_path,
            dest,
            run_id=ctx.run_id,
            patient_id=ctx.patient_id,
            origin_run_id=source_run_id,
            code_lock_hash=ctx.code_lock_hash,
        )
        # Tag original row ids + apply the row filter via the SAME primitive the
        # direct_parquet retriever uses, so a builder row's :TABLE id is built the
        # same way a directly-retrieved table row's is.
        df = read_filtered_table_rows(dest, spec.get("filter"))
        columns = spec.get("columns")
        max_rows = spec.get("max_rows")

        rows_included = 0
        for position, raw_row in enumerate(df.iter_rows(named=True), start=1):
            if max_rows is not None and rows_included >= int(max_rows):
                break
            row_id = raw_row.pop(_ROW_ID_COLUMN)
            row = {c: raw_row[c] for c in columns if c in raw_row} if columns else raw_row
            text = render_table_row(row)
            if not text.strip():
                continue
            items.append({
                "chunk_id": f"{ctx.patient_id}:{name}:{row_id}:TABLE",
                "text": text,
                "source_file": name,
                "doc_type": "builder_table",
                "record_index": int(row_id),
                "rank": position,
                "score": 1.0,
            })
            rows_included += 1

        record.update({
            "parquet_content_hash": sidecar.get("parquet_content_hash"),
            "rows_considered": df.height,
            "rows_included": rows_included,
        })
        records.append(record)
    return items, records


def _usage_get(response, key: str):
    """One token-usage field from an LLMResponse, or None if absent."""
    if response is None or not isinstance(getattr(response, "usage", None), dict):
        return None
    return response.usage.get(key)


def _cited_chunk_ids(data: Any) -> list[str]:
    """Collect every passage id the model cited as evidence, anywhere in its output.

    These are the model's own citations — a rough, independent signal of which
    passages were actually useful. It scans the whole output for any field whose
    name contains ``chunk_id``, covering the ``*_chunk_id`` naming convention, a
    structured ``evidence.chunk_id`` field, and list-valued ``*_chunk_ids``.
    """
    out: list[str] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if "chunk_id" in key and isinstance(value, str) and value:
                    out.append(value)
                elif "chunk_id" in key and isinstance(value, list):
                    out.extend(v for v in value if isinstance(v, str) and v)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data or {})
    return out


def _read_text_or_empty(path) -> str:
    try:
        from pathlib import Path

        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _the_word_null_is_not_a_value(parsed: Any) -> Any:
    """Turn a model's literal ``"null"`` string into an actual null, recursively.

    A model answering under a JSON-object instruction writes the four characters n-u-l-l
    inside quotes often enough to matter, and every prompt skeleton in this library shows
    nullable fields as quoted descriptions, which is what teaches it to. The string is
    then not null to anything downstream: a recipe's ``stop_if`` sees a value and ends
    the recipe, the schema sees a string where a date belongs and fails the variable, and
    a python finalize that would have normalised it never runs. Seen in the wild:
    a living patient, ``date_of_death: "null"``, six of seven steps skipped, and
    only the output schema catching it.

    ONLY the exact string, and only ``null``. Not ``"none"`` -- "none" is a real clinical
    answer ("prior therapy: none") and coercing it would erase findings. Not ``""`` --
    an empty string is a model declining to answer, which is different from answering
    that there is nothing, and the schemas already distinguish them.
    """
    if isinstance(parsed, dict):
        return {key: _the_word_null_is_not_a_value(value) for key, value in parsed.items()}
    if isinstance(parsed, list):
        return [_the_word_null_is_not_a_value(item) for item in parsed]
    if isinstance(parsed, str) and parsed.strip().lower() == "null":
        return None
    return parsed


def _best_effort_json(content: str) -> tuple[Any, str | None]:
    """Pull the JSON object out of a model response, even when it's wrapped in prose.

    We ask the model for JSON, but smaller locally run models often surround it
    with chatter ("Here is the JSON:\n```json…") or cut off the closing code
    fence. So instead of insisting the whole response be clean JSON, we hunt for
    the JSON inside it: prefer a fenced ```json block if present, otherwise take
    the text from the first ``{`` to the last ``}``.
    """
    if not content or not content.strip():
        return None, "empty content"
    text = content.strip()

    # 1) Prefer a fenced ```...``` code block if one appears anywhere.
    if "```" in text:
        after = text.split("```", 1)[1]
        if "\n" in after:
            first_line, rest = after.split("\n", 1)
            if first_line.strip().lower() in ("json", ""):
                after = rest
        after = after.split("```", 1)[0].strip()  # stop at the closing fence if there is one
        if after:
            try:
                return _the_word_null_is_not_a_value(json.loads(after)), None
            except json.JSONDecodeError:
                pass  # fall through to scanning for { ... } braces instead

    # 2) Otherwise take the first '{' onward; trim to the last '}' on failure.
    start = text.find("{")
    if start != -1:
        try:
            return _the_word_null_is_not_a_value(json.loads(text[start:])), None
        except json.JSONDecodeError:
            end = text.rfind("}")
            if end > start:
                try:
                    return _the_word_null_is_not_a_value(json.loads(text[start : end + 1])), None
                except json.JSONDecodeError as e:
                    return None, f"JSONDecodeError: {e}"

    # 3) Last resort: the whole string.
    try:
        return _the_word_null_is_not_a_value(json.loads(text)), None
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
