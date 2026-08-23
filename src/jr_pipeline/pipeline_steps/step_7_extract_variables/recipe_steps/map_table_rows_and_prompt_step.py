"""Map one prompt over every row of one or more ingested structured tables.

Relevance retrieval is intentionally the wrong primitive when a recipe promises
that *every* source row will be classified: a retriever may omit a row. This step
discovers the configured ingested tables, renders each row as one ``:TABLE``
evidence block, and makes one bounded, cacheable model call per row.

The handler, not the model, stamps the source table, row index, and evidence id on
each result. That keeps provenance complete even when the model forgets to cite
the row or invents a citation.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl

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
    _usage_get,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    structured_dir,
)


def _normalized_name(value: str) -> str:
    return Path(value).stem.strip().lower()


def _matched_tables(ctx: StepContext) -> list[tuple[str, Path, str, str]]:
    """Return ``(stem, parquet_path, original_source_file, doc_type)`` in stable order.

    ``doc_type`` is the document type ingest assigned this table's rows (read from
    the chunk index), so the evidence each row becomes is labeled with what the
    table actually is. A table ingest gave no single doc_type falls back to the
    neutral ``structured_table`` rather than guessing a clinical document type."""
    cfg = ctx.step.config
    exact = {
        _normalized_name(str(value))
        for value in (cfg.get("tables") or [])
        if str(value).strip()
    }
    contains = tuple(
        str(value).strip().lower()
        for value in (cfg.get("table_name_contains") or [])
        if str(value).strip()
    )
    doc_types = {
        str(value).strip().lower()
        for value in (cfg.get("doc_types") or [])
        if str(value).strip()
    }
    doc_type_contains = tuple(
        str(value).strip().lower()
        for value in (cfg.get("doc_type_name_contains") or [])
        if str(value).strip()
    )
    if not exact and not contains and not doc_types and not doc_type_contains:
        raise ValueError(
            f"step {ctx.step.id}: configure a table-name or doc-type matcher"
        )

    root = structured_dir(ctx.corpus.patient_root)
    if not root.is_dir():
        return []

    # Ingestion may give an arbitrarily named file a semantic document type
    # (e.g. results.csv rows labeled pathology_report). Preserve that ingestion
    # decision instead of forcing every site to rename the source. The same pass
    # collects each table's doc_type(s) so matched rows can be labeled with what
    # the table actually is.
    semantic_stems: set[str] = set()
    doc_types_by_name: dict[str, set[str]] = {}
    index = ctx.corpus.chunk_index
    if {"source_file", "doc_type"}.issubset(index.columns):
        for row in index.select("source_file", "doc_type").unique().iter_rows(named=True):
            source_file = row.get("source_file")
            doc_type = str(row.get("doc_type") or "").strip().lower()
            if not source_file:
                continue
            if doc_type:
                doc_types_by_name.setdefault(
                    _normalized_name(str(source_file)), set()
                ).add(doc_type)
            if doc_type in doc_types or any(
                token in doc_type for token in doc_type_contains
            ):
                semantic_stems.add(_normalized_name(str(source_file)))

    matched: list[tuple[str, Path, str, str]] = []
    for path in sorted(root.glob("*.parquet"), key=lambda item: item.name.lower()):
        stem = path.stem
        source_file = ctx.corpus.source_file_for_stem(stem) or path.name
        names = (_normalized_name(stem), _normalized_name(source_file))
        if (
            any(name in exact for name in names)
            or any(token in name for token in contains for name in names)
            or any(name in semantic_stems for name in names)
        ):
            labels: set[str] = set()
            for name in names:
                labels |= doc_types_by_name.get(name, set())
            table_doc_type = (
                next(iter(labels)) if len(labels) == 1 else "structured_table"
            )
            matched.append((stem, path, source_file, table_doc_type))
    return matched


def _combined_evidence_packet(
    packets: list[dict[str, Any]],
    *,
    patient_id: str,
    variable: str,
    max_context_tokens: int,
) -> dict[str, Any]:
    """Combine per-row packets for Step 8 without treating their union as one call.

    ``max_context_tokens`` is enforced independently while each row packet is built.
    The union may legitimately exceed that number because no model call sees the
    union.
    """
    blocks = [block for packet in packets for block in (packet.get("blocks") or [])]
    tokens_by_type: dict[str, int] = {}
    for packet in packets:
        for doc_type, tokens in (packet.get("evidence_tokens_by_doc_type") or {}).items():
            tokens_by_type[doc_type] = tokens_by_type.get(doc_type, 0) + int(tokens)
    return {
        "patient_id": patient_id,
        "variable": variable,
        "formatted_evidence_text": "\n\n".join(
            packet.get("formatted_evidence_text", "") for packet in packets
        ),
        "block_count": len(blocks),
        "total_evidence_tokens": sum(
            int(packet.get("total_evidence_tokens") or 0) for packet in packets
        ),
        "evidence_tokens_by_doc_type": tokens_by_type,
        "max_context_tokens": max_context_tokens,
        "context_scope": "per_row",
        "included_chunk_ids": [block["chunk_id"] for block in blocks],
        "blocks": blocks,
    }


@register_step("map_table_rows_and_prompt")
class MapTableRowsAndPromptHandler:
    """Call one prompt once for every row in all matched ingested tables."""

    kind = "map_table_rows_and_prompt"

    def execute(self, ctx: StepContext) -> StepResult:
        step = ctx.step
        if not step.prompt:
            raise ValueError(f"step {step.id}: 'prompt' path required")
        provider = ctx.provider
        if provider is None:
            raise RuntimeError(f"step {step.id}: no provider available")

        template = load_prompt(step.prompt)
        tables = _matched_tables(ctx)
        row_results: list[dict[str, Any]] = []
        call_receipts: list[dict[str, Any]] = []
        packets: list[dict[str, Any]] = []
        failures: list[str] = []
        # One entry per row whose answer cited something other than that row.
        rows_with_invented_citations: list[dict[str, Any]] = []
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_s": 0.0,
            "cached_calls": 0,
        }

        row_counts: dict[str, int] = {}
        for stem, parquet_path, source_file, table_doc_type in tables:
            row_count = pl.read_parquet(parquet_path).height
            row_counts[stem] = row_count
            for row_index in range(row_count):
                chunk_id = f"{ctx.patient_id}:{stem}:{row_index}:TABLE"
                row_text = ctx.corpus.text_for(chunk_id)
                direct_item = {
                    "chunk_id": chunk_id,
                    "text": row_text,
                    "source_file": source_file,
                    "doc_type": table_doc_type,
                    "record_index": row_index,
                    "chunk_index": 0,
                    "rank": 1,
                    "score": 1.0,
                }
                packet = prepare_evidence(
                    corpus=ctx.corpus,
                    reranked_candidates=[],
                    variable=f"{ctx.recipe.name}/{step.id}/row",
                    patient_id=ctx.patient_id,
                    max_context_tokens=ctx.recipe.evidence.max_context_tokens,
                    direct_evidence_items=[direct_item],
                )
                packets.append(packet)

                evidence_json = json.dumps(
                    [{"chunk_id": chunk_id, "text": row_text}],
                    ensure_ascii=False,
                )
                sys_rendered, usr_rendered = render(
                    template,
                    {
                        "patient_id": ctx.patient_id,
                        "evidence_json": evidence_json,
                        "evidence_text": packet.get("formatted_evidence_text", ""),
                        "OUTPUT_SCHEMA": _read_text_or_empty(
                            ctx.recipe.output_schema_path
                        ),
                        "vars": ctx.upstream_vars,
                        "steps": ctx.step_outputs,
                    },
                )
                messages = [
                    {"role": "system", "content": sys_rendered},
                    {"role": "user", "content": usr_rendered},
                ]
                req = LLMRequest(
                    endpoint_name=provider.provider_config().get(
                        "endpoint_name", "unknown"
                    ),
                    messages=messages,
                    temperature=ctx.recipe.llm.temperature,
                    max_tokens=ctx.recipe.llm.max_tokens,
                    response_format=ctx.recipe.llm.response_format,
                    seed=ctx.recipe.llm.seed,
                    task_name=ctx.recipe.name,
                    quarantine_path=(
                        ctx.quarantine_dir / "quarantine.jsonl"
                        if ctx.quarantine_dir
                        else None
                    ),
                    retry_cut_off_answers=ctx.recipe.llm.retry_cut_off_answers,
                )

                def _call_once(conversation, _req=req):
                    call_req = replace(_req, messages=conversation)
                    if ctx.llm_cache is not None:
                        key = LLMCache.make_key(
                            req=call_req, provider_config=provider.provider_config()
                        )
                        hit = ctx.llm_cache.get(
                            key, expected_fingerprint=ctx.recipe.llm.expected_fingerprint
                        )
                        if hit is not None:
                            return hit, True
                        fresh = provider.chat(call_req)
                        ctx.llm_cache.put(key, fresh)
                        return fresh, False
                    return provider.chat(call_req), False

                # This prompt is shown exactly one row, so that row is the only passage
                # its answer may cite.
                outcome = ask_until_grounded(
                    call_once=_call_once,
                    messages=messages,
                    shown_chunk_ids={chunk_id},
                    max_retries=int(ctx.max_provenance_retries or 0),
                    parse=_best_effort_json,
                )
                response = outcome["response"]
                parsed, parse_error = outcome["parsed"], outcome["parse_error"]
                was_cached = outcome["was_cached"]
                if outcome["attempts"]:
                    rows_with_invented_citations.append(
                        {"chunk_id": chunk_id, "attempts": outcome["attempts"]}
                    )
                if not isinstance(parsed, dict):
                    parsed = {}
                    parse_error = parse_error or "parsed response was not a JSON object"
                if parse_error:
                    failures.append(f"{chunk_id}: {parse_error}")

                # Source identity is pipeline-authored and cannot be overridden by
                # model output.
                row_results.append(
                    {
                        "source_table": stem,
                        "source_file": source_file,
                        "source_row_index": row_index,
                        "evidence_chunk_id": chunk_id,
                        "extraction": parsed,
                    }
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = _usage_get(response, key)
                    if isinstance(value, (int, float)):
                        totals[key] += int(value)
                totals["llm_s"] += float(response.latency_s or 0.0)
                totals["cached_calls"] += int(was_cached)
                call_receipts.append(
                    {
                        "chunk_id": chunk_id,
                        "source_table": stem,
                        "source_file": source_file,
                        "source_row_index": row_index,
                        "messages_sent": messages,
                        "resolved_model": response.resolved_model.to_dict(),
                        "response_raw": response.response_raw,
                        "response_parsed": parsed,
                        "validation": {
                            "parse_ok": parse_error is None,
                            "parse_error": parse_error,
                        },
                        "timings": {
                            "llm_s": response.latency_s,
                            "cached": was_cached,
                        },
                    }
                )

        combined_packet = _combined_evidence_packet(
            packets,
            patient_id=ctx.patient_id,
            variable=f"{ctx.recipe.name}/{step.id}",
            max_context_tokens=ctx.recipe.evidence.max_context_tokens,
        )
        receipt = {
            "table_matching": {
                "tables": list(step.config.get("tables") or []),
                "table_name_contains": list(
                    step.config.get("table_name_contains") or []
                ),
                "doc_types": list(step.config.get("doc_types") or []),
                "doc_type_name_contains": list(
                    step.config.get("doc_type_name_contains") or []
                ),
                "matched": [
                    {
                        "source_table": stem,
                        "source_file": source_file,
                        "doc_type": table_doc_type,
                        "row_count": row_counts[stem],
                    }
                    for stem, _path, source_file, table_doc_type in tables
                ],
            },
            "prompt": {
                "template_name": template.name,
                "template_hash": template.template_hash,
                "call_count": len(call_receipts),
            },
            "row_calls": call_receipts,
            "validation": {
                "all_rows_parsed": not failures,
                "failures": failures,
            },
            # `cached` is the boolean the run-level economics reader looks for
            # (extraction_outcome._economics_from_receipts): this step was free only
            # when every one of its row calls came from cache.
            "economics": {
                **totals,
                "cached": bool(call_receipts)
                and totals["cached_calls"] == len(call_receipts),
            },
        }
        error = (
            f"{len(failures)} table row(s) could not be parsed"
            if failures
            else None
        )
        return StepResult(
            data={
                "rows": row_results,
                "n_source_rows": len(row_results),
                "source_tables": [stem for stem, _path, _source, _doc_type in tables],
            },
            receipt_payload=receipt,
            error=error,
            step8_payload={
                "variable": f"{ctx.recipe.name}/{step.id}",
                "evidence_packet": combined_packet,
                # Rows whose answer cited a passage other than the row it was
                # shown. Empty is the ordinary case.
                "provenance_retries": rows_with_invented_citations,
            },
        )
