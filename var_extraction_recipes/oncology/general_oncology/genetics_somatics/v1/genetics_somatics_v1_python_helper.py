"""Finalize step for genetics_somatics.

(An evidence_chunk_id is a pointer back to the exact chunk a value came from, so
every value is traceable to the chart.)

The listing pass emits the somatic / tumor-panel result; this finalize step (no
language-model call) passes that data through UNCHANGED and stamps a top-level
``evidence_chunk_id`` so the ``somatic_testing_done`` determination is backed by
evidence — even the negative "no somatic testing documented" default, where ``genes``
is empty and there is no per-gene pointer. That supporting chunk is the model's cited
chunk if it cited one that the pass actually read, and nothing when it cited
pass read. Each listed gene keeps its own per-gene evidence_chunk_id. No language-model
call.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

_grounding_chunk = load_shared_validation_rule("evidence_grounding", __file__).grounding_chunk


def finalize(ctx: StepContext) -> StepResult:
    st = (ctx.step_outputs or {}).get("extract_somatic") or {}
    data: dict[str, Any] = dict(st.get("data") or {})
    read = st.get("evidence_chunk_ids") or []
    data["evidence_chunk_id"] = _grounding_chunk(data, read)
    # Back each listed gene with the chunk it was read from (the model's cited chunk if
    # that is one the pass read; null otherwise) so a positive gene the
    # model forgot to cite still has supporting evidence rather than shipping with none.
    # Build new dicts so the upstream step_outputs are left unchanged.
    data["genes"] = [
        ({**g, "evidence_chunk_id": _grounding_chunk(g, read)} if isinstance(g, dict) else g)
        for g in (data.get("genes") or [])
    ]
    return StepResult(data=data, receipt_payload={"python_finalize": {
        "somatic_testing_done": data.get("somatic_testing_done"),
        "evidence_chunk_id": data["evidence_chunk_id"],
    }})
