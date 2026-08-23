"""Final reconciliation step for genetics_germline.

The earlier "extract_germline" pass (one retrieve-then-ask-the-model attempt) lists the
hereditary-panel result — which germline genes were tested and any pathogenic findings.
This step passes that data through UNCHANGED and stamps a top-level
``evidence_chunk_id`` so the ``genetic_testing_done`` determination is backed by
evidence, even the negative "no germline testing documented" default where ``genes`` is
empty and no per-gene citation exists. (The citing chunk is the model's cited chunk if
it cited one the pass actually read; nothing otherwise.) Each listed
gene keeps its own per-gene evidence_chunk_id.

This step runs pure Python only — it does not call the language model.
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
    st = (ctx.step_outputs or {}).get("extract_germline") or {}
    data: dict[str, Any] = dict(st.get("data") or {})
    read = st.get("evidence_chunk_ids") or []
    data["evidence_chunk_id"] = _grounding_chunk(data, read)
    # Back each listed gene with the chunk it was read from (its model citation if that is
    # one the pass read; null otherwise) so a positive gene the model forgot to
    # cite is still backed by evidence rather than shipped with none. Build new dicts so the
    # upstream step_outputs are left unchanged.
    data["genes"] = [
        ({**g, "evidence_chunk_id": _grounding_chunk(g, read)} if isinstance(g, dict) else g)
        for g in (data.get("genes") or [])
    ]
    return StepResult(data=data, receipt_payload={"python_finalize": {
        "genetic_testing_done": data.get("genetic_testing_done"),
        "evidence_chunk_id": data["evidence_chunk_id"],
    }})
