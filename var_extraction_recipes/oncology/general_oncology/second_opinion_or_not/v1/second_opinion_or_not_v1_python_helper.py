"""Final reconciliation step for second_opinion_or_not — combine the two search passes
into the single yes/no output.

Two earlier passes each keyword-searched the notes (a "pass" is one
retrieve-then-ask-the-model attempt): one to identify the treating oncologist, one to
decide whether a second opinion occurred. An "evidence_chunk_id" cites the chunk
that backs a claim.

  * second_opinion: the deciding pass's yes/no (null only when it cannot be decided);
  * treating_oncologist: carried from the identification pass, for an evidence trail;
  * consulting_institution / reason / evidence_chunk_id: from the deciding pass.

This step runs pure Python only — it does not call the language model.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

# Cite the chunk the DECIDING pass read, and only that one. Falling back to the
# oncologist-identification pass's chunk when the deciding pass cited nothing usable
# looked like "never ship without evidence", but it names a passage the deciding pass
# was never shown: the value then satisfies the grounding gate by construction and ships
# ok=true beside a quote that does not support it. An uncited determination is nulled and
# reported by step 8 instead, which is the honest answer.
_grounding_chunk = load_shared_validation_rule("evidence_grounding", __file__).grounding_chunk


def finalize(ctx: StepContext) -> StepResult:
    steps = ctx.step_outputs or {}

    def pass_of(step_id: str) -> tuple[dict[str, Any], list[str]]:
        st = steps.get(step_id) or {}
        return (st.get("data") or {}), (st.get("evidence_chunk_ids") or [])

    onc, onc_read = pass_of("identify_oncologist")
    dec, dec_read = pass_of("decide_second_opinion")

    decided = dec.get("second_opinion")
    second_opinion = decided if isinstance(decided, bool) else None

    out = {
        "second_opinion": second_opinion,
        "treating_oncologist": onc.get("treating_oncologist"),
        "consulting_institution": dec.get("consulting_institution"),
        "reason": (dec.get("reason") or "")[:400],
        "evidence_chunk_id": _grounding_chunk(dec, dec_read),
    }
    receipt = {"python_finalize": {"second_opinion": second_opinion}}
    return StepResult(data=out, receipt_payload=receipt)
