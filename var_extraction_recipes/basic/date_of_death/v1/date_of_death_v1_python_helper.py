"""Final reconciliation step for date_of_death — combine three search passes.

Three earlier passes each looked at the chart in a different way (a "pass" is one
retrieve-then-ask-the-model attempt): pass 1 read the structured demographics row,
pass 2 keyword-searched the notes for an explicit death statement, and pass 3 judged
the patient's overall trajectory. This step merges their answers. An
"evidence_chunk_id" points at the specific chunk that backs a claim.

  * an EXPLICIT, date-shaped date_of_death from pass 1 (demographics) or pass 2
    (death search) is preserved — inference (pass 3) never sets a date;
  * vital_status is the latest definite signal (dead/alive over unknown);
  * confidence is the last reported confidence;
  * evidence_chunk_id grounds the determination in the demographics row / note the
    deciding pass actually read (vital_status is a real clinical claim, so it must
    carry its own citation for the later provenance check in step 8, which audits that
    every claim points at real evidence the model was actually shown).

The grounding chunk is the model's own citation when it names a chunk we actually
showed it; otherwise the top chunk that pass read (so a flaky/absent citation from a
small local model still yields an honest citation, and a made-up chunk id the
model was never shown is rejected). The strongest claim wins the object-level pointer:
an explicit date grounds on the pass that found it, else the winning vital_status does.

This step runs pure Python only — it does not call the language model.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

# Date-shaped check (shared parsing): only an explicit date sets date_of_death.
_pd = load_shared_validation_rule("partial_date", __file__)
_is_date = _pd.is_date


def _grounding_chunk(pass_data: dict[str, Any], read_chunk_ids: list[str]) -> str | None:
    """The chunk that grounds this pass's determination: the model's cited chunk when
    it is one we actually showed the pass, otherwise the top chunk the pass read."""
    cited = pass_data.get("evidence_chunk_id")
    if isinstance(cited, str) and cited and cited in read_chunk_ids:
        return cited
    return read_chunk_ids[0] if read_chunk_ids else None


def merge_passes(ctx: StepContext) -> StepResult:
    steps = ctx.step_outputs or {}

    def pass_of(step_id: str) -> tuple[dict[str, Any], list[str]]:
        s = steps.get(step_id) or {}
        return (s.get("data") or {}), (s.get("evidence_chunk_ids") or [])

    p1, r1 = pass_of("demographics_grab")
    p2, r2 = pass_of("death_search")
    p3, r3 = pass_of("vital_status_inference")
    # The holistic llm_only pass reads structured summaries rather than raw chunks.
    # Treat the chunks read by those summary passes as its admissible citation set.
    for summary_step in ("eol_trajectory", "performance_status", "recent_activity"):
        _summary_data, summary_read = pass_of(summary_step)
        for chunk_id in summary_read:
            if chunk_id not in r3:
                r3.append(chunk_id)

    # Explicit date only from passes 1-2; the trajectory inference (p3) never sets a
    # date. Remember which pass supplied it so we can cite the chunk it was read from.
    date_of_death = None
    date_grounding = None
    for data, read in ((p1, r1), (p2, r2)):
        if _is_date(data.get("date_of_death")):
            date_of_death = data["date_of_death"]
            date_grounding = _grounding_chunk(data, read)

    # Vital-status precedence: an EXPLICIT death -- a structured deceased indicator
    # (p1 demographics) or an explicit death statement (p2 death_search) -- is
    # authoritative and may NOT be downgraded by the soft trajectory inference (p3).
    # p3 (and any "alive"/"unknown") only decides when no explicit-death pass fired.
    vital_status = "unknown"
    status_grounding = None
    explicit_dead = next(
        ((data, read) for data, read in ((p1, r1), (p2, r2)) if data.get("vital_status") == "dead"),
        None,
    )
    if explicit_dead is not None:
        vital_status = "dead"
        status_grounding = _grounding_chunk(explicit_dead[0], explicit_dead[1])
    else:
        # No explicit death -> the latest definite alive/unknown signal (p3 is the holistic arbiter).
        for data, read in ((p1, r1), (p2, r2), (p3, r3)):
            if data.get("vital_status") in ("dead", "alive"):
                vital_status = data["vital_status"]
                status_grounding = _grounding_chunk(data, read)

    # An "unknown" final status is still backed by the chart that was examined: cite
    # the most recent pass that actually read a chunk (p3 -> p2 -> p1).
    if status_grounding is None:
        for data, read in ((p3, r3), (p2, r2), (p1, r1)):
            g = _grounding_chunk(data, read)
            if g:
                status_grounding = g
                break

    # The strongest claim owns the object-level pointer: an explicit date if present,
    # else the vital-status determination.
    evidence_chunk_id = date_grounding or status_grounding

    # Reconcile the two fields: a death date is only meaningful
    # when the patient is confirmed dead. When the holistic vital_status is not "dead", an
    # explicit date from a partial pass is an over-call the inference overrode (the
    # death-search pass can over-read metastatic / palliative / advance-directive context
    # as a death) -> drop the date and ground the object on the vital-status determination,
    # so we never ship an alive/unknown patient with a death date.
    if vital_status != "dead":
        date_of_death = None
        evidence_chunk_id = status_grounding

    confidence = None
    for data in (p1, p2, p3):
        if isinstance(data.get("confidence"), int) and not isinstance(data.get("confidence"), bool):
            confidence = data["confidence"]

    out = {
        "date_of_death": date_of_death,
        "vital_status": vital_status,
        "evidence_chunk_id": evidence_chunk_id,
        "confidence": confidence,
    }
    receipt = {"python_merge": {
        "date_of_death": date_of_death,
        "vital_status": vital_status,
        "evidence_chunk_id": evidence_chunk_id,
    }}
    return StepResult(data=out, receipt_payload=receipt)
