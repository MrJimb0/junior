"""Finalize step for ``stage`` — assemble the AJCC cancer stage at original diagnosis.

(AJCC = the standard cancer-staging body; TNM = tumor / nodes / metastasis. An
evidence_chunk_id is a pointer back to the exact chunk a value came from, so every
value is traceable to the chart.)

  * pathologic_tnm  <- pass 1 (the stage confirmed on the earliest cancer-removal
    specimen);
  * clinical_tnm    <- pass 2 (the pre-surgery stage from the notes, original
    diagnosis);
  * stage_at_diagnosis.overall_group cites the pass it ACTUALLY came from —
    pathologic staging is the more definitive read after surgery, so a pathologic
    group that has a supporting chunk wins; otherwise a clinical group that has one.
    ``basis`` and ``evidence_chunk_id`` track that same source (so the label and the
    cited evidence always agree);
  * EVERY emitted value is traceable: a TNM object or an overall stage group with no
    supporting chunk to cite is not a usable claim, so it is dropped (None) rather
    than shipped without evidence; everything is null (NOT "unknown") when not found.

No language-model call.
"""
from __future__ import annotations

from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)

# The exact set of allowed AJCC stage-group labels (must match the output_schema
# enum). "occult" is the one lowercase label; everything else is a digit / Roman
# numeral (0, I, IA, ... IV, IVC).
_GROUPS = frozenset({
    "0", "I", "IA", "IB", "II", "IIA", "IIB", "IIC",
    "III", "IIIA", "IIIB", "IIIC", "IV", "IVA", "IVB", "IVC", "occult",
})
# Case-insensitive lookup -> the standard form, so "occult"/"Occult"/"OCCULT" all
# recover the lowercase standard label (the prompts emit "occult").
_GROUP_BY_UPPER = {token.upper(): token for token in _GROUPS}


def _group(value: Any) -> str | None:
    """Return the standard AJCC stage-group label, else None. Case-insensitive;
    tolerates a leading 'Stage' word. Rejects 'unknown' / free text (-> None, which
    leaves the stage unknown)."""
    if not isinstance(value, str):
        return None
    token = value.strip().upper().replace("STAGE", "").strip()
    return _GROUP_BY_UPPER.get(token)


def _token(value: Any) -> str | None:
    """A non-empty TNM token string, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _tnm(data: dict[str, Any], keys: tuple[str, str, str]) -> dict[str, Any] | None:
    """Return a {T, N, M, evidence_chunk_id} object, built ONLY when at least one of
    T/N/M is stated AND this pass cited a supporting chunk. A TNM object with no
    supporting chunk is not a citable claim, so it is dropped (None) — same rule as
    the stage-group drop, and it is never given the OTHER pass's chunk as evidence."""
    t, n, m = (_token(data.get(k)) for k in keys)
    if t is None and n is None and m is None:
        return None
    evidence = data.get("evidence_chunk_id")
    if not evidence:
        return None
    return {keys[0]: t, keys[1]: n, keys[2]: m, "evidence_chunk_id": evidence}


def finalize(ctx: StepContext) -> StepResult:
    steps = ctx.step_outputs or {}

    def data_of(step_id: str) -> dict[str, Any]:
        return (steps.get(step_id) or {}).get("data") or {}

    p = data_of("pathologic_tnm")
    c = data_of("clinical_tnm")
    post = data_of("posttreatment_path_stage")

    pathologic_tnm = _tnm(p, ("pT", "pN", "pM"))
    clinical_tnm = _tnm(c, ("cT", "cN", "cM"))
    posttreatment_tnm = _tnm(post, ("ypT", "ypN", "ypM"))
    post_group = _group(post.get("overall_group"))
    post_rcb = (
        post.get("rcb_class")
        if post.get("rcb_class") in {"0", "I", "II", "III"}
        else None
    )
    post_effect = (
        str(post.get("treatment_effect"))[:240]
        if post.get("treatment_effect")
        else None
    )
    post_evidence = post.get("evidence_chunk_id")
    has_post_value = (
        posttreatment_tnm is not None
        or post_group is not None
        or post_rcb is not None
        or post_effect is not None
    )
    posttreatment_path_stage = None
    if has_post_value and post_evidence:
        posttreatment_path_stage = posttreatment_tnm or {
            "ypT": None,
            "ypN": None,
            "ypM": None,
            "evidence_chunk_id": post_evidence,
        }
        posttreatment_path_stage.update({
            "overall_group": post_group,
            "rcb_class": post_rcb,
            "treatment_effect": post_effect,
        })

    # The overall stage group cites the pass it came from. Pathologic staging is the
    # more definitive read after surgery, so a pathologic group with a supporting
    # chunk wins; otherwise a clinical group with one. basis + evidence track that
    # source. A group with no chunk to cite is dropped (the else branch), never
    # relabeled or given the wrong evidence pointer.
    path_group, path_evidence = _group(p.get("overall_group")), p.get("evidence_chunk_id")
    clin_group, clin_evidence = _group(c.get("overall_group")), c.get("evidence_chunk_id")
    if path_group is not None and path_evidence:
        overall_group, basis, stage_evidence = path_group, "pathologic", path_evidence
    elif clin_group is not None and clin_evidence:
        overall_group, basis, stage_evidence = clin_group, "clinical", clin_evidence
    else:
        overall_group, basis, stage_evidence = None, None, None

    out = {
        "stage_at_diagnosis": {
            "overall_group": overall_group,
            "basis": basis,
            "evidence_chunk_id": stage_evidence,
        },
        "clinical_tnm": clinical_tnm,
        "pathologic_tnm": pathologic_tnm,
        "pretreatment_path_stage": pathologic_tnm,
        "posttreatment_path_stage": posttreatment_path_stage,
        "rationale": (c.get("verbatim_stage_quote") or p.get("rationale") or "")[:600],
    }
    receipt = {"python_finalize": {
        "overall_group": overall_group,
        "basis": basis,
        "has_clinical_tnm": clinical_tnm is not None,
        "has_pathologic_tnm": pathologic_tnm is not None,
        "has_posttreatment_path_stage": posttreatment_path_stage is not None,
    }}
    return StepResult(data=out, receipt_payload=receipt)
