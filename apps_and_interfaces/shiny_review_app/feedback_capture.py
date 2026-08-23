"""Human review capture for the workbench.

Two affordances, both capture-now-irreversible at a remote site:
  * ``extraction_correction`` — flag a value wrong + supply the correct one.
  * ``extraction_confirmation`` — reviewed and AGREED. Without this, the only
    human channel records errors, so the denominator for sensitivity (TP/(TP+FN))
    is a non-random, error-enriched convenience sample and sensitivity is
    unmeasurable *by construction*. The confirm action supplies the agreed cases.

Each entry carries the reviewer's identity + tier (expert/trainee), because
accuracy-weighting labels later needs to know *who* labeled. And a per-run
**sampling frame** records which (patient, variable) were drawn for review and by
what rule — the random draw the denominator requires.

One file per (patient, variable), in the shape ``junior eval-values`` reads
(``ground_truth_evaluation``): top-level ``patient_id``/``variable``/``run_id``/
``annotator`` + a ``feedback`` list. Every writer requires a real pipeline run id —
a label that names no run can never be joined back to the extraction it judged.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
)

ANNOTATOR = "shiny_review_app"
REVIEWER_TIERS = ("expert", "trainee", "unknown")


def _file_lock(path: Path) -> FileLock:
    """A cross-process lock around the read-modify-write so two reviewers appending to
    the same (patient, variable) file never lose an entry."""
    return FileLock(str(path) + ".lock", timeout=30)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a unique temp file + os.replace so a crash mid-write never leaves a
    truncated, irreversible expert-label file."""
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def feedback_dir(run_id: str, dr: Path | None = None) -> Path:
    """Resolve the PHI-side corrections dir for one run."""
    return clinician_feedback_dir(run_id, dr)


def _require_run_id(run_id: str | None) -> str:
    if not run_id:
        raise ValueError(
            "a feedback record needs the run id of the extraction it judges — "
            "without one it can never be joined back to what was reviewed"
        )
    return run_id


def _reviewer(reviewer_id: str | None, reviewer_role: str | None) -> tuple[str, str]:
    rid = reviewer_id or os.environ.get("JR_REVIEWER_ID") or ANNOTATOR
    role = (reviewer_role or os.environ.get("JR_REVIEWER_ROLE") or "unknown").lower()
    if role not in REVIEWER_TIERS:
        role = "unknown"
    return rid, role


def _append_entry(
    *, patient_id: str, variable: str, entry: dict[str, Any],
    run_id: str, reviewer_id: str, dr: Path | None,
) -> Path:
    """Append one feedback entry to the (patient, variable) file, preserving history."""
    out_dir = feedback_dir(run_id, dr)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{patient_id}__{variable}.json"

    # The whole read-modify-write is under one lock + an atomic replace, so a
    # second reviewer's concurrent append cannot clobber the first's (lost-update) and a
    # crash mid-write cannot truncate an irreversible expert-label file.
    with _file_lock(path):
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            record.setdefault("feedback", []).append(entry)
        else:
            record = {
                "patient_id": patient_id,
                "variable": variable,
                "run_id": run_id,
                "annotator": reviewer_id,
                "feedback": [entry],
            }
        _atomic_write_text(path, json.dumps(record, indent=2, ensure_ascii=False))
    return path


def write_correction(
    *,
    patient_id: str,
    variable: str,
    correct_value: str,
    original_value: str | None = None,
    run_id: str | None = None,
    reviewer_id: str | None = None,
    reviewer_role: str | None = None,
    model: str | None = None,
    extracted_at: str | None = None,
    evidence_chunk_id: str | None = None,
    dr: Path | None = None,
) -> Path:
    """Record that a reviewer flagged a value wrong and supplied the correct one.
    Carries provenance (the model, when it was extracted, the evidence pointer) so the
    correction is attributable to a specific extraction."""
    run_id = _require_run_id(run_id)
    rid, role = _reviewer(reviewer_id, reviewer_role)
    entry: dict[str, Any] = {
        "type": "extraction_correction",
        "field": variable,
        "original_value": original_value,
        "correct_value": correct_value,
        "reviewer_id": rid,
        "reviewer_role": role,
        "model": model,
        "extracted_at": extracted_at,
        "evidence_chunk_id": evidence_chunk_id,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return _append_entry(patient_id=patient_id, variable=variable, entry=entry,
                         run_id=run_id, reviewer_id=rid, dr=dr)


def write_confirmation(
    *,
    patient_id: str,
    variable: str,
    reviewed_value: str | None = None,
    run_id: str | None = None,
    reviewer_id: str | None = None,
    reviewer_role: str | None = None,
    dr: Path | None = None,
) -> Path:
    """Record that a reviewer reviewed the value and AGREED (the confirm-correct
    signal). Supplies the agreed cases the sensitivity denominator needs."""
    run_id = _require_run_id(run_id)
    rid, role = _reviewer(reviewer_id, reviewer_role)
    entry: dict[str, Any] = {
        "type": "extraction_confirmation",
        "field": variable,
        "reviewed_value": reviewed_value,
        "agreed": True,
        "reviewer_id": rid,
        "reviewer_role": role,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return _append_entry(patient_id=patient_id, variable=variable, entry=entry,
                         run_id=run_id, reviewer_id=rid, dr=dr)


def write_chunk_relevance(
    *,
    patient_id: str,
    variable: str,
    chunk_id: str,
    relevant: bool,
    run_id: str | None = None,
    reviewer_id: str | None = None,
    reviewer_role: str | None = None,
    dr: Path | None = None,
) -> Path:
    """Record a reviewer's relevant/not-relevant judgment on one evidence chunk.
    PHI-side (raw chunk_id); the app also mints the NO_PHI counterpart via
    ``emit_relevance_label_exhaust``."""
    run_id = _require_run_id(run_id)
    rid, role = _reviewer(reviewer_id, reviewer_role)
    entry: dict[str, Any] = {
        "type": "chunk_relevance",
        "field": variable,
        "chunk_id": chunk_id,
        "relevant": bool(relevant),
        "reviewer_id": rid,
        "reviewer_role": role,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return _append_entry(patient_id=patient_id, variable=variable, entry=entry,
                         run_id=run_id, reviewer_id=rid, dr=dr)


def emit_relevance_label_exhaust(
    *,
    patient_id: str,
    variable: str,
    chunk_id: str,
    relevant: bool,
    run_id: str,
    reviewer_id: str | None = None,
    reviewer_role: str | None = None,
    dr: Path | None = None,
) -> Path:
    """Mint the NO_PHI exhaust counterpart of one chunk_relevance judgment: a
    ``relevance_label`` record whose chunk surrogate (shared evidence-chunk domain)
    joins back to the run's ``selection_judgment`` ranks.

    Call only with a REAL pipeline run id: the record must join a real run's
    selection ranks, and emitting derives that run's PHI-side surrogate secret.
    ``result.json`` does not record which recipe step produced an evidence pointer,
    so ``step_id`` is ``unknown``; the join key that matters (the chunk surrogate)
    does not depend on it. A write-gate failure raises; the caller surfaces it,
    never drops it silently."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        record_relevance_label,
    )

    rid, role = _reviewer(reviewer_id, reviewer_role)
    return record_relevance_label(
        run_id=run_id,
        patient_id=patient_id,
        reviewer_id=rid,
        chunk_id=chunk_id,
        label="relevant" if relevant else "not_relevant",
        recipe_id=variable,
        step_id="unknown",
        reviewer_tier=role,
        data_root=dr,
    )


def write_sampling_frame(
    *,
    drawn: list[dict[str, str]],
    rule: str,
    run_id: str | None = None,
    dr: Path | None = None,
) -> Path:
    """Record which (patient, variable) pairs were drawn for review and by what rule.

    The sampling frame is the random draw that turns confirmations + corrections into
    an unbiased denominator for sensitivity/PPV — without it, reviewed cases are a
    convenience sample. ``drawn`` is a list of {patient_id, variable}; ``rule``
    is the sampling rule (e.g. "random_10_per_variable", "all_low_confidence").

    There is ONE frame per run and every call MERGES into it. A reviewer walks a cohort
    one patient at a time, so each call carries only the patient on screen: replacing
    the file would leave a 40-patient review with a frame naming one patient, and a
    denominator that is wrong in the direction nobody checks. Pairs are unioned on
    (patient_id, variable) so re-recording a patient does not double-count them,
    ``drawn_at`` stays the moment the frame was opened, and ``n_drawn`` is recounted
    from the merged set."""
    run_id = _require_run_id(run_id)
    out_dir = feedback_dir(run_id, dr)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "_sampling_frame.json"

    # Read-modify-write under one lock, as for the per-(patient, variable) files: two
    # reviewers drawing at the same time must both end up in the frame.
    with _file_lock(path):
        previous: dict[str, Any] = {}
        if path.exists():
            # Refuse rather than start a fresh frame over it: a frame that silently
            # lost the patients already drawn is the defect this merge exists to stop.
            # Every way the file can be damaged means one thing to the reviewer, so all
            # of them have to end here, as this sentence. The shape is checked all the
            # way down to the entries: a `drawn` holding null, or strings, or a mapping
            # reaches `.get` on the merge below and comes out as a Python TypeError in
            # the middle of a review session.
            cannot_read = (
                f"The review sample already recorded for this run cannot be read: {path}\n"
                "Move that file aside and record the sample again — every patient "
                "already drawn for this run has to be re-recorded."
            )
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                raise RuntimeError(cannot_read) from None
            if not isinstance(previous, dict):
                raise RuntimeError(cannot_read)
            already_drawn = previous.get("drawn", [])
            if not isinstance(already_drawn, list) or not all(
                isinstance(pair, dict) for pair in already_drawn
            ):
                raise RuntimeError(cannot_read)
        earlier_rule = previous.get("rule")
        if earlier_rule and earlier_rule != rule:
            raise RuntimeError(
                f"This run's review sample was drawn by the rule '{earlier_rule}' and this "
                f"draw uses '{rule}'. Two rules in one frame make the denominator "
                f"uninterpretable. Review this draw under its own run, or move {path} "
                "aside if the earlier draw was a mistake."
            )
        # Keyed union: an entry already in the frame keeps its position, so the file
        # reads in the order the reviewer walked the cohort.
        merged = {(pair.get("patient_id"), pair.get("variable")): pair
                  for pair in list(previous.get("drawn") or []) + list(drawn)}
        record = {
            "run_id": run_id,
            "rule": rule,
            # The first draw's timestamp, carried forward: it is when the frame was
            # opened, which is what dates the sample.
            "drawn_at": previous.get("drawn_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_drawn": len(merged),
            "drawn": list(merged.values()),
        }
        _atomic_write_text(path, json.dumps(record, indent=2, ensure_ascii=False))
    return path


def read_sampling_frame(path: Path) -> dict[str, Any]:
    """Read back a run's sampling frame — the record ``write_sampling_frame`` returns the
    path to. Lets a caller report the cohort-wide total it just added to, rather than the
    handful of pairs it passed in."""
    return json.loads(path.read_text(encoding="utf-8"))
