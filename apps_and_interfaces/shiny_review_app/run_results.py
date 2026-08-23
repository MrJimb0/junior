"""Load a real pipeline run's per-variable ``result.json`` for the review app.

The pipeline writes, per variable, an artifact envelope at
``<data_root>/CONTAINS_PHI/pipeline_run_receipts/<run_id>/patients/<pid>/
extract/<variable>/result.json``. This module reads those envelopes, flattens
the per-recipe ``payload.data`` into a uniform display row, and resolves the
evidence ``chunk_id``s back to readable text via ``PatientChunkStore`` — the
same chunk ids every downstream consumer of a judgement joins on.

Run selection (first match wins):
  * ``$JR_REVIEW_RUN_ID`` / ``$JR_REVIEW_PATIENT_ID`` if set;
  * otherwise the newest run dir that has extract output, first patient.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.cohort_runner import RUN_ID_PATTERN
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
    pipeline_run_receipts_root,
    validate_patient_id,
)

REPO = Path(__file__).resolve().parents[2]

# Anchor the data root absolutely so the loader is independent of the launch
# cwd. Mirrors jr_pipeline's JR_DATA_ROOT (default ./data) but resolved to REPO.
DATA_ROOT = Path(os.environ.get("JR_DATA_ROOT") or (REPO / "data")).expanduser()


# ── flatten the per-recipe payload.data into one display value ──────────────
# Each recipe's output schema differs, so the flattener must work over an OPEN
# set of shapes. It separates three concerns regardless of nesting:
#   * value  — the extracted clinical content, rendered for the reviewer;
#   * evidence chunk ids — every ``*evidence_chunk_id`` pointer, anywhere;
#   * confidence + rationale — the structural qualifiers, surfaced on the side.
# Skip lists are EXACT keys only. A suffix/prefix denylist (as step 8's provenance
# grouping uses) over-matches real fields here — e.g. ``er_status``/``her2_status``,
# ``n_primaries``, ``estimated_dob_method`` — and would silently hide them.
_VALUE_SKIP_KEYS = frozenset({
    "ok", "warnings", "errors", "error", "type", "required", "description",
    "method", "status", "additionalproperties", "title",
})
_RATIONALE_KEYS = frozenset({"rationale", "reason", "notes", "note"})
_MAX_VALUE_CHARS = 2000  # bound the rendered value (a deep payload can't blow up the cell)


def _is_confidence_key(kl: str) -> bool:
    return kl == "confidence" or kl.endswith("_confidence") or kl.endswith("_certainty")


def _is_rationale_key(kl: str) -> bool:
    return kl in _RATIONALE_KEYS or kl.endswith("_rationale")


def _skip_for_display(kl: str) -> bool:
    """Keys that are NOT extracted clinical content: the evidence pointer, the
    confidence/rationale qualifiers, the ``*_evidence`` evidence-quality enum
    (e.g. ``dob_evidence: explicit``), and a few exact structural keys. Unlike
    ``_status``/``_count``/``_method``, ``_evidence`` never names a clinical value."""
    return (
        "evidence_chunk_id" in kl
        or _is_confidence_key(kl)
        or _is_rationale_key(kl)
        or kl.endswith("_evidence")
        or kl in _VALUE_SKIP_KEYS
    )


def _schema_confidence_to_fraction(value: float) -> float:
    """Render a recipe's confidence as a 0..1 fraction. Every recipe output schema emits
    an integer 0..100 confidence, so divide by 100 and clamp — scaling by the known
    source scale, not by sniffing the value's magnitude (sniffing misreads an integer 1
    as 1.0 = 100% instead of 1%)."""
    return min(1.0, max(0.0, float(value) / 100.0))


def _is_schema_wrapper(node: dict) -> bool:
    """A serialized JSON-schema-style payload: ``{"type": "object", "properties":
    {field: {"type", "value"}}}``. Detected by a dict-valued ``properties`` plus
    ``type == object`` — tolerant of extra schema siblings ($schema, title, ...)."""
    return isinstance(node.get("properties"), dict) and node.get("type") == "object"


def _is_typed_leaf(value: Any) -> bool:
    """A ``{"type": ..., "value": ...}`` wrapper. Unwrapped to its scalar ``value``."""
    return (
        isinstance(value, dict)
        and "value" in value
        and set(value.keys()) <= {"type", "value", "description", "format"}
    )


def _unwrap(value: Any) -> Any:
    return value["value"] if _is_typed_leaf(value) else value


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _collect_meta(node: Any, chunk_ids: list[str],
                  confidences: list[float], rationales: list[str]) -> None:
    """Gather evidence chunk ids, confidences, and rationales from anywhere in the
    tree (including nested event arrays). Order-independent; cross-cutting. Recursion is
    unbounded: json.loads cannot produce a cycle, and the caller's per-row except degrades
    a pathologically deep payload to one '(unparseable result)' row."""
    if isinstance(node, dict):
        if _is_schema_wrapper(node):
            _collect_meta(node["properties"], chunk_ids, confidences, rationales)
            return
        for key, raw in node.items():
            kl = key.lower()
            unwrapped = _unwrap(raw)
            if "evidence_chunk_id" in kl:
                if isinstance(unwrapped, str) and unwrapped:
                    chunk_ids.append(unwrapped)
                elif isinstance(unwrapped, list):
                    chunk_ids.extend(x for x in unwrapped if isinstance(x, str) and x)
            elif _is_confidence_key(kl):
                if isinstance(unwrapped, (int, float)) and not isinstance(unwrapped, bool):
                    confidences.append(_schema_confidence_to_fraction(unwrapped))
            elif _is_rationale_key(kl):
                if isinstance(unwrapped, str) and unwrapped.strip():
                    rationales.append(unwrapped.strip())
            elif isinstance(unwrapped, (dict, list)):
                _collect_meta(unwrapped, chunk_ids, confidences, rationales)
    elif isinstance(node, list):
        for item in node:
            _collect_meta(item, chunk_ids, confidences, rationales)


def _render_value(node: Any) -> str:
    """Render the extracted clinical content of a node, grouping event-sequence
    arrays per item so distinct events (treatment lines, genes) stay readable."""
    if isinstance(node, dict):
        if _is_schema_wrapper(node):
            return _render_value(node["properties"])
        parts: list[str] = []
        for key, raw in node.items():
            kl = key.lower()
            if _skip_for_display(kl):
                continue
            value = _unwrap(raw)
            if isinstance(value, dict):
                sub = _render_value(value)
                if sub:
                    parts.append(f"{key}: {{{sub}}}")
            elif isinstance(value, list):
                if value and all(isinstance(x, dict) for x in value):
                    events = [s for s in (_render_value(x) for x in value) if s]
                    if events:
                        parts.append(f"{key}: " + " | ".join(f"[{i + 1}] {e}" for i, e in enumerate(events)))
                else:
                    scalars = [_fmt_scalar(x) for x in value if x not in (None, "")]
                    if scalars:
                        parts.append(f"{key}: " + "; ".join(scalars))
            elif value not in (None, ""):
                parts.append(f"{key}: {_fmt_scalar(value)}")
        return "; ".join(parts)
    if isinstance(node, list):
        if node and all(isinstance(x, dict) for x in node):
            events = [s for s in (_render_value(x) for x in node) if s]
            return " | ".join(f"[{i + 1}] {e}" for i, e in enumerate(events))
        return "; ".join(_fmt_scalar(x) for x in node if x not in (None, ""))
    if node not in (None, ""):
        return _fmt_scalar(node)
    return ""


def _value_bearing_top_keys(data: dict) -> list[str]:
    """Top-level keys that hold rendered clinical content (not evidence/qualifiers)."""
    out = []
    for key, raw in data.items():
        kl = key.lower()
        if _skip_for_display(kl):
            continue
        if _unwrap(raw) not in (None, ""):
            out.append(key)
    return out


def _summarize_data(data: Any, variable: str) -> tuple[str, float | None, list[str], str]:
    """Return (display_value, confidence, ordered_unique_chunk_ids, rationale)."""
    chunk_ids: list[str] = []
    confidences: list[float] = []
    rationales: list[str] = []
    _collect_meta(data, chunk_ids, confidences, rationales)

    # A flat single-field result (e.g. date_of_birth) shows just its value.
    if isinstance(data, dict) and _value_bearing_top_keys(data) == [variable]:
        value = _fmt_scalar(_unwrap(data[variable]))
    else:
        value = _render_value(data)
    if not value.strip():
        value = "unknown"
    if len(value) > _MAX_VALUE_CHARS:
        value = value[:_MAX_VALUE_CHARS] + "…"

    # dict.fromkeys rather than the `c in seen or seen.add(c)` trick: that leans on
    # set.add returning None to mean "keep this one", which is true and unreadable.
    ordered_ids = list(dict.fromkeys(chunk_ids))
    confidence = max(confidences) if confidences else None
    return value, confidence, ordered_ids, " ".join(rationales)


# ── view model ──────────────────────────────────────────────────────────────
@dataclass
class EvidenceChunk:
    chunk_id: str
    text: str
    resolved: bool
    # Why there is no readable text for this chunk id. Kept on the record because the
    # alternative — an empty quote — reads as "the chart said nothing here", which is
    # the one thing an unreadable evidence pointer must never be mistaken for.
    text_unavailable_reason: str = ""
    # The chart this chunk points into changed after the pipeline recorded it, so every
    # offset in this run slices text the model never saw. Either integrity check sets it,
    # and both mean the source moved after embed read it. Not a per-quote problem.
    source_changed_since_embed: bool = False
    # What actually clears that mismatch, which is NOT the same command in both cases —
    # see _chart_changed_next_step. Empty when the chart still matches. Carried per chunk
    # because the run-level warning has to name a step that can succeed: an operator told
    # to re-run embed for a table that changed since ingest watches it not work.
    chart_changed_next_step: str = ""


@dataclass
class VariableRow:
    variable: str
    value: str
    confidence: float | None
    ok: bool | None
    rationale: str
    evidence: list[EvidenceChunk] = field(default_factory=list)
    model: str = ""

    @property
    def evidence_chunk_ids(self) -> list[str]:
        return [e.chunk_id for e in self.evidence]

    @property
    def evidence_source(self) -> str:
        return self.evidence[0].chunk_id if self.evidence else ""

    @property
    def evidence_text_is_readable(self) -> bool:
        """True when the first evidence chunk resolved to text a reviewer can read."""
        return bool(self.evidence and self.evidence[0].text)

    @property
    def evidence_quote(self) -> str:
        """The first evidence chunk's text, or — when it could not be resolved — the
        reason it could not be. Both the Review table and the exported CSV read this,
        so an unresolvable chunk id explains itself in the research dataset too rather
        than arriving as a blank cell."""
        if not self.evidence:
            return ""
        first = self.evidence[0]
        if first.text:
            return (first.text[:200] + "…") if len(first.text) > 200 else first.text
        if first.text_unavailable_reason:
            return f"[no evidence text] {first.text_unavailable_reason}"
        return ""


@dataclass
class LoadedRun:
    run_id: str
    patient_id: str
    rows: list[VariableRow]

    @property
    def source_label(self) -> str:
        return f"real run {self.run_id} · patient {self.patient_id}"


def _chunk_store(run_id: str, patient_id: str):
    """A PatientChunkStore for evidence-text resolution, or None if unavailable."""
    try:
        from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore

        return PatientChunkStore(patient_root=phi_patient_run_dir(run_id, patient_id, DATA_ROOT))
    except Exception:
        return None


def _evidence_text_unavailable_reason(failure: Exception) -> str:
    """One sentence a reviewer can act on, per way evidence text fails to resolve.

    An unknown chunk id, a row that is no longer there, and a chart row that changed
    since embed all used to end up as the same empty quote — so a reviewer could not
    tell a broken evidence pointer from a chart that genuinely said nothing.
    """
    if isinstance(failure, KeyError):
        return ("this evidence chunk id is not in the patient's chunk index — "
                "re-run embed and then extract for this patient")
    if isinstance(failure, IndexError):
        return ("the chart row this evidence points at is no longer there — "
                "re-run embed and then extract for this patient")
    # The chunk store's own messages already read as sentences that name the fix; use
    # the message rather than the exception's class name, which tells a reviewer nothing.
    return str(failure).strip() or "the chunk store could not read this evidence chunk"


# The two integrity checks PatientChunkStore makes before it hands back evidence text, and
# the command that clears each. They are different failures: a chart ROW whose text changed
# after embed hashed it invalidates that chunk's offsets, and re-running embed re-cuts them;
# a structured TABLE whose bytes changed after ingest indexed it invalidates the table embed
# was built FROM, so ingest has to rebuild it first — re-running embed alone would re-cut the
# chunks out of the same wrong bytes and report success. Keyed on the phrase the store's own
# message carries, so a chunk-store failure that is neither is never handed one of these to
# follow. Each names the whole chain, because both leave the extracted values stale too.
_CHART_CHANGED_NEXT_STEPS = (
    ("changed since ingest", "re-run ingest for this patient, then embed and extract"),
    ("changed since embed", "re-run embed for this patient, then extract"),
)


def _chart_changed_next_step(failure: Exception) -> str:
    """The command that clears a changed-chart mismatch, or "" when this failure is not one."""
    message = str(failure)
    for phrase, next_step in _CHART_CHANGED_NEXT_STEPS:
        if phrase in message:
            return next_step
    return ""


def _resolve_evidence(store, chunk_id: str) -> EvidenceChunk:
    """Resolve one evidence pointer to readable text, or to the reason there is none."""
    if store is None:
        return EvidenceChunk(
            chunk_id=chunk_id, text="", resolved=False,
            text_unavailable_reason=("this patient's chunk store could not be opened, so "
                                     "no evidence text can be shown — re-run embed for "
                                     "this patient"),
        )
    try:
        text = store.text_for(chunk_id)
    except Exception as failure:  # noqa: BLE001 — every failure becomes a stated reason
        # A chart that no longer matches what the pipeline recorded invalidates this run's
        # evidence as a whole, not one quote — and which stage recorded it decides what to
        # re-run. Any other failure is one unreadable chunk and claims neither.
        next_step = _chart_changed_next_step(failure)
        return EvidenceChunk(
            chunk_id=chunk_id, text="", resolved=False,
            text_unavailable_reason=_evidence_text_unavailable_reason(failure),
            source_changed_since_embed=bool(next_step),
            chart_changed_next_step=next_step,
        )
    if not text:
        # text_for returns empty for a source column it cannot slice (a numeric column,
        # a table row it cannot render). Still not "the chart said nothing here" — and
        # re-running anything reproduces it, so the next step is to check the value at
        # its source instead of waiting for a quote that will never arrive.
        return EvidenceChunk(
            chunk_id=chunk_id, text="", resolved=True,
            text_unavailable_reason=(
                "this evidence points at a chart field with no text to quote — a number, a "
                "date, or a table row that rendered empty — so confirm this value against "
                "the row this chunk id names in the patient's source table"
            ),
        )
    return EvidenceChunk(chunk_id=chunk_id, text=text, resolved=True)


def load_run_variables(run_id: str, patient_id: str) -> list[VariableRow]:
    """Load every ``extract/<variable>/result.json`` for one patient as display rows."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return []
    # phi_patient_run_dir validates the patient id and honors JR_SENSITIVE_DATA_LABEL; the
    # run-id check above keeps a spoofed picker input from walking the run segment out of
    # the receipts root (patient_id has its own chokepoint, run_id did not).
    extract_dir = extract_output_dir(phi_patient_run_dir(run_id, patient_id, DATA_ROOT))
    if not extract_dir.is_dir():
        return []
    store = _chunk_store(run_id, patient_id)
    rows: list[VariableRow] = []
    for var_dir in sorted(p for p in extract_dir.iterdir() if p.is_dir()):
        result_file = var_dir / "result.json"
        if not result_file.is_file():
            continue
        try:
            envelope = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else {}
        variable = payload.get("variable") or var_dir.name
        try:
            value, confidence, chunk_ids, rationale = _summarize_data(payload.get("data") or {}, variable)
        except Exception:
            # A malformed / pathological result.json (incl. RecursionError) must
            # degrade one row, not crash the whole load.
            value, confidence, chunk_ids, rationale = "(unparseable result)", None, [], ""
        evidence = [_resolve_evidence(store, cid) for cid in chunk_ids]
        recipe = payload.get("recipe") or {}
        model = f"{recipe.get('name', variable)} {recipe.get('version', '')}".strip()
        rows.append(VariableRow(
            variable=variable, value=value, confidence=confidence,
            ok=payload.get("ok"), rationale=rationale, evidence=evidence, model=model,
        ))
    return rows


def _iter_patients_with_extract(run_dir: Path) -> Iterator[str]:
    """Patient ids under ``run_dir`` that produced at least one extract result.json, in
    sorted order and yielded lazily so a caller needing only existence (or the first) can
    stop after one hit instead of scanning the whole patient tree."""
    patients_dir = run_dir / "patients"
    if not patients_dir.is_dir():
        return
    for p in sorted(patients_dir.iterdir()):
        if p.is_dir() and any(extract_output_dir(p).glob("*/result.json")):
            yield p.name


def find_latest_run_id() -> str | None:
    """Newest canonical run dir that produced at least one extract result.json.

    The run id is fixed-width (``YYYYMMDD_HHMMSS_<hex>``), so lexicographic order is
    chronological. RUN_ID_PATTERN (owned by cohort_runner) excludes model-comparison scratch
    dirs (``<base>__compare__<name>``), which must never be auto-selected as the latest run.
    """
    root = pipeline_run_receipts_root(DATA_ROOT)
    if not root.is_dir():
        return None
    run_ids = sorted(
        (d.name for d in root.iterdir() if d.is_dir() and RUN_ID_PATTERN.fullmatch(d.name)),
        reverse=True,
    )
    # Newest first: return the first run with any extract output. Both the run scan and each
    # run's patient scan short-circuit, so a boot does not walk every run's whole tree.
    for run_id in run_ids:
        if next(_iter_patients_with_extract(root / run_id), None) is not None:
            return run_id
    return None


def resolve_run() -> LoadedRun | None:
    """Resolve the run the review app should load, honoring env overrides, else the
    newest run on disk. Returns None when there is no usable run."""
    run_id = os.environ.get("JR_REVIEW_RUN_ID") or find_latest_run_id()
    if not run_id:
        return None
    run_dir = pipeline_run_receipts_root(DATA_ROOT) / run_id
    patient_id = os.environ.get("JR_REVIEW_PATIENT_ID")
    if patient_id:
        # Validate an env-supplied id BEFORE it is joined into any path (fail closed).
        patient_id = validate_patient_id(patient_id)
    else:
        # First patient with extract output; short-circuits rather than re-scanning the
        # whole winning run's tree that find_latest_run_id already probed.
        patient_id = next(_iter_patients_with_extract(run_dir), None)
        if patient_id is None:
            return None
    rows = load_run_variables(run_id, patient_id)
    if not rows:
        return None
    return LoadedRun(run_id=run_id, patient_id=patient_id, rows=rows)


def list_reviewable_runs() -> list[str]:
    """Run ids (newest first) that have at least one patient with extract output —
    the runs the review picker offers. Empty when the receipts root is absent. Each
    run's patient scan short-circuits on the first hit."""
    root = pipeline_run_receipts_root(DATA_ROOT)
    if not root.is_dir():
        return []
    runs = sorted(
        (d.name for d in root.iterdir() if d.is_dir() and RUN_ID_PATTERN.fullmatch(d.name)),
        reverse=True,
    )
    return [r for r in runs if next(_iter_patients_with_extract(root / r), None) is not None]


def list_patients_with_extract(run_id: str) -> list[str]:
    """Patient ids under ``run_id`` that produced extract output, sorted. Empty when the
    run has none or the run id is not a canonical run id. Validating the id keeps a
    spoofed picker input from walking out of the receipts root (the sibling scans above
    filter directory names by the same pattern)."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        return []
    return list(_iter_patients_with_extract(pipeline_run_receipts_root(DATA_ROOT) / run_id))


def rows_as_csv(rows: list[VariableRow], patient_id: str, run_id: str, source: str) -> str:
    """The Review-tab CSV export.

    ``run_id`` and ``source`` are written on every row and are not optional: a
    downloaded export becomes a row in somebody's research dataset, and one taken
    from a selection that produced no readable output must never be
    indistinguishable from a real run's.
    """
    import csv
    import io

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow([
        "run_id", "source", "patient_id", "variable", "value", "confidence", "ok",
        "evidence_chunk_ids", "evidence_quote", "rationale", "model",
    ])
    for r in rows:
        writer.writerow([
            run_id, source, patient_id, r.variable, r.value,
            "" if r.confidence is None else f"{r.confidence:.2f}",
            "" if r.ok is None else ("ok" if r.ok else "failed"),
            " | ".join(r.evidence_chunk_ids), r.evidence_quote, r.rationale, r.model,
        ])
    return out.getvalue()
