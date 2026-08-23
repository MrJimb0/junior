"""Read-only inspection of a finished run, for the Workbench tab.

Everything an operator inspecting one (run, patient, variable) wants to see,
loaded from the run's own artifacts through the pipeline's path helpers:

  * the files a patient's run folder holds;
  * the prepared evidence text — exactly what the LLM saw, per recipe step;
  * the evidence-selection metadata — how big each bundle was and what filled it;
  * the LLM exchange — the rendered messages and the raw response, per step;
  * the validation verdicts — per-variable rules and the patient-level
    cross-variable invariants;
  * the recipe itself — read from the run's sealed code bundle when it has one,
    so what is shown is what actually ran, not what the working tree says today;
  * the NO_PHI exhaust manifest, and the shareable zip export.

All of it is JSON + text reads — no encoder, no model. The evidence text and the
LLM messages are chart-derived and stay PHI-side; the app shows them to the local
operator only, the same way the CLI's `inspect` command prints them at a shell.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.cohort_runner import RUN_ID_PATTERN
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    evidence_selection_metadata_dir,
    extract_output_dir,
    no_phi_manifest_path,
    no_phi_root,
    no_phi_run_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    prepared_evidence_text_dir,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES_ROOT = REPO / "var_extraction_recipes"

# Bounds on what one panel renders. A step's rendered prompt carries the whole
# evidence bundle, so an unbounded panel can put a megabyte of chart text into one
# server response; the full artifact stays on disk and the panel says where.
MAX_TEXT_CHARS = 8000
MAX_EXCHANGE_CHARS = 4000


def _valid_run(run_id: str | None) -> bool:
    """The same spoofed-input rule as the review loaders: a run id is joined into
    paths only when it has the canonical shape."""
    return bool(run_id) and bool(RUN_ID_PATTERN.fullmatch(run_id))


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… clipped — {len(text) - limit:,} more characters in the file on disk"


# ── what the run folder holds ────────────────────────────────────────────────

@dataclass
class PatientFile:
    rel_path: str
    size_bytes: int


def list_patient_files(run_id: str, patient_id: str, dr: Path | None = None) -> list[PatientFile]:
    """Every file under this patient's run folder, relative path + size."""
    if not _valid_run(run_id):
        return []
    patient_dir = phi_patient_run_dir(run_id, patient_id, dr)
    if not patient_dir.is_dir():
        return []
    return [
        PatientFile(rel_path=p.relative_to(patient_dir).as_posix(), size_bytes=p.stat().st_size)
        for p in sorted(patient_dir.rglob("*"))
        if p.is_file()
    ]


# ── prepared evidence (what the LLM saw) ─────────────────────────────────────

@dataclass
class EvidenceBundle:
    step_id: str
    text: str


def read_prepared_evidence(run_id: str, patient_id: str, variable: str,
                           dr: Path | None = None) -> list[EvidenceBundle]:
    """The formatted evidence text per recipe step. Evidence is assembled per RECIPE
    STEP, so each bundle sits one level below the variable
    (``<variable>/<step>/formatted_evidence.txt``)."""
    if not _valid_run(run_id):
        return []
    evidence_dir = prepared_evidence_text_dir(run_id, patient_id, dr) / variable
    bundles = []
    for evidence_file in sorted(evidence_dir.glob("*/formatted_evidence.txt")):
        bundles.append(EvidenceBundle(
            step_id=evidence_file.parent.name,
            text=_clip(evidence_file.read_text(encoding="utf-8"), MAX_TEXT_CHARS),
        ))
    return bundles


# ── evidence selection metadata (what step 6 assembled, and its size) ────────

@dataclass
class SelectionSummary:
    step_id: str
    block_count: Any
    evidence_tokens: Any
    max_context_tokens: Any
    tokens_by_doc_type: dict


def read_evidence_selection(run_id: str, patient_id: str, variable: str,
                            dr: Path | None = None) -> list[SelectionSummary]:
    """One selection record per recipe step, beside the bundle it describes."""
    if not _valid_run(run_id):
        return []
    selection_dir = evidence_selection_metadata_dir(run_id, patient_id, dr) / variable
    summaries = []
    for selection_file in sorted(selection_dir.glob("*/evidence_selection.json")):
        try:
            sel = json.loads(selection_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(SelectionSummary(
            step_id=selection_file.parent.name,
            block_count=sel.get("block_count", "?"),
            evidence_tokens=sel.get("total_evidence_tokens", "?"),
            max_context_tokens=sel.get("max_context_tokens", "?"),
            tokens_by_doc_type=sel.get("evidence_tokens_by_doc_type", {}) or {},
        ))
    return summaries


# ── the LLM exchange (exact prompts + responses) ─────────────────────────────

@dataclass
class StepExchange:
    step_id: str
    messages: str   # the rendered messages_sent, pretty-printed and clipped
    response: str   # the raw response, pretty-printed and clipped


def read_llm_exchanges(run_id: str, patient_id: str, variable: str,
                       dr: Path | None = None) -> list[StepExchange]:
    """Each step keeps its own receipt, so there is one prompt/response pair per step
    of the recipe rather than one file for the variable."""
    if not _valid_run(run_id):
        return []
    steps_dir = extract_output_dir(phi_patient_run_dir(run_id, patient_id, dr)) / variable / "steps"
    if not steps_dir.is_dir():
        return []
    exchanges = []
    for receipt in sorted(steps_dir.glob("*/receipt.json")):
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8")).get("payload", {})
        except (json.JSONDecodeError, OSError):
            continue
        exchanges.append(StepExchange(
            step_id=receipt.parent.name,
            messages=_clip(json.dumps(payload.get("messages_sent"), indent=2,
                                      ensure_ascii=False), MAX_EXCHANGE_CHARS),
            response=_clip(json.dumps(payload.get("response_raw"), indent=2,
                                      ensure_ascii=False), MAX_EXCHANGE_CHARS),
        ))
    return exchanges


# ── validation verdicts ──────────────────────────────────────────────────────

@dataclass
class InvariantReport:
    name: str    # which check file this is
    text: str    # its content, pretty-printed


def read_invariants(run_id: str, patient_id: str, variable: str,
                    dr: Path | None = None) -> list[InvariantReport]:
    """Two different checks with similar names: the per-variable rules for this one
    answer, and the cross-variable consistency checks over everything extracted for
    the patient."""
    if not _valid_run(run_id):
        return []
    this_patient = phi_patient_run_dir(run_id, patient_id, dr)
    reports = []
    for invariants_file in (
        extract_output_dir(this_patient) / variable / "invariants.json",
        this_patient / "clinical_invariants.json",
    ):
        if not invariants_file.is_file():
            continue
        try:
            content = json.loads(invariants_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        reports.append(InvariantReport(
            name=invariants_file.name,
            text=_clip(json.dumps(content, indent=2, ensure_ascii=False), MAX_TEXT_CHARS),
        ))
    return reports


# ── the recipe that ran ──────────────────────────────────────────────────────

@dataclass
class RecipeText:
    source: str  # "this run's sealed code bundle" or "the working tree (not this run's sealed copy)"
    path: str
    text: str


def _highest_version_recipe(root: Path, variable: str) -> Path | None:
    candidates = sorted(root.rglob(f"{variable}/v*/{variable}_v*_recipe.yaml"))
    return candidates[-1] if candidates else None


def read_recipe_text(run_id: str | None, variable: str, dr: Path | None = None) -> RecipeText | None:
    """The recipe YAML for one variable. Preferred source is the run's sealed code
    bundle (``<run_root>/code/recipes`` — the tree the seal copied and hashed): that
    copy is what actually executed; the working tree may have moved on since. Falls
    back to the working tree, labeled."""
    if _valid_run(run_id):
        sealed_root = phi_intermediate_run_dir(run_id, dr) / "code"
        sealed_recipes = sealed_root / "recipes"
        sealed = _highest_version_recipe(sealed_recipes, variable) if sealed_recipes.is_dir() else None
        if sealed is not None:
            return RecipeText(
                source="this run's sealed code bundle",
                path=sealed.relative_to(sealed_root).as_posix(),
                text=_clip(sealed.read_text(encoding="utf-8"), MAX_TEXT_CHARS),
            )
    current = _highest_version_recipe(RECIPES_ROOT, variable) if RECIPES_ROOT.is_dir() else None
    if current is None:
        return None
    return RecipeText(
        source="the working tree (not this run's sealed copy)",
        path=current.relative_to(REPO).as_posix(),
        text=_clip(current.read_text(encoding="utf-8"), MAX_TEXT_CHARS),
    )


def list_recipes() -> list[str]:
    """Every recipe on disk as ``<collection>/<variable> (<version>)`` lines — the
    same rglob the extract step resolves by, so anything listed is runnable."""
    if not RECIPES_ROOT.is_dir():
        return []
    lines = []
    for recipe_yaml in sorted(RECIPES_ROOT.rglob("*_recipe.yaml")):
        version_dir = recipe_yaml.parent
        collection = version_dir.parent.parent.relative_to(RECIPES_ROOT).as_posix()
        lines.append(f"{collection}/{version_dir.parent.name} ({version_dir.name})")
    return lines


# ── run rollup + exhaust ─────────────────────────────────────────────────────

def read_run_summary(run_id: str, dr: Path | None = None) -> dict | None:
    """The run's ``summary.json`` (status + per-step counts), written when the run
    closes out. None when the run has not been summarized yet."""
    if not _valid_run(run_id):
        return None
    path = phi_intermediate_run_dir(run_id, dr) / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@dataclass
class ExhaustSummary:
    schema_version: Any
    vocab_version: Any
    surrogate_version: Any
    secret_fingerprint: Any
    record_types: list[tuple[str, Any, Any]] = field(default_factory=list)  # (name, n_rows, n_failed)


def read_exhaust_manifest(run_id: str, dr: Path | None = None) -> ExhaustSummary | None:
    """The NO_PHI exhaust manifest: which de-identified record types this run threw
    off, and how many rows of each. None when the run has no finalized exhaust yet."""
    if not _valid_run(run_id):
        return None
    manifest_file = no_phi_manifest_path(run_id, dr)
    if not manifest_file.is_file():
        return None
    try:
        m = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return ExhaustSummary(
        schema_version=m.get("schema_version"),
        vocab_version=m.get("vocab_version"),
        surrogate_version=m.get("surrogate_version"),
        secret_fingerprint=m.get("secret_fingerprint"),
        record_types=[
            (name, entry.get("n_rows"), entry.get("n_records_failed"))
            for name, entry in (m.get("record_types") or {}).items()
        ],
    )


def export_shareable_zip(run_id: str, dr: Path | None = None) -> Path:
    """Write the run's shareable metadata zip — the same artifact as
    ``junior export-metadata``, produced by the same function: pending exhaust
    shards are compacted first, every file is scanned, and one hit stops the
    export. Raises with the reason when the run has nothing shareable or the
    scan finds something."""
    if not _valid_run(run_id):
        raise ValueError(f"'{run_id}' is not a run id, so there is nothing to export")
    from jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata import (
        export_run_metadata,
    )

    if not no_phi_run_dir(run_id, dr).is_dir():
        raise FileNotFoundError(
            f"run {run_id} has no shareable summary yet — it is written when "
            "`junior extract` closes a run out"
        )
    try:
        from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust

        finalize_exhaust(run_id, dr)
    except Exception:
        # Best-effort, mirroring the CLI: exhaust is telemetry, and a finalize failure
        # must not sink the export of everything else the run can share.
        pass
    output_path = no_phi_root(dr) / f"{run_id}_metadata.zip"
    return export_run_metadata(run_id=run_id, output_path=output_path, dr=dr)


def run_values_csv(run_id: str, *, shape: str = "wide", dr: Path | None = None) -> str:
    """The whole run's extracted values as CSV text — every patient, one table.

    THIS IS PHI: every cell is a value read out of a patient's chart. It is offered
    as a download the operator chooses to take, and it belongs beside the chart
    data if saved — never in the shareable tree."""
    if not _valid_run(run_id):
        return ""
    from jr_pipeline.runtime_infrastructure.values_table import values_as_csv

    return values_as_csv(phi_intermediate_run_dir(run_id, dr), shape=shape)
