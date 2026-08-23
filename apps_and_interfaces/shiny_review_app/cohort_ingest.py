"""Cohort selection + Step 1 (preflight + ingest) for the workbench's Start tab.

Keeps the pipeline wiring out of app.py. Three operations, all Step 1 only — no
encoder, no model, just polars, so they run in-process in a fraction of a second
per example patient:

  * scan_cohort   — list the patient subfolders under an input folder, each with a
                    count of the supported source files discovered inside it;
  * preflight_cohort — the pipeline's dry run over the chosen patients (writes
                    nothing), reporting each as OK or blocked with the reasons;
  * ingest_cohort — run the real Step 1 ingest over the chosen patients into
                    structured/ parquets, one run_id for the whole selection.

Outputs land under the same data root the app reads runs from (JR_DATA_ROOT,
defaulted to <repo>/data below), independent of the launch cwd.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
    preflight_patients,
    run_ingest_one,
)
from jr_pipeline.runtime_infrastructure.cohort_runner import _new_run_id
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    input_patient_records_dir,
)
from jr_pipeline.runtime_infrastructure.source_file_discovery import discover_source_files

REPO = Path(__file__).resolve().parents[2]

# Write ingested parquets under <repo>/data unless the operator points JR_DATA_ROOT
# elsewhere — matches run_results.DATA_ROOT so a fresh ingest lands where the app
# looks. setdefault, so an explicit JR_DATA_ROOT still wins.
os.environ.setdefault("JR_DATA_ROOT", str(REPO / "data"))


def default_input_folder() -> Path:
    """The bundled example cohort (Test_Patient) — works out of the box."""
    return REPO / "examples"


def phi_input_folder() -> Path:
    """The production home for raw patient folders —
    <data_root>/CONTAINS_PHI/patient_data_input. Shown as a hint; may be empty."""
    return input_patient_records_dir()


# ── dataclasses the UI renders ──────────────────────────────────────────────

@dataclass
class PatientFolder:
    patient_id: str
    file_count: int


@dataclass
class ScanResult:
    folder: Path
    patients: list[PatientFolder]
    empty_dirs: list[str]        # subfolders with no supported source files (a hint to look deeper)
    error: str | None = None     # set when the folder can't be scanned / has no usable layout


@dataclass
class PreflightRow:
    patient_id: str
    ok: bool
    problems: list[str] = field(default_factory=list)   # blockers; empty when ok
    advisories: list[str] = field(default_factory=list)  # worth seeing, still ingestable


@dataclass
class IngestRow:
    patient_id: str
    ok: bool
    cached: bool = False
    files_written: int = 0
    total_rows: int = 0
    error: str | None = None


@dataclass
class IngestOutcome:
    run_id: str
    rows: list[IngestRow]
    error: str | None = None     # set when nothing could run (e.g. nothing selected)


def _ingest_cfg(folder: Path, run_id: str) -> dict:
    """Minimal Step-1 config: the input folder as source_root (so patient_id names
    its subfolder) and auto file discovery. project is omitted — source_root wins
    over it in the pipeline's resolver.

    Every corpus-shaping setting the project names rides along, because ingest reads
    them and records the answer per file, permanently. Leaving one out does not mean
    "not set today", it means this button produces a differently-interpreted corpus
    from the same project's `junior ingest`, and the difference is frozen into the
    sidecars before anyone can see it. That happened once with chart_columns_file;
    carrying the whole set rather than the one key that bit us is what stops it
    happening again with the next one — text_column reads 'text' by default and is
    silently wrong for a site whose export names the column anything else.

    source_root, files and run_id are the button's own: the point of it is to ingest
    the folder the operator picked, so those override whatever the project says."""
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
        CORPUS_AFFECTING_CFG_KEYS,
    )

    cfg = {key: value for key, value in _project_settings().items()
           if key in CORPUS_AFFECTING_CFG_KEYS and value is not None}
    cfg.update({"run_id": run_id, "source_root": str(folder), "files": "auto"})
    column_map = _project_column_map()
    if column_map:
        cfg["chart_columns_file"] = column_map
    else:
        cfg.pop("chart_columns_file", None)  # named but unreadable: same as unnamed
    return cfg


def _project_settings() -> dict:
    """The project's resolved settings, or empty when there is no project to read."""
    from jr_pipeline.runtime_infrastructure.config_loading import load_config
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        return load_config(find_config().path) or {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _project_column_map() -> str | None:
    """This project's column map, if its settings name one and the file is there.

    Read from the project the app was opened on rather than passed in, because every
    caller here already resolves the same project and none of them held the config."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
    except (FileNotFoundError, ValueError, OSError):
        return None
    named = _project_settings().get("chart_columns_file")
    if not named:
        return None
    # Resolved the way load_cohort_config resolves it, not against this process's working
    # directory. A Shiny server's cwd is wherever it was launched, so testing
    # `Path(named).is_file()` there meant the two spellings the codebase actually
    # recommends — a bare name beside the settings file, and a repo-relative
    # deployment/<site>/ path — both came back "no map", and this button ingested with
    # generic column guesses while the same project's `junior ingest` used the map.
    expanded = Path(str(named)).expanduser()
    if expanded.is_absolute():
        return str(expanded) if expanded.is_file() else None
    beside_the_config = (choice.path.parent / expanded).resolve()
    if beside_the_config.is_file():
        return str(beside_the_config)
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (  # noqa: E501
        resolve_repo_root,
    )

    anchored = (resolve_repo_root() / expanded).resolve()
    return str(anchored) if anchored.is_file() else None


# ── operations ──────────────────────────────────────────────────────────────

def scan_cohort(folder: Path) -> ScanResult:
    """List the immediate subfolders of ``folder`` that hold at least one supported
    source file — each is one selectable patient. Subfolders with none are returned
    as ``empty_dirs`` (a nested project layout, e.g. a deeper 'ready' folder)."""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        return ScanResult(folder, [], [], error=f"Not a folder: {folder}")

    patients: list[PatientFolder] = []
    empty_dirs: list[str] = []
    for sub in sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: p.name):
        stems = sorted(discover_source_files(sub))
        if stems:
            patients.append(PatientFolder(patient_id=sub.name, file_count=len(stems)))
        else:
            empty_dirs.append(sub.name)

    if not patients:
        # A common mis-point: the operator selected a single patient's own folder.
        # Its files sit directly inside, not in subfolders — say so instead of a
        # bare "found nothing".
        if discover_source_files(folder):
            return ScanResult(
                folder, [], empty_dirs,
                error=(f"{folder.name!r} holds source files directly — point at its "
                       f"PARENT folder so {folder.name!r} becomes one selectable patient."),
            )
        return ScanResult(folder, [], empty_dirs,
                          error=f"No patient subfolders with source files under {folder}.")
    return ScanResult(folder, patients, empty_dirs)


def preflight_cohort(folder: Path, patient_ids: list[str]) -> list[PreflightRow]:
    """The pipeline's dry run over the selected patients. Writes nothing; returns
    one row per patient, OK or blocked with the concrete problems."""
    if not patient_ids:
        return []
    run_id = _new_run_id()  # throwaway — preflight writes nothing, but config wants one
    report = preflight_patients(cfg=_ingest_cfg(folder, run_id), patients=list(patient_ids))
    # Blockers decide OK/blocked; advisories (e.g. a chart metadata column this cohort's
    # files don't carry) are shown but never stop a patient — most tables legitimately
    # carry only some metadata.
    problems: dict[str, list[str]] = {}
    advisories: dict[str, list[str]] = {}
    for issue in report.issues:
        where = f" [{issue.file}]" if issue.file else ""
        line = f"({issue.kind}){where} {issue.detail}"
        bucket = problems if getattr(issue, "blocking", True) else advisories
        bucket.setdefault(issue.patient_id, []).append(line)
    return [
        PreflightRow(pid, pid not in problems, problems.get(pid, []), advisories.get(pid, []))
        for pid in patient_ids
    ]


def ingest_cohort(folder: Path, patient_ids: list[str]) -> IngestOutcome:
    """Run Step 1 ingest over the selected patients under one shared run_id. One
    failing patient is recorded, not raised — the rest still ingest (mirrors the
    cohort runner's per-patient try/except)."""
    if not patient_ids:
        return IngestOutcome(run_id="", rows=[], error="No patients selected.")

    run_id = _new_run_id()
    cfg = _ingest_cfg(folder, run_id)
    rows: list[IngestRow] = []
    for pid in patient_ids:
        try:
            summary = run_ingest_one(cfg=cfg, patient_id=pid, code_lock_hash=None)
        except Exception as exc:
            rows.append(IngestRow(pid, ok=False, error=f"{type(exc).__name__}: {exc}"))
            continue
        files_written = summary["files_written"]
        rows.append(IngestRow(
            pid,
            ok=True,
            cached=summary["cached"],
            files_written=len(files_written),
            total_rows=sum(int(f["rows"]) for f in files_written),
        ))
    return IngestOutcome(run_id=run_id, rows=rows)
