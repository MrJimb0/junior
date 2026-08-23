"""Shared I/O helpers: atomic writes + the canonical data-tree layout.

The data root (``JR_DATA_ROOT``, default ``./data``) can point at any
filesystem — local disk, a network mount, or a SLURM scratch partition.
Under it, two top-level directories enforce the PHI classification:

    <data_root>/
    ├── CONTAINS_PHI/                          HIPAA-scoped, restricted access
    │   ├── patient_data_input/        raw CSVs / EHR exports (one folder per patient)
    │   └── pipeline_run_receipts/<run>/  ALL per-patient artifacts + final extractions
    │       └── patients/<pid>/               (structured tables, chunks, embeddings,
    │                                          index, evidence, extract/<variable>,
    │                                          sidecar metadata)
    │
    └── NO_PHI__shareable/                     ONLY run-level metadata
        ├── sealed_code/                       code snapshot + hashes
        └── run_config/                        settings used for this run

Classification rule: EVERYTHING per-patient is PHI. NO_PHI contains
only run-level metadata — code snapshots, configs, aggregate numbers.
Nothing patient-derived, nothing patient-identifiable, ever.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any
from uuid import uuid4

SENSITIVE_LABEL = os.environ.get("JR_SENSITIVE_DATA_LABEL", "CONTAINS_PHI")
NON_SENSITIVE_LABEL = os.environ.get("JR_NON_SENSITIVE_LABEL", "NO_PHI__shareable")

# A patient id becomes a single filesystem path segment AND part of a ':'-delimited
# chunk id; a run id becomes a path segment AND part of every answer table's filename.
# Anything outside this charset (a '/', '\\', ':', whitespace, NUL, or an
# absolute/traversal shape) could escape the per-patient dir, collide with another
# patient, or corrupt ':TABLE' chunk resolution — so it is rejected at the door.
#
# The end anchor is \Z, not $: Python's $ also matches immediately before a trailing
# newline, so '..\n' satisfies the pattern and slips past the '..' check below as well,
# since the string is not equal to '..'. That is a real directory name with a newline
# in it on POSIX, and a traversal on Windows and anywhere the id is later stripped.
_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+\Z")


def validate_patient_id(patient_id: str) -> str:
    """Reject any patient id that is not a safe single path segment, BEFORE it is joined
    into a filesystem path. Patient isolation is the crown-jewel property; this is the
    one chokepoint every per-patient path helper routes through. Returns the id so
    callers can wrap inline."""
    if not isinstance(patient_id, str) or not patient_id:
        raise ValueError("patient id must be a non-empty string")
    if patient_id in {".", ".."}:
        raise ValueError(f"patient id {patient_id!r} is a path-traversal segment")
    if not _SAFE_PATH_SEGMENT_RE.match(patient_id):
        raise ValueError(
            f"patient id {patient_id!r} contains characters outside [A-Za-z0-9._-] "
            "(no '/', ':', whitespace, or path separators); rename the patient folder."
        )
    return patient_id


def validate_run_id(run_id: str) -> str:
    """Reject any run id that is not a safe single path segment, BEFORE it is joined
    into a filesystem path or a downloadable filename.

    The same chokepoint ``validate_patient_id`` is for the other id that reaches disk.
    A run id names a directory under pipeline_run_receipts AND is stamped into every
    answer table's filename, so a '/' or a traversal segment escapes the receipts root
    or writes the answers somewhere nobody looks for them. Run ids are generated, but
    they are also passed in by hand (``--run-id``) and derived (compare_llm_models
    appends ``__compare__<model>``), so the shape is checked rather than assumed.
    Returns the id so callers can wrap inline."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run id must be a non-empty string")
    if run_id in {".", ".."}:
        raise ValueError(f"run id {run_id!r} is a path-traversal segment")
    if not _SAFE_PATH_SEGMENT_RE.match(run_id):
        raise ValueError(
            f"run id {run_id!r} contains characters outside [A-Za-z0-9._-] "
            "(no '/', ':', whitespace, or path separators)."
        )
    return run_id


def data_root() -> Path:
    """Resolve the data root from ``JR_DATA_ROOT`` env var (default ``./data``)."""
    return Path(os.environ.get("JR_DATA_ROOT", "./data"))


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> Path:
    """Write JSON via a per-process-unique tmp file + atomic ``os.replace``, serialized
    with a file lock for SLURM safety.

    The lock-unlink race: py-filelock removes the lock file on release, and across
    processes that has a TOCTOU window (writer B keeps the old inode while a fresh
    caller C re-creates the lock file as a new inode and acquires it) that can admit two
    writers into the critical section at once. The defense is the UNIQUE tmp name:
    each writer renders into its own ``.<name>.<pid>.<uuid>.tmp`` and the final
    ``os.replace`` is atomic, so even a double-admission cannot leave a torn/partial
    file — a reader always sees a complete prior-or-new copy. (Writes to one path are
    idempotent re-renders, so last-writer-wins is correct.)"""
    from filelock import FileLock

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    lock = FileLock(str(lock_path), timeout=30)
    with lock:
        tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=indent, sort_keys=True)
            os.replace(tmp, path)  # atomic on POSIX / same-filesystem Windows
        finally:
            tmp.unlink(missing_ok=True)  # no-op after a successful replace; clears a failed write
    return path


def phi_root(dr: Path | None = None) -> Path:
    return (dr or data_root()) / SENSITIVE_LABEL


def input_patient_records_dir(dr: Path | None = None) -> Path:
    """Default home for raw patient folders (one subfolder per patient) when config omits ``source_root``."""
    return phi_root(dr) / "patient_data_input"


def pipeline_run_receipts_root(dr: Path | None = None) -> Path:
    """The directory holding every run's receipts — one subdirectory per run id.
    The parent of ``phi_intermediate_run_dir``; centralized here so the only place
    that knows the ``pipeline_run_receipts`` literal is this module (which is what
    lets a caller enumerate prior runs, e.g. ``also_include_tables`` resolving the
    latest run that produced a builder table)."""
    return phi_root(dr) / "pipeline_run_receipts"


def phi_intermediate_run_dir(run_id: str, dr: Path | None = None) -> Path:
    # Through the chokepoint, exactly as the patient id goes through its own one below.
    # A run id arrives from --run-id, from JR_RUN_ID (which the SLURM scripts export),
    # and from a config file, so an absolute path or a `..` segment here relocates the
    # whole run: every patient's parquet, every receipt, outside CONTAINS_PHI entirely.
    # check_phi_containment only walks what is inside the tree, so it cannot see that.
    return pipeline_run_receipts_root(dr) / validate_run_id(run_id)


def phi_patient_run_dir(run_id: str, patient_id: str, dr: Path | None = None) -> Path:
    return phi_intermediate_run_dir(run_id, dr) / "patients" / validate_patient_id(patient_id)


def structured_dir(patient_root: Path) -> Path:
    """Per-patient structured parquets (``<patient_root>/structured/<stem>.parquet``).
    Keyed off the patient run dir (a Path), not (run_id, patient_id): the
    PatientChunkStore read path and the direct_parquet retriever hold only a
    patient_root, and centralizing the ``structured`` literal here is what lets the
    PHI-seam audit trust the path. ``phi_patient_run_dir(run_id, pid)`` returns the root."""
    return patient_root / "structured"


def derived_tables_dir(patient_root: Path) -> Path:
    """Per-patient normalized **builder-intermediate** tables
    (``<patient_root>/derived_tables/<stem>.parquet``). Kept OUT of ``structured/``
    on purpose: embedding globs ``structured/*.parquet`` and ingest reconciles that
    folder against the patient's source files, so a builder table placed there would
    be embedded and could be quarantined as an "orphan" (it has no source CSV in the
    patient input dir). A retrieve_and_prompt step reads these directly to merge rows
    into the evidence bundle; nothing else in the pipeline touches them. Patient-root
    keyed, like ``structured_dir``."""
    return patient_root / "derived_tables"


def builder_intermediates_dir(run_id: str, patient_id: str, dr: Path | None = None) -> Path:
    """Where an upstream *builder* drops its per-patient intermediate CSVs (e.g. a
    pathology CAP table) for a run, as ``<dir>/<name>.csv``. Keyed by (run_id,
    patient_id) so a recipe can reuse a PRIOR run's intermediates by pinning that
    run's id. PHI (the CSVs hold patient data); routes through the patient-id
    chokepoint via ``phi_patient_run_dir``."""
    return phi_patient_run_dir(run_id, patient_id, dr) / "builder_intermediates"


def quarantined_structured_dir(run_id: str, patient_id: str, dr: Path | None = None) -> Path:
    """Holding area for structured parquets whose source file was removed. Kept
    out of ``structured/`` so embedding (which globs ``structured/*.parquet``)
    stops reading them; never deleted, so a mistaken source removal stays
    recoverable. PHI (the parquets hold patient data)."""
    return phi_patient_run_dir(run_id, patient_id, dr) / "quarantined_structured"


def retrieval_output_dir(patient_root: Path) -> Path:
    """Per-patient retrieval envelopes (``<patient_root>/retrieve/<query_id>.json``). Patient-root keyed."""
    return patient_root / "retrieve"


def extract_output_dir(patient_root: Path) -> Path:
    """Per-patient extraction outputs (``<patient_root>/extract/<recipe>/...``). Patient-root keyed."""
    return patient_root / "extract"


def eval_metrics_dir(run_root: Path) -> Path:
    """Retrieval-quality eval artifacts (``<run_root>/data/eval/{candidate,bundle}_metrics.json``).
    Run-root keyed so the writer (run.py) and the cross-run reader (compare_runs.py) agree on
    one path instead of hardcoding the literal."""
    return Path(run_root) / "data" / "eval"


def evidence_selection_metadata_dir(run_id: str, patient_id: str, dr: Path | None = None) -> Path:
    return phi_patient_run_dir(run_id, patient_id, dr) / "evidence_selection_metadata"


def prepared_evidence_text_dir(run_id: str, patient_id: str, dr: Path | None = None) -> Path:
    return phi_patient_run_dir(run_id, patient_id, dr) / "prepared_evidence_text"


def no_phi_root(dr: Path | None = None) -> Path:
    return (dr or data_root()) / NON_SENSITIVE_LABEL


def no_phi_run_dir(run_id: str, dr: Path | None = None) -> Path:
    # Through the chokepoint, for the reason given on phi_intermediate_run_dir. The
    # NO_PHI side matters just as much: a run id that escapes here writes the exhaust
    # and the manifest somewhere the egress scan never looks before an export.
    return no_phi_root(dr) / validate_run_id(run_id)


def no_phi_exhaust_dir(run_id: str, dr: Path | None = None) -> Path:
    """The NO_PHI exhaust subtree: finalized ``<record_type>.parquet`` tables and the
    exhaust manifest live here. Run-scoped only (a NO_PHI helper must not take a
    patient_id -- the layout reflection guardrail enforces it)."""
    return no_phi_run_dir(run_id, dr) / "exhaust"


def no_phi_exhaust_shards_dir(run_id: str, dr: Path | None = None) -> Path:
    """Process-unique JSONL shards that ``emit()`` appends to, before finalize compacts
    them into ``<record_type>.parquet``. Run-scoped."""
    return no_phi_exhaust_dir(run_id, dr) / "_shards"


def no_phi_manifest_path(run_id: str, dr: Path | None = None) -> Path:
    """The shareable NO_PHI exhaust manifest (record-type inventory, file hashes, schema
    versions, secret fingerprint). Carries NO patient roster -- the roster lives PHI-side
    in ``run_roster.json``. Run-scoped."""
    return no_phi_run_dir(run_id, dr) / "manifest.json"


def sealed_code_dir(run_id: str, dr: Path | None = None) -> Path:
    return no_phi_run_dir(run_id, dr) / "sealed_code"


def run_config_dir(run_id: str, dr: Path | None = None) -> Path:
    return no_phi_run_dir(run_id, dr) / "run_config"


RUN_ARTIFACT_GUIDE_NAME = "WHAT_IS_IN_HERE.txt"

# Every artifact a run can leave behind, in the order the pipeline produced it, said in
# words a clinician-researcher can act on. A run folder is flat on purpose — every cached
# stage is a presence test at these exact paths, so re-nesting the tree would silently
# re-run finished work and hide finished runs from the review app — which means the
# folder has to explain itself instead. Each entry is
# (stage_label, path_pattern, plain_english); ``<...>`` stands for a name you will see on
# disk. Paths are relative to the run folder, except those under the shareable label,
# which sit in the sibling tree beside CONTAINS_PHI/.
RUN_ARTIFACT_GUIDE: list[tuple[str, str, str]] = [
    (
        "1 ingest",
        "patients/<patient>/source_snapshot.json",
        "The chart files ingest read for this patient, each with a fingerprint, so you can "
        "tell later exactly which export this patient was built from.",
    ),
    (
        "1 ingest",
        "patients/<patient>/structured/<table>.parquet",
        "One kind of chart data — notes, labs, diagnoses, medications — as ingest turned it "
        "into a table with one row per record.",
    ),
    (
        "1 ingest",
        "patients/<patient>/structured/<table>.parquet.meta.json",
        "The note ingest left beside each table saying which source file it came from and "
        "how many rows it holds.",
    ),
    (
        "1 ingest",
        "patients/<patient>/quarantined_structured/<table>.parquet",
        "A table whose source file has since disappeared from the input folder; ingest moves "
        "it here so nothing reads it, and never deletes it.",
    ),
    (
        "1 ingest",
        "patients/<patient>/quarantined_structured/<table>.parquet.meta.json",
        "The note that travelled with the quarantined table above, saying which source file "
        "it had come from.",
    ),
    (
        "2 embed",
        "patients/<patient>/chunk_index.parquet",
        "Every passage of this patient's chart, one row each, as embed cut them up — the text "
        "itself and where in the chart it came from.",
    ),
    (
        "2 embed",
        "patients/<patient>/chunk_index.parquet.meta.json",
        "The settings embed used to cut those passages and the model that read them, plus how "
        "many passages were reused from an earlier run.",
    ),
    (
        "2 embed",
        "patients/<patient>/embeddings.npy",
        "One row of numbers per passage in the same order as chunk_index.parquet; this is what "
        "embed produced and what a search compares against.",
    ),
    (
        "3 index",
        "patients/<patient>/hnsw.bin",
        "The lookup structure index built so a question can find its closest passages without "
        "reading all of them.",
    ),
    (
        "3 index",
        "patients/<patient>/hnsw.bin.meta.json",
        "The settings that lookup structure was built with, so a later stage can tell whether "
        "it still matches the passages.",
    ),
    (
        "4 extract",
        "patients/<patient>/extract/<variable>/result.json",
        "The answer for one variable, the evidence behind it, and whether it passed its checks "
        "— the file the whole run exists to produce.",
    ),
    (
        "4 extract",
        "patients/<patient>/extract/<variable>/invariants.json",
        "The rules extract checked that one answer against, and which of them it passed.",
    ),
    (
        "4 extract",
        "patients/<patient>/extract/<variable>/transcript.json",
        "Extract's account of how that answer was reached, step by step, including the steps "
        "that found nothing.",
    ),
    (
        "4 extract",
        "patients/<patient>/extract/<variable>/steps/<step>/receipt.json",
        "One step's own record: what extract asked, what came back, how long it took and how "
        "many tokens it cost.",
    ),
    (
        "4 extract",
        "patients/<patient>/evidence_selection_metadata/<variable>/<step>/evidence_selection.json",
        "How much evidence extract chose to put in front of the model for that step, and how "
        "close it came to the model's reading limit.",
    ),
    (
        "4 extract",
        "patients/<patient>/prepared_evidence_text/<variable>/<step>/formatted_evidence.txt",
        "The chart text exactly as the model saw it during extract — the first file to open "
        "when an answer looks wrong.",
    ),
    (
        "4 extract",
        "patients/<patient>/prepared_evidence_text/<variable>/<step>/evidence_blocks.json",
        "The same passages one at a time, each with the id an answer cites, so a value can be "
        "traced back to the sentence it came from.",
    ),
    (
        "4 extract",
        "../answers/<run id>/<variable>_results.xlsx",
        "The answers, one file per variable and one row per patient — or one row per "
        "record for a recipe that reads a table. Every field the recipe produced, "
        "including each value's evidence and the passage it came from, so a row can be "
        "checked without opening anything else. This is what a run is for. "
        "NOT in this folder: they sit one level out under answers/, a dated folder per "
        "run, so asking the same recipes of one cohort twice gives two sets you can "
        "open side by side. Workbooks rather than CSV, because a cell carries a whole "
        "cited passage and multi-line clinical text in a CSV is a quoted field spanning "
        "forty lines; read one with pandas.read_excel or R readxl.",
    ),
    (
        "4 extract",
        "patients/<patient>/clinical_invariants.json",
        "The consistency checks extract ran across all of this patient's answers at once — for "
        "example, a death date that cannot come before a diagnosis.",
    ),
    (
        "4 extract",
        "patients/<patient>/derived_tables/<name>.parquet",
        "A table a recipe assembled for its own use during extract, kept out of structured/ so "
        "it is never mistaken for chart data.",
    ),
    (
        "4 extract",
        "patients/<patient>/derived_tables/<name>.parquet.builder_meta.json",
        "How the table beside it was assembled — which columns were read and what was done to "
        "them — so a value taken from it can be traced back.",
    ),
    (
        "4 extract",
        "patients/<patient>/extract/<variable>/quarantine.jsonl",
        "Model answers that could not be used, one per line, kept so a recipe that keeps "
        "failing can be diagnosed from what the model actually said.",
    ),
    (
        "4 extract",
        "patients/<patient>/builder_intermediates/<name>.csv",
        "Tables an upstream tool dropped into this run for a recipe to read during extract; "
        "Junior reads these, it does not write them.",
    ),
    (
        "4 extract",
        "patients/<patient>/retrieve/<question>.json",
        "The passages a search returned for one question, written when you run `retrieve` to "
        "probe the same search extract uses internally.",
    ),
    (
        "run",
        RUN_ARTIFACT_GUIDE_NAME,
        "This file. Written when the run's folders are laid out, and rewritten by a later "
        "command only if what a run can contain has changed since.",
    ),
    (
        "run",
        "manifest.json",
        "What this run is: when it opened, which settings and which code it is pinned to, and "
        "how many patients it was asked to cover.",
    ),
    (
        "run",
        "run_roster.json",
        "The patient ids this run was explicitly given, if any — the list the shareable-export "
        "scan searches the shareable tree for. A run that found its patients by reading the "
        "input folder leaves this empty; summary.json is where the patients it actually "
        "processed are listed.",
    ),
    (
        "run",
        "summary.json",
        "The tally written when the run closes: how many patients, how many passages, what "
        "failed, and how long it took.",
    ),
    (
        "run",
        "state.jsonl",
        "One line per patient per stage saying when it started and how it ended — the answer "
        "to 'did this patient's embed actually run?'",
    ),
    (
        "run",
        "state_fragments/state.<process>.jsonl",
        "The pieces of that log each process wrote for itself; they are merged into state.jsonl "
        "when the run closes.",
    ),
    (
        "run",
        "run_log.jsonl",
        "Every event the run emitted, in order — the detail behind each line the terminal "
        "summarised.",
    ),
    (
        "run",
        "code/",
        "A copy of the code, recipes and settings this run is pinned to, taken before any "
        "patient was touched, so the run can be repeated exactly.",
    ),
    (
        "run",
        "llm_cache.db",
        "The model's answers, kept so a re-run does not ask the same question twice; it holds "
        "chart text, which is why it lives on this side of the boundary.",
    ),
    (
        "run",
        "llm_cache.db-wal",
        "The same cached chart text, mid-write. SQLite keeps this beside the database while it "
        "is open, and it carries the same content, so it belongs to the same side of the "
        "boundary.",
    ),
    (
        "run",
        "llm_cache.db-shm",
        "SQLite's coordination file for the database beside it, present only while the cache "
        "is open.",
    ),
    (
        "run",
        "cohort_result.json",
        "What a whole-cohort run made of each stage for each patient — including the stages it "
        "skipped, which the per-stage logs cannot show.",
    ),
    (
        "2 embed",
        "patients/<patient>/validate_embed.json",
        "The check embed ran over its own output for this patient: how many passages there "
        "were, and whether their vectors came out the shape they should.",
    ),
    (
        "run",
        "exhaust_hmac_secret",
        "The key that turns real patient ids into the stand-ins used in the shareable tree; it "
        "never leaves this folder.",
    ),
    (
        "run",
        "data/eval/<metric set>.json",
        "How well retrieval scored against an answer key (candidate_metrics.json, "
        "bundle_metrics.json), written by `eval` rather than by a stage.",
    ),
    (
        "run",
        f"{NON_SENSITIVE_LABEL}/<run>/manifest.json",
        "The shareable inventory of what this run threw off: which tables, how many rows each, "
        "and their fingerprints — and no list of patients.",
    ),
    (
        "run",
        f"{NON_SENSITIVE_LABEL}/<run>/exhaust/<record type>.parquet",
        "One shareable table per kind of record — whether each extraction worked and what it "
        "cost, which passages were chosen, which checks passed — patients as stand-ins only.",
    ),
    (
        "run",
        f"{NON_SENSITIVE_LABEL}/<run>/exhaust/_shards/<record type>/<process>.jsonl",
        "The rows each process appended as it worked, before they were combined into the "
        "shareable table above.",
    ),
    (
        "run",
        f"{NON_SENSITIVE_LABEL}/<run>/run_config/",
        "The settings this run was given, in shareable form — for instance which model it was "
        "allowed to call.",
    ),
    (
        "run",
        f"{NON_SENSITIVE_LABEL}/<run>/sealed_code/",
        "The copy of code/ that `export-metadata` makes, so the recipes and settings can be "
        "shared or archived with no chart data attached.",
    ),
]


def render_run_artifact_guide() -> str:
    """``RUN_ARTIFACT_GUIDE`` as plain text, grouped by the stage that wrote each thing,
    ready to print or write into the run folder."""
    lines = ["WHAT IS IN HERE", "===============", ""]
    for paragraph in (
        "This folder is one run: everything Junior read, everything it worked out, and the "
        "record of how. Below is every file a run can leave here, in the order the stages "
        "produced it. Paths are relative to this folder; a name in <angle brackets> stands "
        "for a folder or file name you will see on disk.",

        f"Everything here is patient-identifiable and belongs inside {SENSITIVE_LABEL}/. The "
        f"exception is the {NON_SENSITIVE_LABEL}/ lines at the end: those sit in the "
        f"shareable tree beside {SENSITIVE_LABEL}/, hold no chart text and no real patient "
        "ids, and are the only part of a run meant to be copied elsewhere.",

        "Not every line appears in every run — a stage you have not run yet writes nothing.",
    ):
        lines += [textwrap.fill(paragraph, width=88), ""]
    for stage_label in dict.fromkeys(stage for stage, _, _ in RUN_ARTIFACT_GUIDE):
        lines += ["", stage_label, "-" * len(stage_label)]
        for stage, path_pattern, plain_english in RUN_ARTIFACT_GUIDE:
            if stage == stage_label:
                # Paths keep their own line however long they are (a wrapped path cannot be
                # copied); the sentence wraps to stay readable in a terminal.
                lines.append(f"  {path_pattern}")
                lines += textwrap.wrap(plain_english, width=88,
                                       initial_indent="      ", subsequent_indent="      ")
    return "\n".join(lines) + "\n"


def write_run_artifact_guide(run_root: Path) -> Path:
    """Write the guide into the run folder and return its path.

    Rendered fresh each time rather than written once at run creation: a run made by an
    older Junior gains the current guide the next time any command touches it, so the
    explanation on disk never describes a layout that has moved on. The unique temp name
    plus ``os.replace`` is the same defence ``atomic_write_json`` uses — a SLURM fan-out
    has many processes ensuring the same layout at once, and none of them may leave a
    half-written file for the next reader."""
    path = Path(run_root) / RUN_ARTIFACT_GUIDE_NAME
    text = render_run_artifact_guide()
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace; clears a failed write
    return path


PROJECT_RUN_INDEX_NAME = "RUNS.txt"

# Where a project puts its answers: one dated folder per run, all of them siblings.
#
# They used to sit inside the run receipt, in a folder called run_output_all — which
# meant every run had a folder of that same name, buried under its own run id, beside
# the response cache and the hmac secret. Asking the same recipes of one embedded cohort
# twice is the ordinary way to work here (`extract --new-run` inherits the corpus for
# exactly that), and comparing the two answers meant two deep paths whose last two
# segments were identical. As siblings under one folder they sort by date and open side
# by side.
#
# One file per recipe, because each recipe has its own fields and joining them into a
# single wide table produced sparse columns nobody asked for.
ANSWERS_DIR_NAME = "answers"


def answers_root(dr: Path | None = None) -> Path:
    """Every run's answers, one dated folder each. PHI: these are chart values."""
    return phi_root(dr) / ANSWERS_DIR_NAME


def run_output_dir(run_root: Path) -> Path:
    """The folder holding one run's per-recipe answer tables.

    Derived from the run's own path rather than taking a data root, because every caller
    already holds the run. A run folder that is NOT inside a project's receipts — one
    copied off somewhere on its own — has no project to put answers beside, so they go
    in the run itself and it stays readable."""
    run_root = Path(run_root)
    if run_root.parent.name == "pipeline_run_receipts":
        # Back out to the data root and down again, so the CONTAINS_PHI label is spelled
        # in one place and an overridden JR_SENSITIVE_DATA_LABEL still lands right.
        return answers_root(run_root.parents[2]) / run_root.name
    return run_root / ANSWERS_DIR_NAME


def recipe_table_name(run_id: str, variable: str) -> str:
    """One recipe's answers across every patient, stamped with the run that made them.

    Same shape the pre-rewrite pipeline used for the 100-patient experiment —
    date_of_birth_results_20260311_120527.xlsx — because these files leave the folder,
    and a name that only means something while you are standing in it is no name.

    The WHOLE run id, not the timestamp part. Trimming to the first fifteen characters
    dropped the hex suffix, which is the only thing distinguishing two runs started in
    the same second — and compare_llm_models starts several deliberately, naming them
    <run id>__compare__<model>. Comparing three endpoints wrote four different sets of
    answers to one filename, each overwriting the last. The run now names the FOLDER,
    which keeps them apart without the id being in the name twice; the run id is on
    every row besides, so a file emailed on its own still says which run made it."""
    validate_run_id(run_id)
    return f"{variable}_results{ANSWER_TABLE_SUFFIX}"


# A workbook, as the pre-rewrite pipeline wrote. Why not CSV is in values_table's
# module docstring, which owns the format.
ANSWER_TABLE_SUFFIX = ".xlsx"

# The per-run facts, once, beside the tables whose rows they describe. Not a recipe's
# answers, so it is not one of the tables the run index counts or the close-out prunes.
RUN_METADATA_TABLE_NAME = f"run_metadata{ANSWER_TABLE_SUFFIX}"

def recipe_tables_in(run_root: Path) -> list[Path]:
    """Every per-recipe answer table a run has written."""
    output = run_output_dir(run_root)
    if not output.is_dir():
        return []
    return sorted(path for path in output.glob(f"*{ANSWER_TABLE_SUFFIX}")
                  if path.name != RUN_METADATA_TABLE_NAME)

# What a person opens a run folder to find, in the order they want it: the answers first,
# then the working material behind them. Everything else in a run -- the response cache,
# the hmac secret, the state fragments, two jsonl logs -- is machinery, and listing it
# here would rebuild the haystack this index exists to remove.
_WORTH_POINTING_AT = (
    ("patients", "per-patient evidence and working files"),
    ("summary.json", "what finished, what failed"),
)


def _readable_run_date(run_id: str) -> str:
    """A generated run id is a timestamp, and reads as one to nobody.

    20260820_105033_b74d is 20 Aug 2026 at 10:50, which is the thing somebody scanning a
    folder for "the run I did this morning" is actually looking for."""
    from datetime import datetime

    try:
        return datetime.strptime(run_id[:15], "%Y%m%d_%H%M%S").strftime("%d %b %Y  %H:%M")
    except ValueError:
        return " " * 18  # a hand-named run keeps its name and gets no invented date


def render_project_run_index(receipts_root: Path) -> str:
    """A dated list of this project's runs and where the answers are in each.

    A project accumulates runs named by timestamp, in a folder called
    pipeline_run_receipts, each holding fifteen entries of which two are the point. Every
    piece of that is defensible and the whole is unreadable, so this sits above it and
    says which run is which and what to open."""
    receipts_root = Path(receipts_root)
    runs = sorted(
        (d for d in receipts_root.iterdir() if d.is_dir() and not d.name.startswith(".")),
        key=lambda d: d.name,
        reverse=True,
    ) if receipts_root.is_dir() else []

    lines = [
        "RUNS IN THIS PROJECT",
        "====================",
        "",
        "Newest first. Each folder below is one run of the pipeline.",
        "Rewritten when a run starts and again when it finishes, so it never goes stale.",
        "",
    ]
    if not runs:
        lines.append("  No runs yet. `junior run` makes one.")
        return "\n".join(lines) + "\n"

    for run in runs:
        lines.append(f"  {_readable_run_date(run.name)}   {run.name}")
        # Answers first, then the working material behind them. A recipe that answers
        # with a list per patient gets its own table, and there may be several, so they
        # are found rather than named — but they belong beside values.csv, not after
        # the summary.
        tables = recipe_tables_in(run)
        found: list[tuple[str, str]] = []
        if tables:
            # The answers are one level out, beside every other run's, so the index
            # spells the way there rather than a name that is not in this folder.
            found.append((
                f"../{ANSWERS_DIR_NAME}/{run.name}/",
                f"the answers — {len(tables)} recipe table(s), every patient in each",
            ))
        found += [(name, what_it_is) for name, what_it_is in _WORTH_POINTING_AT
                  if (run / name).exists()]
        # Width from the longest name in THIS run rather than a fixed column: the
        # extraction tables carry the run id so they still say what they are once they
        # have been downloaded somewhere else, which makes them far longer than
        # "patients" and overran any constant.
        width = max((len(name) for name, _ in found), default=0) + 2
        for name, what_it_is in found:
            lines.append(f"      {name:<{width}}{what_it_is}")
        # Said whenever there are no answers, not only when the folder is bare. A run
        # that ingested and embedded has plenty in it and still cannot answer anything,
        # and listing its working files without saying so invites opening them to
        # look for values that were never produced.
        if not tables:
            lines.append("      (no values yet — this run has not been through extract)")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_project_run_index(receipts_root: Path) -> Path:
    """Write RUNS.txt beside the runs. Same atomic replace as the per-run guide."""
    receipts_root = Path(receipts_root)
    # Never creates the folder it indexes. An index of runs where there are no runs is
    # nothing, and conjuring the directory as a side effect makes an empty project look
    # started — which is exactly what `new-project` tests for when it refuses to
    # overwrite one that already exists.
    if not receipts_root.is_dir():
        return receipts_root / PROJECT_RUN_INDEX_NAME
    path = receipts_root / PROJECT_RUN_INDEX_NAME
    text = render_project_run_index(receipts_root)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return path
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def ensure_layout(run_id: str, dr: Path | None = None) -> None:
    """Create the run's directory tree and the guide that explains it. Idempotent.

    Only pre-creates the dirs the pipeline actually populates. ``sealed_code`` has
    no writer that runs every time, so it is not pre-created here;
    ``atomic_write_json`` mkdirs the parent on demand, so when a writer does run the
    dir appears then — without empty placeholders.

    Every stage calls this, so the guide is born with the run and refreshed on any run
    a later command touches.
    """
    run_root = phi_intermediate_run_dir(run_id, dr)
    # Whether this call is what brought the run into being. The project index lists one
    # line per run, so it goes stale exactly twice: when a run appears, and when a run
    # gains its answers (write_summary re-renders it then). It does NOT go stale because
    # a stage touched a run that already existed — and ensure_layout is called per
    # patient per stage, and per QUERY in retrieve, so rebuilding it here cost a full
    # scan of every run in the project thousands of times per cohort: measured at 10ms
    # against 200 runs, which is half a minute of pure metadata I/O on a 500-patient
    # cohort and far more during a retrieval eval.
    is_a_new_run = not run_root.is_dir()
    run_root.mkdir(parents=True, exist_ok=True)
    run_config_dir(run_id, dr).mkdir(parents=True, exist_ok=True)
    write_run_artifact_guide(run_root)
    if is_a_new_run:
        # One level up: which run is which, by date, and what to open in each.
        write_project_run_index(run_root.parent)


def clinician_feedback_dir(run_id: str, dr: Path | None = None) -> Path:
    """Per-run expert corrections from Shiny review. PHI (references patients)."""
    return phi_root(dr) / "expert_label_corrections" / run_id

