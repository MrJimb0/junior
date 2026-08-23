"""Step 1: the front door where a patient's raw chart files enter the pipeline.

For each patient, ingest reads every supported file (csv / tsv / xlsx / json /
jsonl / parquet) into a table, keeping every value as plain text. It drops
fully-empty trailing rows (some EHR exports leave these behind), rewrites
date-shaped columns into one standard format (date_canonicalization.py),
and saves a clean copy of each as ``structured/<stem>.parquet`` in the patient's
run folder ("stem" = the file name without its extension, e.g. ``notes`` for
``notes.csv``). Alongside each
parquet it writes a small companion metadata file (a "sidecar") that records
where the data came from and which exact version of the pipeline code produced
it, so any result can be traced back later.

Source files are never modified — we only read them.

``files: auto`` (or leaving out the ``files`` key) ingests every supported file
in the patient's folder. An explicit list restricts ingest to those file stems;
any supported file the list doesn't name is logged as skipped, never silently
ignored. Step 1 keeps each source file whole; you choose which text columns to
search over later, in the Step 2 embed config.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.date_canonicalization import (
    _canonicalize_dates,
    _sample_date_offenders,
    _two_digit_year_offenders,
)
from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.source_format_readers import _read_source
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.per_patient_source_snapshot import (
    snapshot_matches,
    write_source_snapshot,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    ambiguous_columns,
    columns_from_config,
    data_columns_for,
    find_column,
    text_columns_for,
)
from jr_pipeline.runtime_infrastructure.config_loading import validate_ingest_config
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    ensure_layout,
    input_patient_records_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    quarantined_structured_dir,
    validate_patient_id,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.source_file_discovery import (
    SUPPORTED_SOURCE_EXTENSIONS,
    discover_source_files,
)

_log = get_logger("ingest")


# ---------------------------------------------------------------------------
# Folder discovery + file-entry resolution
# ---------------------------------------------------------------------------

def _resolve_source_root(cfg: dict) -> Path:
    """Work out the folder that holds patients' source files. An explicit
    ``cfg['source_root']`` wins; otherwise it defaults to
    ``<data_root>/CONTAINS_PHI/patient_data_input/[project/]``. Just path
    computation — reads and writes nothing on disk."""
    source_root_cfg = cfg.get("source_root")
    project = cfg.get("project")
    if source_root_cfg:
        return Path(source_root_cfg).expanduser().resolve()
    base = input_patient_records_dir()
    return (base / project if project else base).resolve()


def _normalize_file_entry(entry: dict) -> dict:
    """Put a single files-config entry into a consistent ``{stem, optional}``
    shape so callers don't have to fill in defaults at every use site."""
    return {
        "stem": entry["stem"],
        "optional": bool(entry.get("optional", False)),
    }


def resolve_file_entries(files_cfg, discovered: dict[str, Path]) -> list[dict]:
    """Decide which file stems to ingest. ``auto`` (or None) -> one entry per
    file actually found on disk; an explicit list -> one normalized
    ``{stem, optional}`` entry each. Pure: touches no disk, logs nothing, raises
    nothing — shared by ingest and preflight (the dry-run check), which differ
    only in how they handle an empty ``auto`` result (ingest raises an error,
    preflight records it as a problem to report)."""
    if files_cfg is None or files_cfg == "auto":
        return [{"stem": s, "optional": False} for s in sorted(discovered)]
    return [_normalize_file_entry(entry) for entry in files_cfg]


def _resolve_file_entries(cfg: dict, patient_src: Path, discovered: dict[str, Path], log) -> list[dict]:
    """The ingest version of choosing which files to read: calls
    ``resolve_file_entries`` and adds operator logging plus the error when
    ``auto`` finds nothing. ``discovered`` is the shared stem-to-file-path
    lookup from ``discover_source_files``."""
    files_cfg = cfg.get("files", "auto")
    entries = resolve_file_entries(files_cfg, discovered)
    if files_cfg is None or files_cfg == "auto":
        if not entries:
            raise FileNotFoundError(
                f"No supported source files found under {patient_src}; "
                f"supported extensions: {list(SUPPORTED_SOURCE_EXTENSIONS)}"
            )
        log.info("ingest_auto_discovered", extra_={"stems": [e["stem"] for e in entries]})
        return entries

    configured = {e["stem"] for e in entries}
    skipped = [s for s in sorted(discovered) if s not in configured]
    if skipped:
        log.warning(
            "ingest_unconfigured_files_skipped",
            extra_={
                "stems": skipped,
                "hint": "add these stems to the 'files' list or set files: auto",
            },
        )
    return entries


# The form Excel rewrites a large integer into when a CSV is opened and saved in it
# (e.g. 131019000000 -> "1.31019E+11"). We read every column as text and never create
# these, but a file someone opened in Excel arrives already corrupted.
_SCIENTIFIC_NOTATION = re.compile(r"^\s*-?\d+(\.\d+)?[eE][+-]?\d+\s*$")


def _excel_mangled_columns(df: pl.DataFrame) -> list[tuple[str, str]]:
    """Columns holding scientific-notation values, with one example each — the
    tell-tale that a CSV was opened and saved in Excel (which rewrites large IDs like
    131019000000 to '1.31019E+11' and drops leading zeros). A 200-value sample per
    column suffices because Excel mangles a whole column at once."""
    found: list[tuple[str, str]] = []
    for col in df.columns:
        series = df[col]
        if series.dtype != pl.Utf8:
            continue
        for value in series.drop_nulls().head(200).to_list():
            if isinstance(value, str) and _SCIENTIFIC_NOTATION.match(value):
                found.append((col, value))
                break
    return found


def _warn_if_excel_mangled(df: pl.DataFrame, stem: str, log) -> None:
    """Warn loudly for each Excel-mangled column. EHR CSV exports must never be opened in
    Excel; re-export cleanly if this fires."""
    for col, example in _excel_mangled_columns(df):
        log.warning("excel_mangled_column_suspected", extra_={
            "stem": stem, "column": col, "example": example,
            "hint": "values look like Excel scientific notation; a CSV opened/saved in "
                    "Excel corrupts large IDs and drops leading zeros — re-export from "
                    "the source without opening it in Excel.",
        })


def _strip_empty_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that are completely empty (every column blank) — trailing empty
    rows that some EHR exports leave behind. A row is kept if any cell has a
    value."""
    if df.is_empty():
        return df
    return df.filter(pl.any_horizontal(pl.all().is_not_null()))


# ---------------------------------------------------------------------------
# Parquet output
#
# The compression method used for the parquet is recorded in the sidecar
# (companion metadata file), so an audit that re-runs the pipeline can notice if
# it changed, independently of the file's raw bytes. If you switch compression
# methods, also update the allowed list in ingest_file.schema.json.
# ---------------------------------------------------------------------------
_PARQUET_COMPRESSION = "snappy"


def write_dataframe_parquet_atomically(df: pl.DataFrame, parquet_path: Path) -> dict:
    """Write ``df`` to ``parquet_path`` all-or-nothing — a temp file then a rename, snappy
    compression — so a crash can never leave a half-written parquet behind, and return the
    parquet metadata every structured-parquet sidecar records (row/column counts, the
    column schema, the parquet content hash, the polars version, and the compression).
    Callers add their own provenance fields and write their own sidecar. Sharing this one
    write is what guarantees an ingested parquet and a builder-derived parquet are
    byte-identical for the same dataframe."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=_PARQUET_COMPRESSION)
    tmp.rename(parquet_path)
    return {
        "row_count": df.height,
        "column_count": df.width,
        "columns": [{"name": c, "dtype": str(dt)} for c, dt in df.schema.items()],
        "parquet_content_hash": hash_file(parquet_path),
        "polars_version": pl.__version__,
        "compression": _PARQUET_COMPRESSION,
    }


def resolve_metadata_columns_for(
    columns: list[str], cfg: dict, source_name: str
) -> dict[str, str | None]:
    """Which source column holds each piece of chart metadata for one file.

    THE mapping step. A site's export names its columns however it names them; every
    step after this one reads the standard names (author, document_date, specialty, ...).
    Resolving it here — once, at ingest — means the decision is made against the real
    file, recorded in that file's sidecar, and checked by preflight before any run.

    Columns are NOT renamed. An ingested parquet stays a faithful copy of its source
    file (that is what its content hash attests), so the mapping is recorded beside the
    data rather than applied to it.
    """
    aliases = columns_from_config(cfg, source_name)
    return {
        field_name: find_column(field_aliases, columns)
        for field_name, field_aliases in aliases.items()
    }


def _write_parquet_with_sidecar(
    df: pl.DataFrame,
    parquet_path: Path,
    *,
    run_id: str,
    patient_id: str,
    source_file: str,
    code_lock_hash: str | None,
    source_content_hash: str,
    metadata_columns: dict[str, str | None] | None = None,
    data_columns: dict[str, str] | None = None,
) -> dict:
    """Write the parquet table plus its schema-validated ``ingest_file`` sidecar. The
    sidecar records a content hash of the source file, a content hash of the parquet, the
    polars version, and the compression method — enough for an auditor to figure out why
    two runs on the same input produced different parquet bytes."""
    common = write_dataframe_parquet_atomically(df, parquet_path)
    meta_payload = {
        "patient_id": patient_id,
        "source_file": source_file,
        "parquet_path": parquet_path.name,
        "source_content_hash": source_content_hash,
        # Which source column each standard metadata field came from, resolved against
        # this file's real header. Null means the file carries no such column.
        "metadata_columns": metadata_columns or {},
        # {our name: this file's column} for the table's own data — what a recipe
        # reading this table directly asks for by name.
        "data_columns": data_columns or {},
        **common,
    }
    env = envelope_for(
        artifact_type="ingest_file",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="ingest",
        patient_id=patient_id,
        payload=meta_payload,
        code_lock_hash=code_lock_hash,
        parent_artifacts=[{"path": source_file, "content_hash": source_content_hash}],
    )
    # This sidecar lives right next to its parquet, so we pass the path explicitly;
    # write_artifact still handles fingerprinting, validation, and the all-or-nothing
    # write.
    write_artifact(env, path=parquet_path.with_suffix(".parquet.meta.json"))
    return env


def _cached_files_written_entry(structured_dir: Path, run_root: Path, stem: str) -> dict:
    """Rebuild one file's ``files_written`` summary entry from its parquet sidecar,
    so a cache hit returns the same ``{stem, rows, cols, parquet}`` shape as a
    fresh ingest. The sidecar is always present here — the cache only counts as a
    hit when it exists."""
    parquet_path = structured_dir / f"{stem}.parquet"
    sidecar = json.loads(
        (structured_dir / f"{stem}.parquet.meta.json").read_text(encoding="utf-8")
    )
    payload = sidecar["payload"]
    return {
        "stem": stem,
        "rows": int(payload["row_count"]),
        "cols": int(payload["column_count"]),
        "parquet": parquet_path.relative_to(run_root).as_posix(),
    }


# ---------------------------------------------------------------------------
# Reconciling the structured/ folder with the sources still on disk
#
# Step 2 (embed) reads every parquet in structured/. So a parquet left behind
# after its source file was deleted would keep feeding into the pipeline forever.
# On a fresh ingest we compare structured/ against the source files still present:
# under auto-discovery, move (never delete) the now-orphaned parquet + sidecar
# into a quarantine folder, so they stop feeding embedding but can still be
# recovered; under an explicit files: list, only warn — the operator controls
# that set on purpose.
# ---------------------------------------------------------------------------

def _orphaned_structured_stems(structured_dir: Path, present_source_stems: set[str]) -> list[str]:
    """File stems that still have a ``structured/<stem>.parquet`` but whose
    original source file is no longer on disk (an "orphan"). ``present_source_stems``
    is every stem that currently has a source file (the keys of the discovery
    lookup). Sorted so log output is stable from run to run."""
    if not structured_dir.is_dir():
        return []
    return sorted(
        p.stem for p in structured_dir.glob("*.parquet") if p.stem not in present_source_stems
    )


def _quarantine_structured_stem(structured_dir: Path, quarantine_dir: Path, stem: str) -> list[str]:
    """Move ``<stem>.parquet`` and its sidecar out of structured/ into the
    quarantine dir, replacing a same-named prior orphan (the quarantine holds
    the latest). Returns the moved file names."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for name in (f"{stem}.parquet", f"{stem}.parquet.meta.json"):
        src = structured_dir / name
        if src.is_file():
            src.replace(quarantine_dir / name)
            moved.append(name)
    return moved


def _reconcile_structured_dir(
    *, structured_dir: Path, quarantine_dir: Path, files_cfg, present_source_stems: set[str], log,
) -> dict[str, list[str]]:
    """Compare structured/ with the source files still on disk. An "orphan" is a
    parquet whose source file is gone (its stem is absent from
    ``present_source_stems``, the keys of the discovery lookup) — judged purely by
    whether the source still exists, not by which stems the config selects.
    Orphans are quarantined under auto-discovery, and only warned about under an
    explicit files: list. Returns ``{'quarantined': [...stems], 'warned': [...stems]}``."""
    orphans = _orphaned_structured_stems(structured_dir, present_source_stems)
    is_auto = files_cfg is None or files_cfg == "auto"
    quarantined: list[str] = []
    warned: list[str] = []
    for stem in orphans:
        if is_auto:
            _quarantine_structured_stem(structured_dir, quarantine_dir, stem)
            quarantined.append(stem)
        else:
            warned.append(stem)
    if quarantined:
        log.warning("ingest_quarantined_orphan_parquets", extra_={"stems": quarantined})
    if warned:
        log.warning(
            "ingest_orphan_parquets_unmanaged",
            extra_={
                "stems": warned,
                "hint": "source file gone but files: list is explicit; left in place, not quarantined",
            },
        )
    return {"quarantined": quarantined, "warned": warned}


# ---------------------------------------------------------------------------
# Orchestrator
#
# The main entry point. CLI and quickstart both land here. Wires every helper
# above into one ordered flow per patient.
# ---------------------------------------------------------------------------

def run_ingest_one(
    *,
    cfg: dict,
    patient_id: str,
    code_lock_hash: str | None = None,
    force: bool = False,
) -> dict:
    """Ingest one patient from start to finish. Returns a summary dict
    (``{patient_id, cached, files_written}``).

    Safe to re-run: if a recorded snapshot (``source_snapshot.json``) shows the
    source files are unchanged and every expected parquet already exists, it skips
    the work and reports ``cached``. ``force=True`` ignores that and re-does
    everything. ``code_lock_hash`` is a fingerprint of the exact pipeline code
    version, passed in from the command line so outputs can be tied back to the
    code that made them; programmatic and test callers may pass None."""
    validate_ingest_config(cfg)

    # Reject an unsafe patient id up front — before it is ever used to build a file
    # path. This blocks path-traversal tricks (e.g. ".." or slashes that could
    # escape the patient folder) and the ':' character, which would corrupt the
    # '{patient_id}:{stem}:{row}:{idx}' chunk identifiers used later. This is the
    # central check; the per-patient path helpers re-check as a backstop.
    validate_patient_id(patient_id)

    run_id = cfg["run_id"]
    source_root = _resolve_source_root(cfg)
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)

    patient_src = source_root / patient_id
    if not patient_src.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {patient_src}")

    patient_out = phi_patient_run_dir(run_id, patient_id)
    patient_out.mkdir(parents=True, exist_ok=True)
    structured_dir = patient_out / "structured"
    structured_dir.mkdir(parents=True, exist_ok=True)

    log = _log.bind(run_id=run_id, patient_id=patient_id)
    log.info("ingest_start")

    # Find the patient's source files once, into a single stem-to-file-path lookup.
    # The snapshot, the cache check, and the read loop below all use this same
    # lookup, so they can't disagree about which file a stem points to (for
    # instance if the filename's capitalization differs from what's configured).
    discovered = discover_source_files(patient_src)
    file_entries = _resolve_file_entries(cfg, patient_src, discovered, log)

    # Cache check: skip the work only if the source-file fingerprints still match
    # the snapshot AND every expected parquet plus its sidecar already exists.
    # Restrict the check to stems whose source file is actually present, so an
    # optional file that's simply absent doesn't keep invalidating the cache
    # forever. The sidecar must exist too: a parquet with no sidecar can't be used
    # by later steps (which read the sidecar), so it doesn't count as a cache hit.
    if not force and snapshot_matches(patient_out, patient_src):
        present_stems = [e["stem"] for e in file_entries if e["stem"] in discovered]
        outputs_complete = bool(present_stems) and all(
            (structured_dir / f"{stem}.parquet").is_file()
            and (structured_dir / f"{stem}.parquet.meta.json").is_file()
            for stem in present_stems
        )
        if outputs_complete:
            log.info("ingest_skip_cached")
            record_transition(
                run_root,
                entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="ingest"),
                from_state=None,
                to_state="completed",
                reason="cached: source_snapshot matches and parquet outputs exist",
                step_context="ingest",
                code_lock_hash=code_lock_hash,
            )
            return {
                "patient_id": patient_id,
                "cached": True,
                "files_written": [
                    _cached_files_written_entry(structured_dir, run_root, stem)
                    for stem in present_stems
                ],
                "files_asked_for": len(file_entries),
            }

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="ingest"),
        from_state=None,
        to_state="running",
        reason="ingest_start",
        step_context="ingest",
        code_lock_hash=code_lock_hash,
    )

    try:
        # code_lock_hash (the pipeline-code fingerprint) is None for test and
        # programmatic callers; the metadata schema allows that.
        snapshot_path = write_source_snapshot(
            run_id=run_id,
            patient_id=patient_id,
            patient_dir=patient_src,
            patient_out_dir=patient_out,
            code_lock_hash=code_lock_hash,
        )
        log.info("source_snapshot_written", extra_={"path": snapshot_path.name})

        # Before re-writing, reconcile structured/ against the source files still
        # on disk: a parquet whose source was deleted must not keep feeding into
        # Step 2 (embed). An orphan is defined by the source being gone (its stem
        # is missing from the discovery lookup), not by what the config selects —
        # so a configured stem whose source was deleted IS caught, while a file
        # that's present but deliberately not selected is left alone.
        _reconcile_structured_dir(
            structured_dir=structured_dir,
            quarantine_dir=quarantined_structured_dir(run_id, patient_id),
            files_cfg=cfg.get("files", "auto"),
            present_source_stems=set(discovered),
            log=log,
        )

        files_written: list[dict[str, Any]] = []
        for entry in file_entries:
            stem = entry["stem"]
            if ":" in stem:
                raise ValueError(
                    f"source stem {stem!r} contains ':', which collides with the "
                    "chunk-id separator; rename the source file."
                )
            src = discovered.get(stem)
            if src is None:
                if entry["optional"]:
                    log.info("ingest_missing_optional", extra_={"stem": stem})
                    continue
                raise FileNotFoundError(
                    f"No source file found for stem {stem!r} under {patient_src}; "
                    f"tried extensions: {list(SUPPORTED_SOURCE_EXTENSIONS)}"
                )

            df = _read_source(src)
            _warn_if_excel_mangled(df, stem, log)
            df = _strip_empty_rows(df)
            df = _canonicalize_dates(df)

            # THE mapping step: which of this file's columns holds each piece of chart
            # metadata. Resolved against the real header, logged, and recorded in the
            # sidecar so every later step reads one settled answer instead of guessing
            # again. Preflight has already reported anything missing.
            metadata_columns = resolve_metadata_columns_for(df.columns, cfg, stem)
            # The table's own data, under the names recipes use. Resolved the same way
            # and recorded beside it, so a step reading this table directly does not
            # have to know what this site calls its columns.
            data_columns = {
                our_name: column
                for our_name, column in data_columns_for(cfg, stem).items()
                if column in df.columns
            }
            log.info("ingest_metadata_columns", extra_={
                "stem": stem,
                "resolved": metadata_columns,
                "not_found": [f for f, c in metadata_columns.items() if c is None],
                "data_columns": len(data_columns),
            })

            parquet_path = structured_dir / f"{stem}.parquet"
            _write_parquet_with_sidecar(
                df,
                parquet_path,
                run_id=run_id,
                patient_id=patient_id,
                source_file=src.name,
                code_lock_hash=code_lock_hash,
                source_content_hash=hash_file(src),
                metadata_columns=metadata_columns,
                data_columns=data_columns,
            )
            files_written.append({
                "stem": stem,
                "rows": int(df.height),
                "cols": int(df.width),
                "parquet": parquet_path.relative_to(run_root).as_posix(),
            })
            log.info("ingest_file_ok", extra_={"stem": stem, "rows": int(df.height)})

        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="ingest"),
            from_state="running",
            to_state="completed",
            reason=f"wrote {len(files_written)} parquet files",
            step_context="ingest",
            code_lock_hash=code_lock_hash,
        )
        log.info("ingest_done", extra_={"files": len(files_written)})
        # files_asked_for lets the caller say "19 of 19" rather than "19". A count
        # on its own cannot be checked: a patient with 17 source files and a patient
        # who lost two both read as 17.
        return {"patient_id": patient_id, "cached": False,
                "files_written": files_written, "files_asked_for": len(file_entries)}
    except Exception as exc:
        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="ingest"),
            from_state="running",
            to_state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            step_context="ingest",
            code_lock_hash=code_lock_hash,
        )
        raise


# ---------------------------------------------------------------------------
# Preflight
#
# A dry run: scan every patient for problems and write nothing. One bad chart
# does not stop the scan — all the problems across the cohort surface at once, so
# they can be fixed in a single pass. This is mainly a convenience for
# development and the quickstart notebook; for very large cohorts (e.g. 30k
# patients) it's cheaper to let each SLURM array job simply skip the patients it
# can't process than to scan them all serially up front.
# ---------------------------------------------------------------------------

@dataclass
class PreflightIssue:
    """One problem found for one patient during the dry-run scan. See
    _preflight_one and preflight_patients for the set of ``kind`` labels used.

    ``blocking`` separates "this patient cannot be ingested" from "ingest will work,
    but you should know this". A chart metadata column Junior could not find is the
    second kind: most tables legitimately carry only some of the metadata, so refusing
    the patient would refuse every real cohort — but the operator still needs to see it
    BEFORE a run, because a recipe filtering on that field will drop all of the file's
    evidence."""
    patient_id: str
    kind: str
    file: str | None
    detail: str
    blocking: bool = True
    # The chart fields this issue is about, when it is about fields. ``detail`` names
    # them too, but as prose, and a caller that wants to group a cohort's issues BY
    # FIELD — every table missing `specialty`, rather than every table's own sentence —
    # would have to parse them back out of it.
    fields: tuple[str, ...] = ()
    # Whether the file this issue is about contributes any text to search. An unmapped
    # metadata field only costs something where there is evidence to filter: a labs
    # table with no author is normal and nothing is ever quoted from it, so an operator
    # surface that reported every such table would bury the handful that matter.
    file_contributes_text: bool = False


@dataclass
class PreflightReport:
    """Result of scanning a whole cohort. Splits patients into OK vs. blocked and
    keeps the detail of every problem, so the user can fix all the data-prep
    issues in one pass."""
    patients_checked: list[str]
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def blocked_patient_ids(self) -> set[str]:
        return {i.patient_id for i in self.issues if i.blocking}

    @property
    def advisories(self) -> list[PreflightIssue]:
        """Issues worth showing that do not stop a patient being ingested."""
        return [i for i in self.issues if not i.blocking]

    @property
    def ok(self) -> list[str]:
        """Patients with zero issues — safe to ingest."""
        blocked = self.blocked_patient_ids
        return [p for p in self.patients_checked if p not in blocked]

    @property
    def blocked(self) -> list[str]:
        """Patients with at least one issue — don't ingest until fixed."""
        blocked = self.blocked_patient_ids
        return [p for p in self.patients_checked if p in blocked]

    def summary(self) -> str:
        """Human-readable summary — print this in the quickstart."""
        n = len(self.patients_checked)
        ok = len(self.ok)
        bad = len(self.blocked)
        lines = [
            f"PREFLIGHT REPORT — {n} patient(s) scanned",
            "=" * 70,
            f"  OK:       {ok}",
            f"  Blocked:  {bad}",
            "",
        ]
        if bad:
            by_patient: dict[str, list[PreflightIssue]] = {}
            for issue in self.issues:
                by_patient.setdefault(issue.patient_id, []).append(issue)
            lines.append("Blocked patient(s):")
            for pid in sorted(by_patient):
                lines.append(f"  {pid}:")
                for issue in by_patient[pid]:
                    file_part = f" [{issue.file}]" if issue.file else ""
                    lines.append(f"    - ({issue.kind}){file_part}  {issue.detail}")
            lines.append("")
        if bad and ok:
            lines.append(f"To ingest the {ok} OK patient(s) now, run the next cell.")
            lines.append("To re-check after fixing the blocked CSVs, re-run this cell.")
        elif bad and not ok:
            lines.append("All patients blocked — fix the issues above and re-run this cell.")
        elif ok and not bad:
            lines.append("All patients clean. Run the next cell to ingest.")
        return "\n".join(lines)


def _preflight_one(*, cfg: dict, patient_id: str, source_root: Path,
                   issues: list[PreflightIssue]) -> None:
    """Dry-run scan for one patient. Adds any problems to the ``issues`` list
    instead of raising, so one bad patient doesn't stop the whole cohort scan."""
    patient_src = source_root / patient_id
    if not patient_src.is_dir():
        issues.append(PreflightIssue(
            patient_id=patient_id,
            kind="missing_folder",
            file=None,
            detail=f"patient folder not found at {patient_src}",
        ))
        return

    patient_text_files: list[str] = []
    discovered = discover_source_files(patient_src)
    files_cfg = cfg.get("files", "auto")
    file_entries = resolve_file_entries(files_cfg, discovered)
    if (files_cfg is None or files_cfg == "auto") and not file_entries:
        issues.append(PreflightIssue(
            patient_id=patient_id,
            kind="no_files",
            file=None,
            detail=(
                f"folder has no supported source files "
                f"(looked for {list(SUPPORTED_SOURCE_EXTENSIONS)})"
            ),
        ))
        return

    for entry in file_entries:
        stem = entry["stem"]
        src = discovered.get(stem)
        if src is None:
            if not entry["optional"]:
                issues.append(PreflightIssue(
                    patient_id=patient_id,
                    kind="missing_required_file",
                    file=None,
                    detail=(
                        f"no file matches required stem {stem!r} "
                        f"(tried {list(SUPPORTED_SOURCE_EXTENSIONS)})"
                    ),
                ))
            continue

        # Read the file; capture any reader error rather than raising.
        try:
            df = _read_source(src)
        except Exception as e:
            issues.append(PreflightIssue(
                patient_id=patient_id,
                kind="unreadable",
                file=src.name,
                detail=f"{type(e).__name__}: {str(e)[:200]}",
            ))
            continue

        # Resolve the chart metadata mapping for this file and report what it could not
        # find. Done here, before a single parquet is written, because a missing column
        # is a data-prep problem the operator fixes in the source or in this site's
        # column map. ONE advisory per file, listing the fields — a row per field would
        # bury a real problem under dozens of lines a cohort scan multiplies by every
        # patient.
        # Free text is what gets embedded. A file without any is normal (a lab table);
        # a PATIENT without any cannot produce a corpus at all, which is checked after
        # this loop. Caught here rather than at embed, because by then the operator has
        # already ingested a whole cohort.
        text_named = [c for c in text_columns_for(cfg, src.name) if c in df.columns]
        default_text = cfg.get("text_column", "text")
        if text_named or default_text in df.columns:
            patient_text_files.append(src.name)

        resolved = resolve_metadata_columns_for(df.columns, cfg, src.name)
        not_found = [field for field, column in resolved.items() if column is None]
        if not_found:
            issues.append(PreflightIssue(
                patient_id=patient_id,
                kind="metadata_column_not_found",
                file=src.name,
                detail=(
                    f"no column found for: {', '.join(not_found)}. Chunks from this file "
                    f"carry no value for {'it' if len(not_found) == 1 else 'any of them'}, "
                    f"so a recipe filtering on "
                    f"{'it' if len(not_found) == 1 else 'one of them'} drops all of this "
                    "file's evidence. Map the column in this site's column map "
                    "(chart_columns_file) if the file has one"
                ),
                blocking=False,
                fields=tuple(not_found),
                file_contributes_text=bool(text_named or default_text in df.columns),
            ))
        for collapsed, names in ambiguous_columns(df.columns).items():
            issues.append(PreflightIssue(
                patient_id=patient_id,
                kind="ambiguous_columns",
                file=src.name,
                detail=(
                    f"columns {names} differ only by case or spacing; only {names[0]!r} "
                    f"can be read as {collapsed!r} — rename the others"
                ),
                blocking=False,
            ))

        # Check each text column for date-shaped values and flag any 2-digit-year
        # dates (the ones ingest refuses to interpret).
        if df.is_empty():
            continue
        for col_name, dtype in df.schema.items():
            if dtype != pl.Utf8:
                continue
            # Follow _canonicalize_dates EXACTLY: first decide from a sample whether
            # this is a date column, then scan the WHOLE column for offenders. The
            # sample only looks at the first 100 rows, so a bad value past that
            # point would pass preflight and then crash the real ingest with
            # AmbiguousYearError — breaking the promise that any patient preflight
            # marks OK can actually be ingested. The whole-column scan closes that
            # gap.
            is_date, _ = _sample_date_offenders(df[col_name])
            if not is_date:
                continue
            offenders = _two_digit_year_offenders(df[col_name])
            if not offenders:
                continue
            preview = ", ".join(repr(v) for v in offenders[:5])
            issues.append(PreflightIssue(
                patient_id=patient_id,
                kind="ambiguous_year",
                file=src.name,
                detail=(
                    f"column {col_name!r} has 2-digit years "
                    f"({len(offenders)} found, e.g. {preview}); "
                    "re-export with M/D/YYYY"
                ),
            ))

    if not patient_text_files:
        issues.append(PreflightIssue(
            patient_id=patient_id,
            kind="no_text_to_embed",
            file=None,
            detail=(
                "no file has a free-text column, so this patient produces no evidence to "
                f"search. Junior looked for the columns this site's map names, else "
                f"{cfg.get('text_column', 'text')!r}. If the text lives under another "
                "name, add it as text_columns in the column map — "
                "`junior columns <patient folder>` drafts one from your own headers"
            ),
        ))


def preflight_patients(*, cfg: dict, patients: list[str]) -> PreflightReport:
    """Dry-run scan of every patient, collecting any problems: missing folders,
    missing files, files that can't be read, and 2-digit-year dates. Writes
    nothing. Feed ``report.ok`` (the patients with no problems) into the actual
    ingest.

    If the scan itself unexpectedly crashes on a patient, that patient gets a
    ``preflight_error`` problem instead of taking down the whole scan."""
    validate_ingest_config(cfg)
    source_root = _resolve_source_root(cfg)
    issues: list[PreflightIssue] = []
    for pid in patients:
        try:
            _preflight_one(
                cfg=cfg, patient_id=pid, source_root=source_root, issues=issues,
            )
        except Exception as e:
            issues.append(PreflightIssue(
                patient_id=pid,
                kind="preflight_error",
                file=None,
                detail=f"preflight raised {type(e).__name__}: {e}",
            ))
    return PreflightReport(patients_checked=list(patients), issues=issues)
