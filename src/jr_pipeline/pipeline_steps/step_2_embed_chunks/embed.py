"""Embed turns the cleaned-up tables from ingest into, for each patient, a matrix
of embedding vectors plus a small companion index file.

Flow, in three passes:
  pass 0 — walk every source table and collect the text cells to embed
  pass 1 — split each cell into chunks and record them in a list (the "manifest")
  pass 2 — send the chunks to the model in batches and collect their vectors

Three files land in the patient's directory:

  embeddings.npy        an [N rows, dim] float32 matrix, each row scaled to
                        length 1. The file is written with a proper header so it
                        still loads even when a patient has no embeddable text
                        (N = 0 rows).
  chunk_index.parquet   for each row of embeddings.npy, where that chunk came from
                        (which source row, and the character offsets within it). It
                        stores no text itself — PatientChunkStore.text_for reads the
                        actual chunk text from the source on demand.
  chunk_index.parquet.meta.json   companion metadata: the encoder fingerprint,
                        chunker identity, and source hashes the cache check reads.

Re-running is safe and fast: the step skips a patient only when BOTH files already
exist AND the encoder recorded in the companion file matches the currently
configured encoder. That catches a half-written result left by an earlier crash AND
the case where someone edited the encoder settings without changing run_id (which
would otherwise leave stale vectors in place). The cache is per-patient, so running
one job per patient on the cluster (a SLURM array) is safe.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import polars as pl

from jr_pipeline.pipeline_steps.step_2_embed_chunks import build_encoder
from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
    VECTOR_AFFECTING_FINGERPRINT_FIELDS,
    Encoder,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_bytes,
    hash_file,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    STANDARD_METADATA_FIELDS,
    columns_from_config,
    find_column,
    identifier_columns_for,
    match_key,
    metadata_value_as_text,
    text_columns_for,
)
from jr_pipeline.runtime_infrastructure.config_loading import validate_embed_config
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    ensure_layout,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    structured_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

# ── embed config defaults ────────────────────────────────────────────────────
# All keys below come from the project cfg dict. These apply when a key is
# absent. Set them explicitly in config to override.
#
#   chunker.kind       "token_window" = slide a fixed-size window across the text
#                      to make several overlapping chunks; "none" = one chunk per
#                      table row (just truncated if too long)
#   chunker.window     tokens per chunk (kind=token_window); omit to inherit
#                      encoder.max_tokens so the default tracks your model
#   chunker.max_tokens token limit (kind=none); omit to inherit encoder.max_tokens
#   chunker.overlap    tokens shared between adjacent chunks; 128 ≈ 25% of a
#                      512-token window — overlap keeps a phrase that lands on a
#                      chunk boundary from being split across two chunks (at the
#                      cost of more chunks); raise it to catch more boundary cases,
#                      lower it to make the index smaller
#   batch_size        how many chunks the model encodes at once; 16 is safe on
#                     modest GPUs; raise to 64-128 on large-memory hardware
#   text_column             default table column to embed in auto mode
#   files                   None / "auto" → embed text_column from every structured
#                           table that has it; an explicit list uses text_columns
#   parquet_compression     snappy is the jr pipeline default
#
# Chunk metadata (date, age, type, author, title, encounter id, specialty) comes from
# the site's column map — see runtime_infrastructure/chart_metadata_fields.py. The map
# also names each file's free-text columns, which is what auto mode embeds.

_DEFAULT_CHUNKER: dict = {"kind": "token_window", "overlap": 128}
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_TEXT_COLUMN = "text"
# Spelled as the one literal polars accepts, so a typo here is caught before a run.
_DEFAULT_PARQUET_COMPRESSION: Literal["snappy"] = "snappy"
_FILES_AUTO = "auto"

# ── chunking types ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """One chunk plus the character span [char_start, char_end) it came from in the
    source text (start included, end excluded)."""

    text: str
    char_start: int
    char_end: int
    chunk_idx: int
    token_count: int
    parent_tokens: int


class _TokenizerWithOffsets(Protocol):
    def tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]: ...


class _Chunker(Protocol):
    def chunk(self, text: str, *, tokenizer: _TokenizerWithOffsets) -> list[Chunk]: ...


@dataclass(frozen=True)
class NoChunker:
    """one chunk per text, truncated to max_tokens."""

    max_tokens: int

    def chunk(self, text: str, *, tokenizer: _TokenizerWithOffsets) -> list[Chunk]:
        if not text:
            return []
        toks = tokenizer.tokenize_with_offsets(text)
        if not toks:
            return []
        kept = toks[: self.max_tokens]
        char_start = kept[0][1]
        char_end = kept[-1][2]
        return [Chunk(
            text=text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
            chunk_idx=0,
            token_count=len(kept),
            parent_tokens=len(toks),
        )]


@dataclass(frozen=True)
class TokenWindowChunker:
    """Slide a fixed-size window across the tokens, with some overlap between
    neighboring windows; overlap must be smaller than the window so the start
    position always moves forward."""

    window: int
    overlap: int

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.window:
            raise ValueError("overlap must be strictly less than window")

    def chunk(self, text: str, *, tokenizer: _TokenizerWithOffsets) -> list[Chunk]:
        if not text:
            return []
        toks = tokenizer.tokenize_with_offsets(text)
        if not toks:
            return []
        total = len(toks)
        step = self.window - self.overlap
        out: list[Chunk] = []
        start = 0
        idx = 0
        while start < total:
            end = min(start + self.window, total)
            window = toks[start:end]
            char_start = window[0][1]
            char_end = window[-1][2]
            out.append(Chunk(
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                chunk_idx=idx,
                token_count=len(window),
                parent_tokens=total,
            ))
            idx += 1
            if end == total:
                break
            start += step
        return out


def chunk_span_in_tokens(chunker: _Chunker) -> int:
    """The most content tokens one chunk from this chunker can hold.

    The two chunkers record it under different names — the sliding chunker's window, the
    one-chunk chunker's truncation limit — so the comparison against the encoder's
    content budget and the span recorded in the chunk-index metadata both read it here
    rather than each re-deriving it."""
    if isinstance(chunker, TokenWindowChunker):
        return chunker.window
    if isinstance(chunker, NoChunker):
        return chunker.max_tokens
    raise ValueError(
        "chunker reports no token span; only kinds 'token_window' and 'none' do"
    )


def build_chunker(cfg: dict) -> _Chunker:
    kind = cfg.get("kind", "token_window")
    if kind == "token_window":
        return TokenWindowChunker(window=int(cfg["window"]), overlap=int(cfg["overlap"]))
    if kind == "none":
        return NoChunker(max_tokens=int(cfg["max_tokens"]))
    raise ValueError(f"Unknown chunker kind: {kind!r}")


def build_chunker_with_reserved_specials(cfg: dict, encoder) -> _Chunker:
    """Build the chunker, leaving room for the encoder's special marker tokens.

    The chunker counts only real content tokens (``tokenize_with_offsets`` uses
    ``add_special_tokens=False``), but at embed time ``embed_batch`` adds the
    model's [CLS]/[SEP] start/end markers and then cuts the input off at
    ``max_tokens`` — so a chunk holding exactly ``max_tokens`` content tokens would
    silently lose its tail. To prevent that, an omitted window defaults to the
    *content budget* ``max_tokens - num_special_tokens``, and an explicit window
    larger than that budget is rejected loudly rather than quietly truncated.
    Reads ``encoder.num_special_tokens``, which loads the tokenizer — so call this
    only after the cache-skip check (never on a cache hit).
    """
    chunker_cfg = cfg.get("chunker") or _DEFAULT_CHUNKER
    kind = chunker_cfg.get("kind", "token_window")
    content_budget = encoder.max_tokens - encoder.num_special_tokens
    if kind == "token_window" and "window" not in chunker_cfg:
        chunker_cfg = {**chunker_cfg, "window": content_budget}
    elif kind == "none" and "max_tokens" not in chunker_cfg:
        chunker_cfg = {**chunker_cfg, "max_tokens": content_budget}
    chunker = build_chunker(chunker_cfg)
    chunk_span = chunk_span_in_tokens(chunker)
    if chunk_span > content_budget:
        raise ValueError(
            f"chunker span ({chunk_span} tokens) exceeds the encoder content budget "
            f"({content_budget} = max_tokens {encoder.max_tokens} - "
            f"{encoder.num_special_tokens} special tokens); lower the chunker window "
            "or pick a longer-context encoder."
        )
    return chunker


_log = get_logger("embed")

@dataclass(frozen=True)
class _Row:
    """One source text cell collected in pass 0, before it is split into chunks."""

    source_file: str
    source_stem: str
    row_id: int
    source_column: str
    text: str
    # One entry per STANDARD_METADATA_FIELDS name; None where this source had no such
    # column. A dict rather than a field each, so adding a metadata field to the
    # vocabulary needs no edit here.
    metadata: dict[str, str | None] = field(default_factory=dict)

def _source_file_for_stem(structured_dir: Path, stem: str) -> str:
    """The original source filename (e.g. clinical_note.xlsx), read from the
    companion metadata file ingest wrote. Provenance must point at the file the
    user actually gave us, not the internal parquet we converted it to."""
    sidecar = structured_dir / f"{stem}.parquet.meta.json"
    try:
        env = json.loads(sidecar.read_text(encoding="utf-8"))
        return env["payload"]["source_file"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"ingest sidecar missing or unreadable for {stem!r} ({sidecar.name}); "
            "re-run ingest for this patient."
        ) from exc

def _resolve_files_spec(
    structured_dir: Path, files_cfg, text_column: str, metadata_cfg: dict | None = None
) -> list[dict]:
    """"auto" or None → embed each table's free-text column; an explicit list is honored
    as written. An explicit entry must set ``embed: True`` to actually be embedded.

    In auto mode the site's column map decides which column that is, per file — a site
    whose notes live in ``note_text`` or ``report`` names it there instead of renaming
    the column. Only when the map is silent does this fall back to the configured
    ``text_column`` (``text``). A column the map marks as an identifier is never
    embedded, so a mapping mistake cannot put a patient name into every chunk.
    """
    if files_cfg is not None and files_cfg != _FILES_AUTO:
        return list(files_cfg)
    metadata_cfg = metadata_cfg or {}
    specs: list[dict] = []
    for p in sorted(structured_dir.glob("*.parquet")):
        schema = pl.read_parquet_schema(p)
        named = [c for c in text_columns_for(metadata_cfg, p.stem) if c in schema]
        chosen = named or ([text_column] if text_column in schema else [])
        # Compared the way the rest of the mapping compares column names, not with a
        # bare .lower(): a site that writes an identifier as "Patient Name" where the
        # export spells it patient_name would otherwise slip past this guard, and the
        # name it names would be embedded into every chunk of that table.
        identifiers = {match_key(c) for c in identifier_columns_for(metadata_cfg, p.stem)}
        embeddable = [c for c in chosen if match_key(c) not in identifiers]
        if len(embeddable) < len(chosen):
            _log.warning("embed_skipped_identifier_column", extra_={
                "stem": p.stem,
                "columns": [c for c in chosen if match_key(c) in identifiers],
                "hint": "the column map lists these as identifiers; they are never embedded",
            })
        if embeddable:
            specs.append({"stem": p.stem, "embed": True, "text_columns": embeddable})
    return specs

def _require_declared_columns(structured_dir: Path, files_spec: list[dict]) -> None:
    """Explicitly configured text_columns must really exist — otherwise a
    misspelled column name would embed nothing yet still report success."""
    for entry in files_spec:
        if not entry.get("embed", False):
            continue
        pq = structured_dir / f"{entry['stem']}.parquet"
        if not pq.is_file():
            raise FileNotFoundError(
                f"files[].stem {entry['stem']!r} has no structured parquet at {pq}; "
                "run ingest first or remove this embed file spec."
            )
        schema = pl.read_parquet_schema(pq)
        missing = [c for c in entry.get("text_columns") or [] if c not in schema]
        if missing:
            raise ValueError(
                f"files[].text_columns {missing} not in {pq.name} "
                f"(has: {sorted(schema)})"
            )

def _row_text_hash(text: str) -> str:
    # content hash of the whole source-row text (not a single chunk), stored on each chunk for audit purposes.
    return hash_bytes(text.encode("utf-8"))

def _source_parquet_hashes(structured_dir: Path, files_spec: list[dict]) -> dict[str, str]:
    """The content hash of each embeddable source table. If this map matches the one
    recorded last run, the sources are unchanged and we can skip without loading the
    model."""
    hashes: dict[str, str] = {}
    for entry in files_spec:
        if not entry.get("embed", False):
            continue
        stem = entry["stem"]
        pq = structured_dir / f"{stem}.parquet"
        if not pq.is_file():
            continue
        sidecar = structured_dir / f"{stem}.parquet.meta.json"
        try:
            env = json.loads(sidecar.read_text(encoding="utf-8"))
            hashes[stem] = env["payload"]["parquet_content_hash"]
        except (OSError, json.JSONDecodeError, KeyError):
            hashes[stem] = hash_file(pq)
    return hashes

def _cached_sidecar_field(idx_meta_path: Path, field: str) -> Any:
    """One field from the previous run's companion metadata file; None if missing or unreadable."""
    try:
        env = json.loads(idx_meta_path.read_text(encoding="utf-8"))
        return env["payload"][field]
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _cached_encoder_fingerprint(idx_meta_path: Path) -> dict[str, Any] | None:
    return _cached_sidecar_field(idx_meta_path, "encoder")


def _cached_source_hashes(idx_meta_path: Path) -> dict[str, str] | None:
    return _cached_sidecar_field(idx_meta_path, "source_parquet_hashes")


# Versions embed's OUTPUT logic (chunk-boundary math + chunk_index columns). The
# cache key otherwise sees only config and input hashes, so a code change would
# leave stale caches looking valid — increment this to force the rebuild.
#
# 4: chunk_index gained `age` and `specialty`. A cohort embedded before that keeps a
#    chunk index without those columns, and a recipe filtering on either one finds no
#    evidence in any chunk of it — an answer drawn from nothing, reported the same way
#    as an answer drawn from a chart that does not say.
_EMBED_OUTPUT_LOGIC_VERSION = 4


def _chunker_cache_identity(cfg: dict, files_spec: list[dict] | None = None) -> dict[str, Any]:
    """Everything that decides which chunks a patient ends up with, used as part of the
    embed cache key. Computable without loading the model, so a cache hit still loads
    nothing.

    Three things move it. The logic version, when embed's own output logic changes; the
    raw chunker config, when a setting changes; and WHICH COLUMNS ARE EMBEDDED, because
    the site's column map is a separate file an operator is expected to edit between
    runs. Adding a text column to that map changes nothing the other two legs can see —
    the source parquet is byte-identical and the chunker settings are untouched — so the
    run was skipped as cached and the column the operator had just asked for was never
    embedded, searched, or quoted as evidence."""
    chunker = cfg.get("chunker") or _DEFAULT_CHUNKER
    identity: dict[str, Any] = {
        "logic_version": _EMBED_OUTPUT_LOGIC_VERSION,
        "kind": chunker.get("kind", "token_window"),
        "window": chunker.get("window"),
        "overlap": chunker.get("overlap"),
        "max_tokens": chunker.get("max_tokens"),
    }
    if files_spec is not None:
        identity["embedded_columns"] = {
            str(entry.get("stem")): sorted(entry.get("text_columns") or [])
            for entry in files_spec
            if entry.get("embed", False)
        }
    return identity


def _cached_chunker_identity(idx_meta_path: Path) -> dict | None:
    return _cached_sidecar_field(idx_meta_path, "chunker")


def _index_columns() -> list[str]:
    return list(_CHUNK_INDEX_SCHEMA.keys())


def _resolve_metadata_columns(col_names: list[str], aliases: dict[str, list[str]]) -> dict[str, str | None]:
    """Which source column each metadata field will be read from (None if absent)."""
    return {
        field_name: find_column(field_aliases, col_names)
        for field_name, field_aliases in aliases.items()
    }


def _metadata_columns_from_sidecar(structured_dir: Path, stem: str) -> dict[str, str | None] | None:
    """The mapping ingest resolved for this file, from its sidecar, or None if absent.

    Ingest is where the mapping is decided (see resolve_metadata_columns_for). Reading its
    answer keeps embed from re-deriving one that could differ — a config edited between
    the two stages would otherwise silently pull metadata from different columns than the
    ones recorded beside the data. None means an older run, or a table ingest did not
    write (a builder table), so the caller falls back to resolving it here."""
    sidecar = structured_dir / f"{stem}.parquet.meta.json"
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8")).get("payload") or {}
    except (json.JSONDecodeError, OSError):
        return None
    recorded = payload.get("metadata_columns")
    return recorded or None


def _extract_row_metadata(row_dict: dict, resolved: dict[str, str | None]) -> dict[str, str | None]:
    """This row's value for each metadata field, from the columns already resolved."""
    return {
        field_name: metadata_value_as_text(
            row_dict.get(column) if column is not None else None
        )
        for field_name, column in resolved.items()
    }


def _log_resolved_metadata_columns(
    source_file: str, resolved: dict[str, str | None], aliases: dict[str, list[str]]
) -> None:
    """Record which source column each metadata field resolved to, and warn about every
    field that resolved to nothing.

    A field with no column is silently null on every chunk of this file, and a recipe
    filtering on it then drops all of that file's evidence. That has to be visible while
    the corpus is being built, not inferred later from an empty extraction — so each
    miss is warned individually, with the names that were looked for."""
    _log.info("embed_metadata_columns", extra_={"source_file": source_file, "resolved": resolved})
    for field_name, column in resolved.items():
        if column is not None:
            continue
        hint = (
            "this embedded text source has no recognizable date column; its evidence "
            "cannot be date-filtered or shown with a date"
            if field_name == "document_date"
            else f"every chunk from this file will have no {field_name}; a recipe "
                 f"filtering on {field_name} will drop all of its evidence"
        )
        _log.warning("embed_metadata_column_not_found", extra_={
            "source_file": source_file,
            "field": field_name,
            "looked_for": aliases[field_name],
            "hint": f"{hint} — add the column, or map it with chunk_metadata_columns "
                    "in this site's config.",
        })


def _iter_corpus(
    structured_dir: Path,
    files_spec: list[dict],
    metadata_cfg: dict,
) -> Iterable[_Row]:
    """Yield every embeddable row from the structured tables named in files_spec."""
    for entry in files_spec:
        if not entry.get("embed", False):
            continue
        stem = entry["stem"]
        text_columns = entry.get("text_columns") or []
        if not text_columns:
            continue
        path = structured_dir / f"{stem}.parquet"
        if not path.is_file():
            continue
        source_file = _source_file_for_stem(structured_dir, stem)
        df = pl.read_parquet(path)
        col_names = df.columns
        # Ingest already resolved this file's mapping and recorded it; re-derive only
        # when it did not (an older run, or a table ingest never wrote) or when what it
        # recorded no longer matches the table — a column named in the sidecar but
        # missing from the parquet would otherwise read as "no author" on every row.
        resolved_columns = _metadata_columns_from_sidecar(structured_dir, stem)
        stale = [
            column for column in (resolved_columns or {}).values()
            if column is not None and column not in col_names
        ]
        if stale:
            _log.warning("embed_metadata_mapping_stale", extra_={
                "source_file": source_file,
                "columns_missing_from_table": stale,
                "hint": "the sidecar's mapping does not match this parquet; re-resolving "
                        "from the table. Re-run ingest for this patient to settle it.",
            })
        if resolved_columns is None or stale:
            # Column maps are written per document type, so resolve with THIS file's
            # entry rather than one map shared across the corpus.
            aliases = columns_from_config(metadata_cfg, stem)
            resolved_columns = _resolve_metadata_columns(col_names, aliases)
            _log_resolved_metadata_columns(source_file, resolved_columns, aliases)
        rows_as_dicts = df.to_dicts()
        for col in text_columns:
            if col not in df.columns:
                continue
            for row_id, value in enumerate(df[col].to_list()):
                if not isinstance(value, str):
                    continue
                if not value.strip():
                    continue
                meta = _extract_row_metadata(rows_as_dicts[row_id], resolved_columns)
                # yield the original (un-trimmed) text — the character offsets point into it.
                yield _Row(
                    source_file=source_file,
                    source_stem=stem,
                    row_id=row_id,
                    source_column=col,
                    text=value,
                    metadata=meta,
                )

# source_text_sha256 is an audit content hash of the SOURCE ROW's full text (not of
# the individual chunk); source_stem ties each chunk back to the source table it
# came from.
# The values are the column-type classes themselves (pl.Utf8, pl.Int64), not instances
# of them, so the annotation names the class, not the type.
_CHUNK_INDEX_SCHEMA: dict[str, pl.datatypes.DataTypeClass] = {
    "chunk_id": pl.Utf8,
    "patient_id": pl.Utf8,
    "source_file": pl.Utf8,
    "source_stem": pl.Utf8,
    "row_id": pl.Int64,
    "source_column": pl.Utf8,
    "source_text_sha256": pl.Utf8,
    "chunk_idx": pl.Int32,
    "char_start": pl.Int64,
    "char_end": pl.Int64,
    "token_count": pl.Int32,
    "parent_tokens": pl.Int32,
    "num_chunks": pl.Int32,
    **{field_name: pl.Utf8 for field_name in STANDARD_METADATA_FIELDS},
}

def _fingerprint_subset(fp: dict[str, Any]) -> dict[str, Any]:
    # drop device/dim/etc so two fingerprints are compared only on the fields that actually change the output vectors.
    return {k: fp.get(k) for k in VECTOR_AFFECTING_FINGERPRINT_FIELDS}

def run_embed_one(
    *,
    cfg: dict,
    patient_id: str,
    code_lock_hash: str | None = None,
    force: bool = False,
) -> dict:
    """run embed for one patient. returns a summary dict."""
    validate_embed_config(cfg)

    run_id = cfg["run_id"]
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)

    patient_out = phi_patient_run_dir(run_id, patient_id)
    structured_root = structured_dir(patient_out)
    if not structured_root.is_dir():
        raise FileNotFoundError(
            f"Structured dir missing for patient {patient_id!r}; run ingest first."
        )

    emb_path = patient_out / "embeddings.npy"
    idx_path = patient_out / "chunk_index.parquet"
    idx_meta_path = patient_out / "chunk_index.parquet.meta.json"

    log = _log.bind(run_id=run_id, patient_id=patient_id)

    encoder = build_encoder(cfg["encoder"])
    batch_size = int(cfg.get("batch_size", _DEFAULT_BATCH_SIZE))
    files_cfg = cfg.get("files")
    text_column = cfg.get("text_column", _DEFAULT_TEXT_COLUMN)
    # This site's column map: which columns hold metadata, which hold free text, and
    # which are identifiers. Resolved per file, so one map serves a whole export.
    metadata_cfg = {
        k: cfg.get(k) for k in ("chart_columns_file", "chunk_metadata_columns")
        if cfg.get(k) is not None
    }
    files_spec = _resolve_files_spec(structured_root, files_cfg, text_column, metadata_cfg)
    if not files_spec:
        raise ValueError(
            f"No embeddable structured parquets found for patient {patient_id!r}. "
            f"Auto mode uses each file's text_columns from the site column map, else a "
            f"{text_column!r} column; set files_to_embed with explicit text_columns if "
            "your note text uses a different name."
        )
    enabled_specs = [entry for entry in files_spec if entry.get("embed", False)]
    if not enabled_specs:
        raise ValueError(
            "No files are enabled for embedding. Set files_to_embed='auto' or add "
            "`embed: True` plus `text_columns` to at least one Step 2 file spec."
        )
    missing_text_columns = [
        entry["stem"]
        for entry in enabled_specs
        if not entry.get("text_columns")
    ]
    if missing_text_columns:
        raise ValueError(
            "Step 2 file specs with `embed: True` must declare text_columns "
            f"(missing for stems: {missing_text_columns})."
        )
    if files_cfg not in (None, _FILES_AUTO):
        _require_declared_columns(structured_root, files_spec)
    source_hashes = _source_parquet_hashes(structured_root, files_spec)
    chunker_identity = _chunker_cache_identity(cfg, files_spec)

    # Three-part cache check (encoder fingerprint + source-table content hashes +
    # chunk identity), all done from hashes and config alone — none of them loads the
    # model. The chunk-identity part is what catches everything the other two cannot
    # see: a chunk-boundary change from the config or the logic version, and a column
    # map that now names a different set of text columns for the same unchanged tables.
    artifacts_present = emb_path.is_file() and idx_path.is_file() and idx_meta_path.is_file()
    encoder_matches = False
    if not force and artifacts_present:
        cached_fp = _cached_encoder_fingerprint(idx_meta_path)
        encoder_matches = cached_fp is not None and (
            _fingerprint_subset(cached_fp) == _fingerprint_subset(encoder.fingerprint())
        )
        chunker_matches = _cached_chunker_identity(idx_meta_path) == chunker_identity
        if (
            encoder_matches
            and chunker_matches
            and _cached_source_hashes(idx_meta_path) == source_hashes
        ):
            log.info("embed_skip_cached")
            record_transition(
                run_root,
                entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="embed"),
                from_state=None,
                to_state="completed",
                reason="cached: encoder fingerprint, chunk identity, and source hashes match",
                step_context="embed",
                code_lock_hash=code_lock_hash,
            )
            return {"patient_id": patient_id, "cached": True}
        if cached_fp is None:
            reason = "missing sidecar payload"
        elif not encoder_matches:
            reason = "encoder fingerprint mismatch"
        elif not chunker_matches:
            reason = "chunker config changed"
        else:
            reason = "source parquet hashes changed"
        log.info("embed_cache_busted", extra_={"reason": reason})

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="embed"),
        from_state=None,
        to_state="running",
        reason="embed_start",
        step_context="embed",
        code_lock_hash=code_lock_hash,
    )

    try:
        # We are past the cache-skip check here: building the chunker reads
        # encoder.num_special_tokens (which loads the tokenizer), so it never runs on
        # a cache hit. It's inside the try so a load/validation failure is recorded
        # as a running -> failed transition.
        chunker = build_chunker_with_reserved_specials(cfg, encoder)
        return _run_embed_body(
            patient_id=patient_id,
            code_lock_hash=code_lock_hash,
            run_id=run_id,
            run_root=run_root,
            structured_dir=structured_root,
            emb_path=emb_path,
            idx_path=idx_path,
            idx_meta_path=idx_meta_path,
            encoder=encoder,
            chunker=chunker,
            batch_size=batch_size,
            files_spec=files_spec,
            source_hashes=source_hashes,
            chunker_identity=chunker_identity,
            # Resolved per file inside the corpus walk.
            metadata_cfg=metadata_cfg,
            log=log,
        )
    except Exception as exc:
        # without this, a crash leaves the step stuck in the ``running`` state forever.
        record_transition(
            run_root,
            entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="embed"),
            from_state="running",
            to_state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            step_context="embed",
            code_lock_hash=code_lock_hash,
        )
        raise

def _run_embed_body(
    *,
    patient_id: str,
    code_lock_hash: str | None,
    run_id: str,
    run_root: Path,
    structured_dir: Path,
    emb_path: Path,
    idx_path: Path,
    idx_meta_path: Path,
    encoder: Encoder,
    chunker: _Chunker,
    batch_size: int,
    files_spec: list[dict],
    source_hashes: dict[str, str],
    chunker_identity: dict,
    log: Any,
    metadata_cfg: dict | None = None,
) -> dict:
    rows = list(_iter_corpus(structured_dir, files_spec, metadata_cfg or {}))

    log.info("embed_pass1_start")
    chunk_manifest: list[dict[str, Any]] = []
    for row in rows:
        chunk_manifest.extend(_chunks_for_row(row, patient_id, chunker, encoder))
    n = len(chunk_manifest)
    log.info("embed_pass1_done", extra_={"rows": len(rows), "chunks": n})

    _fill_num_chunks(chunk_manifest)

    # guard against a config that lists the same column twice — the vector index would otherwise store the duplicates as separate vectors.
    chunk_ids = [m["chunk_id"] for m in chunk_manifest]
    if len(chunk_ids) != len(set(chunk_ids)):
        dupes = [cid for cid, c in Counter(chunk_ids).items() if c > 1]
        raise RuntimeError(
            f"chunk_id uniqueness violated for patient {patient_id!r}: "
            f"{len(dupes)} duplicates (first: {dupes[:3]}); check files[].text_columns for duplicate entries."
        )

    log.info("embed_pass2_start", extra_={"batch_size": batch_size})
    mat = _encode_manifest(chunk_manifest, encoder, batch_size)
    dim = int(mat.shape[1])

    # the chunk index stores only offsets into the source, never the chunk text itself.
    for m in chunk_manifest:
        m.pop("_text", None)

    # INVARIANT (must always hold): row i of embeddings.npy describes the same chunk
    # as row i of chunk_index.parquet. The vector index identifies each vector by its
    # row position, and retrieval maps that position back to a chunk_index row by
    # position (embedding_v1.py) — so these two files must never be reordered
    # independently. Both are written here from `chunk_manifest` in the same order;
    # the index-build step re-checks only the row counts, not the order. Embedding
    # always processes a patient in one full pass, so the manifest order is fixed and
    # re-embedding the same sources reproduces a byte-for-byte identical result.
    _atomic_save_npy(emb_path, mat)
    df = pl.DataFrame(
        [{k: m[k] for k in _index_columns()} for m in chunk_manifest],
        schema=_CHUNK_INDEX_SCHEMA,
    )
    # The actual window used after reserving room for the special marker tokens,
    # recorded so the companion metadata file is self-explanatory: when the window
    # is omitted it defaults to the content budget, so the raw chunker identity
    # carries window=None — this field records the span that was really used. It is
    # for provenance only; the no-load cache comparison still uses the raw chunker
    # identity (the tokenizer_hash inside the encoder fingerprint invalidates the
    # cache when num_special_tokens shifts the default window).
    resolved_window = chunk_span_in_tokens(chunker)
    _write_chunk_index(
        idx_path=idx_path,
        idx_meta_path=idx_meta_path,
        df=df,
        run_id=run_id,
        patient_id=patient_id,
        code_lock_hash=code_lock_hash,
        encoder=encoder,
        source_hashes=source_hashes,
        chunker_identity=chunker_identity,
        chunker_resolved_window=resolved_window,
        embed_mode="full",
        chunks_reused=0,
        chunks_added=n,
    )

    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="embed"),
        from_state="running",
        to_state="completed",
        reason=(
            f"wrote {n} embeddings (dim={dim})" if n
            else "no embeddable text for this patient"
        ),
        step_context="embed",
        code_lock_hash=code_lock_hash,
    )
    log.info("embed_done", extra_={"chunks": n, "dim": dim})
    return {"patient_id": patient_id, "chunks": n, "dim": dim}

def _chunks_for_row(row: _Row, patient_id: str, chunker: _Chunker, encoder: Encoder) -> list[dict]:
    """Split one source row into chunk manifest entries; the chunk_id is built from
    source_stem (not the full filename) so it stays stable if the file is re-saved
    in a different format, e.g. csv → xlsx."""
    row_hash = _row_text_hash(row.text)
    return [
        {
            "chunk_id": f"{patient_id}:{row.source_stem}:{row.row_id}:{ch.chunk_idx}",
            "patient_id": patient_id,
            "source_file": row.source_file,
            "source_stem": row.source_stem,
            "row_id": row.row_id,
            "source_column": row.source_column,
            "source_text_sha256": row_hash,
            "chunk_idx": ch.chunk_idx,
            "char_start": ch.char_start,
            "char_end": ch.char_end,
            "token_count": ch.token_count,
            "parent_tokens": ch.parent_tokens,
            "num_chunks": 0,
            **{field_name: row.metadata.get(field_name) for field_name in STANDARD_METADATA_FIELDS},
            "_text": ch.text,
        }
        for ch in chunker.chunk(row.text, tokenizer=encoder)
    ]

def _fill_num_chunks(manifest: list[dict]) -> None:
    # num_chunks (how many chunks a row produced) is left at 0 in _chunks_for_row because the total isn't known until every chunk is built; it's filled in here.
    counts: dict[tuple[str, int, str], int] = {}
    for m in manifest:
        key = (m["source_stem"], m["row_id"], m["source_column"])
        counts[key] = counts.get(key, 0) + 1
    for m in manifest:
        m["num_chunks"] = counts[(m["source_stem"], m["row_id"], m["source_column"])]

def _encode_manifest(
    manifest: list[dict], encoder: Encoder, batch_size: int
) -> np.ndarray:
    _ = encoder.embed_batch([])  # force the model to load so we learn its vector length; dim stays 0 until the first call, even for an empty manifest.
    dim = int(getattr(encoder, "dim", 0) or 0)
    if dim == 0:
        raise RuntimeError("Encoder did not report a hidden_size after load")
    mat = np.zeros((len(manifest), dim), dtype=np.float32)
    cursor = 0
    for i in range(0, len(manifest), batch_size):
        batch_texts = [m["_text"] for m in manifest[i : i + batch_size]]
        vecs = encoder.embed_batch(batch_texts)
        if vecs.shape[0] != len(batch_texts):
            raise RuntimeError(
                f"Encoder returned {vecs.shape[0]} vectors for batch of {len(batch_texts)}"
            )
        mat[cursor : cursor + vecs.shape[0]] = vecs
        cursor += vecs.shape[0]
    return mat

def _atomic_save_npy(path: Path, mat: np.ndarray) -> None:
    """Write the .npy file so it lands fully or not at all (write to a temporary
    staging file, then rename into place). The staging name ends in .npy because
    np.save would otherwise append that extension itself."""
    staging = path.parent / f".staging-{path.name}"
    if staging.exists():
        staging.unlink()
    np.save(staging, mat, allow_pickle=False)
    staging.rename(path)

def _write_chunk_index(
    *,
    idx_path: Path,
    idx_meta_path: Path,
    df: pl.DataFrame,
    run_id: str,
    patient_id: str,
    code_lock_hash: str | None,
    encoder: Encoder,
    source_hashes: dict[str, str],
    chunker_identity: dict,
    chunker_resolved_window: int,
    embed_mode: str,
    chunks_reused: int,
    chunks_added: int,
) -> None:
    tmp = idx_path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression=_DEFAULT_PARQUET_COMPRESSION)
    tmp.rename(idx_path)

    payload = {
        "patient_id": patient_id,
        "row_count": df.height,
        "encoder": encoder.fingerprint(),
        "columns": _index_columns(),
        "parquet_content_hash": hash_file(idx_path),
        "source_parquet_hashes": source_hashes,
        "chunker": chunker_identity,
        "chunker_resolved_window": chunker_resolved_window,
        "embed_mode": embed_mode,
        "chunks_reused": chunks_reused,
        "chunks_added": chunks_added,
    }
    env = envelope_for(
        artifact_type="chunk_index",
        sensitivity="medium",
        stream="data",
        run_id=run_id,
        step="embed",
        patient_id=patient_id,
        payload=payload,
        code_lock_hash=code_lock_hash,
    )
    write_artifact(env, path=idx_meta_path)  # companion metadata file, written right next to the chunk index
