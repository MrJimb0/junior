"""Shared types for the retrieval step.

PatientChunkStore wraps everything a retriever needs for one patient — the chunk index,
the structured parquets, the embeddings matrix, and the HNSW index. All of
these load lazily so a BM25 or exact retriever never touches the vector files.

Retriever is the interface every retriever implementation satisfies. Candidate
is one retrieved chunk with its score and rank.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_bytes,
    hash_file,
)
from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    DEFAULT_COLUMN_ALIASES,
    find_column,
    metadata_value_as_text,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import structured_dir


def verify_hnsw_consistency(
    *,
    sidecar_meta: dict | None,
    current_emb_hash: str,
    n_index: int,
    n_emb: int,
    n_chunks: int,
) -> None:
    """Raise if a loaded hnsw index doesn't match the artifacts it indexes.

    Reads the sidecar we already write (``hnsw.bin.meta.json`` -> ``index_meta``):
    the index's ``embeddings_sha256`` must equal the embeddings.npy now on disk
    (else the index is stale — built from different vectors — and would mislabel
    every neighbor), and the index / embeddings / chunk_index row counts must
    agree. Pure: the caller does the I/O (hashing + counts)."""
    if not sidecar_meta:
        raise RuntimeError(
            "hnsw sidecar (hnsw.bin.meta.json) is missing or unreadable; re-run the "
            "index step for this patient."
        )
    stored = sidecar_meta.get("embeddings_sha256")
    if stored != current_emb_hash:
        raise RuntimeError(
            f"hnsw index is stale: it was built from embeddings {stored} but "
            f"embeddings.npy now hashes to {current_emb_hash}. Re-run the index step."
        )
    if not (n_index == n_emb == n_chunks):
        raise RuntimeError(
            f"hnsw/embeddings/chunk_index row-count mismatch: index {n_index}, "
            f"embeddings {n_emb}, chunk_index {n_chunks}. Re-run embed + index."
        )


def _strip_hash_prefix(value: str) -> str:
    """``sha256:abc`` -> ``abc`` so a hash never adds a stray ':' to a pointer."""
    return value.split(":", 1)[1] if ":" in value else value


def build_structured_evidence_pointer(
    *, patient_id: str, file_hash: str, row_hash: str, column: str
) -> str:
    """PHI-side provenance pointer for a value read directly from a structured
    parquet (no embedded chunk): ``patient:<id>:structured:<file_hash>:row:<row_hash>:
    column:<column>``. The file_hash pins the exact parquet bytes and the
    row_hash the exact row, so a direct-table value is as verifiable as a text
    chunk's evidence_chunk_id. Hashes are stored bare (``sha256:`` prefix stripped)
    so they carry no ':'; the column name must not contain ':'."""
    return (
        f"patient:{patient_id}"
        f":structured:{_strip_hash_prefix(file_hash)}"
        f":row:{_strip_hash_prefix(row_hash)}"
        f":column:{column}"
    )


def parse_structured_evidence_pointer(pointer: str) -> dict | None:
    """Inverse of ``build_structured_evidence_pointer``; None if not one.

    Splits on the fixed labels rather than by position, so a patient id is taken
    as everything between ``patient:`` and ``:structured:`` (ingest forbids ':' in
    patient ids and stems) and the column as everything after ``:column:``."""
    if not pointer.startswith("patient:"):
        return None
    rest = pointer[len("patient:"):]
    try:
        patient_id, rest = rest.split(":structured:", 1)
        file_hash, rest = rest.split(":row:", 1)
        row_hash, column = rest.split(":column:", 1)
    except ValueError:
        return None
    if not (patient_id and file_hash and row_hash and column):
        return None
    return {
        "patient_id": patient_id,
        "file_hash": file_hash,
        "row_hash": row_hash,
        "column": column,
    }


def render_table_row(row: dict[str, Any]) -> str:
    """Render one table row as ``col: value`` lines, skipping blank cells. The single
    source of this shape: an embedded ``:TABLE`` chunk (``_text_for_table``) and a
    builder-table evidence row must read identically in the evidence bundle."""
    parts: list[str] = []
    for key, value in row.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


@dataclass(frozen=True)
class Candidate:
    """One chunk returned by a retriever, with its rank and score.

    rank is 1-indexed (best = 1). score meaning depends on the retriever —
    cosine similarity for embedding, raw BM25 for keyword, RRF score for hybrid.
    retriever names which member of a hybrid found this chunk; None for single retrievers.
    """

    chunk_id: str
    rank: int
    score: float
    retriever: str | None = None


@dataclass
class PatientChunkStore:
    """Gives retrievers access to a patient's chunk index, source text, embeddings, and search index."""

    patient_root: Path
    chunk_index: pl.DataFrame = field(init=False)
    _structured: dict[str, pl.DataFrame] = field(default_factory=dict, init=False)
    _embeddings: np.ndarray | None = field(default=None, init=False)
    _hnsw: Any | None = field(default=None, init=False)
    # chunk_id -> chunk_index row, built once so text_for / row_for are O(1).
    # Callers loop over candidates, so a per-call .filter() would be quadratic.
    _row_by_chunk_id: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    # source-parquet stem -> original source filename (lazy, from ingest sidecar).
    _stem_source_file: dict[str, str | None] = field(default_factory=dict, init=False)
    # source-parquet stem -> {metadata field: this file's column} (lazy, from ingest
    # sidecar), so a :TABLE row's metadata is read from the columns ingest resolved.
    _stem_metadata_columns: dict[str, dict[str, str | None]] = field(
        default_factory=dict, init=False
    )
    # structured stems whose content hash matched the ingest sidecar this session,
    # so each parquet is verified at most once.
    _verified_structured_stems: set[str] = field(default_factory=set, init=False)
    # (source_file, row_id) pairs whose text matched the per-row hash embed
    # recorded, so each source row is verified at most once.
    _verified_rows: set[tuple[str, int]] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        idx_path = self.patient_root / "chunk_index.parquet"
        if not idx_path.is_file():
            raise FileNotFoundError(f"chunk_index.parquet missing at {idx_path}")
        self.chunk_index = pl.read_parquet(idx_path)
        self._row_by_chunk_id = {
            r["chunk_id"]: r for r in self.chunk_index.iter_rows(named=True)
        }

    def stored_encoder_fingerprint(self) -> dict[str, Any] | None:
        """Encoder fingerprint embed recorded in the chunk_index sidecar, or None if
        the sidecar does not carry one."""
        sidecar = self.patient_root / "chunk_index.parquet.meta.json"
        try:
            env = json.loads(sidecar.read_text(encoding="utf-8"))
            return env["payload"]["encoder"]
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def has_structured_source(self) -> bool:
        """True if the structured source parquets are present (a re-embed is possible)."""
        return (structured_dir(self.patient_root)).is_dir()

    def row_for(self, chunk_id: str) -> dict[str, Any] | None:
        """The chunk_index row for ``chunk_id`` as a dict, or None for ``:TABLE``
        pseudo-chunks and unknown ids. O(1).

        This is the chunk_index accessor, so it answers only for embedded chunks. Read
        chart metadata through ``metadata_for``, which answers for a table row too.
        """
        return self._row_by_chunk_id.get(chunk_id)

    def _structured_for(self, source_file: str) -> pl.DataFrame:
        # cache keyed by stem — the same key _text_for_table uses, so a table
        # read as a :TABLE row and as a chunk source loads only once.
        stem = Path(source_file).stem
        if stem in self._structured:
            return self._structured[stem]
        path = structured_dir(self.patient_root) / f"{stem}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"structured parquet missing: {path}")
        df = pl.read_parquet(path)
        self._structured[stem] = df
        return df

    def text_for(self, chunk_id: str) -> str:
        """resolve readable evidence text for a chunk_id.

        normal chunks slice the source column at stored offsets; ``:TABLE``
        pseudo-chunks (``<patient_id>:<stem>:<row_id>:TABLE``) render the row
        as ``col: value`` lines instead.
        """
        if chunk_id.endswith(":TABLE"):
            return self._text_for_table(chunk_id)

        r = self._row_by_chunk_id.get(chunk_id)
        if r is None:
            raise KeyError(chunk_id)
        df = self._structured_for(r["source_file"])
        if r["row_id"] >= df.height:
            raise IndexError(f"row_id {r['row_id']} out of range for {r['source_file']}")
        value = df[r["source_column"]][r["row_id"]]
        # non-string columns (numeric, bool) can't be sliced — return empty
        # evidence rather than crash the retriever
        if not isinstance(value, str):
            return ""
        # The chunk's char offsets point into the row text embed saw; if the parquet
        # changed since then they would silently slice the wrong text. Verify the
        # row against the per-row hash embed recorded (once per row), mirroring the
        # whole-file check :TABLE reads get.
        key = (r["source_file"], int(r["row_id"]))
        if key not in self._verified_rows:
            recorded = r.get("source_text_sha256")
            if recorded and hash_bytes(value.encode("utf-8")) != recorded:
                raise RuntimeError(
                    f"source row for {chunk_id!r} changed since embed (content-hash "
                    "mismatch): chunk offsets would slice the wrong text. Re-run "
                    "embed for this patient."
                )
            self._verified_rows.add(key)
        return value[r["char_start"] : r["char_end"]]

    @staticmethod
    def _parse_table_chunk_id(chunk_id: str) -> tuple[str, int] | None:
        """Parse ``{patient_id}:{source_stem}:{row_id}:TABLE`` from the RIGHT so a
        patient id containing ':' cannot shift the fields. Returns (stem, row_id)."""
        if not chunk_id.endswith(":TABLE"):
            return None
        base = chunk_id[: -len(":TABLE")]
        prefix, sep, row_str = base.rpartition(":")
        if not sep:
            return None
        stem = prefix.rpartition(":")[2]
        if not stem:
            return None
        try:
            return stem, int(row_str)
        except ValueError:
            return None

    def source_file_for_stem(self, stem: str) -> str | None:
        """Original source filename (with extension) for a structured-parquet stem,
        from the ingest sidecar — the authoritative source_file for :TABLE chunks
        (whose source may be .xlsx/.tsv/etc, not .csv)."""
        if stem not in self._stem_source_file:
            src: str | None = None
            sidecar = structured_dir(self.patient_root) / f"{stem}.parquet.meta.json"
            if sidecar.is_file():
                try:
                    env = json.loads(sidecar.read_text(encoding="utf-8"))
                    src = (env.get("payload") or {}).get("source_file")
                except Exception:
                    src = None
            self._stem_source_file[stem] = src
        return self._stem_source_file[stem]

    def metadata_columns_for_stem(self, stem: str) -> dict[str, str | None]:
        """Which column of ``structured/<stem>.parquet`` holds each chart metadata field.

        Reads the mapping ingest resolved against the file's real header and recorded in
        its sidecar, so a :TABLE row's metadata comes from the same columns as an
        embedded chunk's. Falls back to matching the default aliases against the
        parquet's own header when the sidecar carries no mapping (an older run, or a
        builder table ingest never saw) — the same fallback embed makes. Cached per stem.
        """
        if stem in self._stem_metadata_columns:
            return self._stem_metadata_columns[stem]
        recorded: dict[str, str | None] | None = None
        sidecar = structured_dir(self.patient_root) / f"{stem}.parquet.meta.json"
        if sidecar.is_file():
            try:
                env = json.loads(sidecar.read_text(encoding="utf-8"))
                recorded = (env.get("payload") or {}).get("metadata_columns") or None
            except (OSError, json.JSONDecodeError):
                recorded = None
        if recorded is None:
            try:
                columns = self._structured_for(stem).columns
            except FileNotFoundError:
                columns = []
            recorded = {
                field_name: find_column(aliases, columns)
                for field_name, aliases in DEFAULT_COLUMN_ALIASES.items()
            }
        self._stem_metadata_columns[stem] = recorded
        return recorded

    def metadata_for(self, chunk_id: str) -> dict[str, Any] | None:
        """The chart metadata a filter or a date sort reads for one chunk, or None when
        the id resolves to nothing.

        An embedded chunk answers from its chunk_index row. A ``:TABLE`` pseudo-chunk has
        no such row, so it answers from the structured parquet row itself through the
        column mapping ingest recorded — that is what lets a recipe filter table evidence
        by author and order it by document date on the same terms as free text. Both
        carry ``source_file``, so a caller never has to know which kind it holds.
        """
        row = self._row_by_chunk_id.get(chunk_id)
        if row is not None:
            return row
        resolved = self._table_row_for(chunk_id)
        if resolved is None:
            return None
        stem, table_row = resolved
        metadata: dict[str, Any] = {
            field_name: (
                metadata_value_as_text(table_row.get(column)) if column is not None else None
            )
            for field_name, column in self.metadata_columns_for_stem(stem).items()
        }
        metadata["source_file"] = self.source_file_for_stem(stem)
        return metadata

    def _table_row_for(self, chunk_id: str) -> tuple[str, dict[str, Any]] | None:
        """``(stem, row)`` for a ``:TABLE`` pseudo-chunk, or None when the id is not one,
        its parquet is gone, or the row is out of range.

        Verifies the parquet still hashes to what ingest recorded before reading it, so
        the text and the metadata resolved from a table row are held to one check.
        """
        parsed = self._parse_table_chunk_id(chunk_id)
        if parsed is None:
            return None
        stem, row_id = parsed
        path = structured_dir(self.patient_root) / f"{stem}.parquet"
        if not path.is_file():
            return None
        self._verify_structured_source_hash(stem, path)
        df = self._structured_for(stem)
        if row_id < 0 or row_id >= df.height:
            return None
        return stem, df.row(row_id, named=True)

    def _verify_structured_source_hash(self, stem: str, path: Path) -> None:
        """A :TABLE value is resolved straight from structured/<stem>.parquet, so the
        parquet must still be the one ingest indexed — verify its content hash
        against the ingest sidecar before reading. Cached per stem; a sidecar
        without parquet_content_hash can't be checked and is skipped."""
        if stem in self._verified_structured_stems:
            return
        sidecar = structured_dir(self.patient_root) / f"{stem}.parquet.meta.json"
        recorded = None
        if sidecar.is_file():
            try:
                env = json.loads(sidecar.read_text(encoding="utf-8"))
                recorded = (env.get("payload") or {}).get("parquet_content_hash")
            except (OSError, json.JSONDecodeError):
                recorded = None
        if recorded is not None and hash_file(path) != recorded:
            raise RuntimeError(
                f"structured parquet {stem!r} changed since ingest (content-hash "
                f"mismatch): a :TABLE value would resolve from the wrong bytes. "
                "Re-run ingest for this patient."
            )
        self._verified_structured_stems.add(stem)

    def _text_for_table(self, chunk_id: str) -> str:
        resolved = self._table_row_for(chunk_id)
        return render_table_row(resolved[1]) if resolved is not None else ""

    def embeddings(self) -> np.ndarray:
        """lazy-load and cache the per-patient embeddings.npy matrix."""
        if self._embeddings is None:
            path = self.patient_root / "embeddings.npy"
            if not path.is_file():
                raise FileNotFoundError(f"embeddings.npy missing: {path}")
            self._embeddings = np.load(path, allow_pickle=False)
        return self._embeddings

    def _hnsw_sidecar_meta(self) -> dict | None:
        """index_meta payload from hnsw.bin.meta.json, or None if legacy/unreadable."""
        try:
            env = json.loads(
                (self.patient_root / "hnsw.bin.meta.json").read_text(encoding="utf-8")
            )
            return env["payload"]["index_meta"]
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def hnsw(self, dim: int, space: str = "ip", ef_search: int = 80) -> Any:
        """lazy-load and cache the per-patient hnswlib index from hnsw.bin.

        Verifies the loaded index against the sidecar we wrote: the
        embeddings it was built from must still be the embeddings on disk, and the
        index / embeddings / chunk_index row counts must agree — otherwise a stale
        or partially rebuilt index would mislabel every neighbor."""
        if self._hnsw is None:
            import hnswlib

            path = self.patient_root / "hnsw.bin"
            if not path.is_file():
                raise FileNotFoundError(f"hnsw.bin missing: {path}")
            idx = hnswlib.Index(space=space, dim=dim)
            idx.load_index(str(path))
            idx.set_ef(ef_search)
            verify_hnsw_consistency(
                sidecar_meta=self._hnsw_sidecar_meta(),
                current_emb_hash=hash_file(self.patient_root / "embeddings.npy"),
                n_index=int(idx.get_current_count()),
                n_emb=int(self.embeddings().shape[0]),
                n_chunks=int(self.chunk_index.height),
            )
            self._hnsw = idx
        return self._hnsw


@dataclass(frozen=True)
class RetrieverInfo:
    """metadata stamped into retrieval.json under 'retriever'."""

    kind: str
    version: str
    config: dict[str, Any]


class Retriever(Protocol):
    """Interface every retriever must satisfy — query and report score normalization."""

    info: RetrieverInfo

    def query(
        self,
        corpus: PatientChunkStore,
        *,
        text: str,
        k: int,
    ) -> list[Candidate]:
        """return up to k candidates ranked best-first for `text` against `corpus`."""

    @property
    def score_normalization(self) -> str:
        """score-scale tag (e.g. `raw`, `rrf`, `l2_normalized_ip`) recorded on artifacts."""
