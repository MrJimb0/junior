"""Read a value straight from a patient's structured table — no model involved.

Some chart variables live in tidy tabular data (a demographics table, a problem
list, a medications table) rather than in free-text notes. For those, there is
no need to retrieve text and ask a language model: this step just opens the
patient's structured parquet, optionally keeps only the rows and columns the
recipe asks for, and hands those rows back.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import polars as pl

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.nonembedded_structured_parquet_table_looker_uper.direct_parquet_retriever_v1 import (
    _apply_filter,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_type_lookup import (
    register_step,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
    hash_json,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import structured_dir
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import build_structured_evidence_pointer

_log = get_logger("extract")


def _data_columns_from_sidecar(patient_root: Path, stem: str) -> dict[str, str]:
    """``{our name: this site's column}`` for one table, as ingest resolved it. Empty
    when the sidecar is absent or names none — the caller then uses the literal name."""
    sidecar = structured_dir(patient_root) / f"{stem}.parquet.meta.json"
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8")).get("payload") or {}
    except (json.JSONDecodeError, OSError):
        return {}
    return dict(payload.get("data_columns") or {})



@register_step("direct_parquet")
class DirectParquetHandler:
    """Read rows from a patient's table, optionally filtering rows and keeping only some columns.

    When a step's rows ARE the final answer for a chart variable (for example
    race_ethnicity, read straight from a demographics table), the recipe sets
    ``emit_evidence_pointers: true``. With that flag on, the step also produces,
    for every row and every kept column, an "evidence pointer" — a short string
    that records exactly which table cell the value came from, so the answer is
    traceable back to its source. The pointer looks like
    ``patient:<id>:structured:<file_hash>:row:<row_hash>:column:<col>`` (here a
    "hash" is a short fingerprint computed from the file or row contents, so any
    change to the data is detectable). A later python step copies the pointer for
    the column it used onto an ``*_evidence_chunk_id`` field, so a value pulled
    from a table is documented to its source just as carefully as a value quoted
    from a clinical note.

    The flag is off by default: when a step's rows are only an intermediate
    input to a later language-model step (for example feeding number_of_primaries
    a problem list to reason over), no cell-level pointer is needed and nothing
    changes.
    """

    kind = "direct_parquet"

    def execute(self, ctx: StepContext) -> StepResult:
        cfg = ctx.step.config
        table = cfg.get("table")
        if not table:
            raise ValueError(f"step {ctx.step.id}: 'table' required")
        stem = Path(table).stem
        path = structured_dir(ctx.corpus.patient_root) / f"{stem}.parquet"
        emit_pointers = bool(cfg.get("emit_evidence_pointers"))

        receipt: dict[str, Any] = {"table": table, "columns_requested": cfg.get("columns")}

        # not every patient has every table — record and return empty rather than crash.
        if not path.is_file():
            receipt["missing"] = True
            data: dict[str, Any] = {"rows": [], "count": 0}
            if emit_pointers:
                data["evidence_pointers"] = []
            return StepResult(data=data, receipt_payload=receipt)

        t0 = time.perf_counter()
        df = pl.read_parquet(path)
        df = _apply_filter(df, cfg.get("filter"))
        cols = cfg.get("columns")
        if cols:
            # A recipe asks for OUR name for a column ("race", "result_value"); this
            # site's export calls it whatever it calls it. Ingest recorded the mapping
            # beside the table, so translate through it, then fall back to the literal
            # name for a column the map does not mention.
            site_columns = _data_columns_from_sidecar(ctx.corpus.patient_root, stem)
            resolved = {c: site_columns.get(c, c) for c in cols}
            # Keep each column ONCE: two of our names can legitimately resolve to one
            # column at a site (or a recipe can list the same column twice), and
            # selecting it twice is an error, not a duplicate value.
            keep = list(dict.fromkeys(
                column for column in resolved.values() if column in df.columns
            ))
            unresolved = [
                requested for requested, column in resolved.items()
                if column not in df.columns
            ]
            receipt["columns_kept"] = keep
            if unresolved:
                # This used to be a silent drop, which reads downstream as "the chart
                # does not say" rather than "we asked for a column that is not there".
                receipt["columns_not_found"] = unresolved
                _log.warning("direct_parquet_column_not_found", extra_={
                    "table": table,
                    "requested": unresolved,
                    "available": df.columns,
                    "hint": "name it in this site's column map under data_columns, or "
                            "fix the recipe's columns list",
                })
            df = df.select(keep)
        rows = df.to_dicts()
        elapsed = time.perf_counter() - t0

        data = {"rows": rows, "count": len(rows)}
        if emit_pointers:
            # file_hash is a fingerprint of the exact table file; row_hash a
            # fingerprint of the exact row — together they nail down which cell a
            # value came from. One pointer is built per (row, kept column) so the
            # later python step can cite the specific cell. The fingerprints are
            # written as sha256:<hex>; the pointer helper drops the "sha256:"
            # prefix and stores the bare hex value.
            file_hash = hash_file(path)
            pointers = []
            for row in rows:
                # Hash the row once per row, not once per (row, column): the row is
                # identical across its columns, so hashing once here is R serializations of
                # a possibly-wide row rather than R x C.
                row_hash = hash_json(row)
                pointers.append({
                    column: build_structured_evidence_pointer(
                        patient_id=ctx.patient_id,
                        file_hash=file_hash,
                        row_hash=row_hash,
                        column=column,
                    )
                    for column in row
                })
            data["evidence_pointers"] = pointers

        receipt.update({
            "row_count": len(rows),
            "elapsed_s": round(elapsed, 6),
        })
        return StepResult(data=data, receipt_payload=receipt)
