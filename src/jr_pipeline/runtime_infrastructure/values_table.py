"""A run's answers as one table per recipe, in the shape the original Junior used.

Values live in ``patients/<patient>/extract/<variable>/result.json`` — one nested
envelope per patient per variable. Right for an audit, wrong for the question asked
after every run, which is "what did it find". Answering it meant opening the browser
workbench one patient at a time or walking JSON by hand.

The shape here is taken from the run outputs of the pre-rewrite pipeline, which were
used in anger for the 100-patient experiment:

    Run_Outputs/exp5/20260311_120527/gpt5/
        date_of_birth_results_20260311_120527.xlsx
        pathology_stage_results_20260311_120527.xlsx
        ...

One file per variable, every file stamped with the run, all of them in one folder. Each
sheet was one row per patient with columns ``var_name, ok, <every field the recipe
produced>, patient_id`` — including the certainty scores, the rationale, the evidence
chunk ids and the verbatim quotes.

That last part is the correction worth recording: an earlier version of this module
dropped those as "the model's commentary". They are not commentary. ``pathology_er_status``
beside ``pathology_er_status_verbatim`` beside the chunk id it came from is what makes a
row checkable, which is the entire claim this pipeline makes. A table of bare values is
the one thing a clinical extraction must not hand back.

WHERE THE COLUMNS COME FROM. The recipe's sealed output schema, not the answers. See
``sealed_recipe_output_columns``: a header derived from what the model returned is a
different header for every cohort, and two such files do not stack. A recipe whose
schema the run no longer holds falls back to the returned keys, sorted, and says so in
the run_metadata table — a header nobody can vouch for must not look like one that is.

WHAT IS NOT ON THE ROW. Five fields describe the machinery rather than the patient and
are identical on every row of a file: recipe_version, recipe_hash, schema_hash, model
and code. They live once in the run_metadata table, joined on run_id + var_name. What
identifies a row when tables are pooled — patient_id, var_name, run_id, site_id,
study_id, patient_surrogate — stays on the row, because a pooled table whose rows
cannot say where they came from is not poolable at all.

FORMAT. A workbook, as the pre-rewrite pipeline wrote. These tables carry the cited
passage beside the value, and a passage is multi-line clinical text: as CSV that is a
correctly-quoted field spanning forty physical lines, so a two-patient file read as a
shredded one in every text editor, pager and diff. The old outputs never had that
problem because they were never CSV, and because they carried a short quoted span
rather than the whole passage. A spreadsheet cell holds the passage without comment.

Three other things come free with it, each of which was a workaround before: no
byte-order mark is needed for Excel to decode the chart text correctly; TRUE/FALSE and
the counts are real booleans and numbers rather than strings every reader has to coerce
back; and a passage beginning with '=' cannot be mistaken for a formula, which in CSV
is a live injection into whatever opens it.

The one-file download the workbench offers is still CSV — it feeds scripts, it is asked
for by name, and it is not the file a clinician double-clicks.

THIS OUTPUT IS PHI. Every cell is read out of a patient's chart, so it lands under the
run's own CONTAINS_PHI folder and is deliberately not offered as a shareable export.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.sealed_recipe_output_columns import (
    RecipeColumns,
    columns_from_schema,
    sealed_schema_for,
    sealed_schema_matches,
)

# Columns that identify the row rather than answer anything, kept in front so a reader
# can see whose row it is and whether to trust it before the fields begin.
# `ok` is a verdict on the MACHINERY — every step ran, the answer matched its schema,
# every value carried a source span. It is not a verdict on whether anything was found,
# and reading it as one inverts the truth: in a real run `stage` came back ok=TRUE with
# every one of its eighteen fields empty, and ok=FALSE on the only patient who had a
# stage. Filtering on `ok` selected exactly the wrong patient and said nothing about it.
#
# Rather than redefine a recorded field, the question it is mistaken for gets its own
# column, next to it, where anyone about to filter will see both.
_LEADING_COLUMNS = ("patient_id", "var_name", "ok", "any_value_found")

# What lets one site's table sit under another's. Without them these are site-local
# files: two institutions' rows for patient "STSS10c40439" are indistinguishable, and
# nothing here joins to the NO_PHI exhaust, which is the entire point of running two
# streams. patient_surrogate is minted in the same domain extraction_outcome uses, so
# the join needs no second derivation and never needs the HMAC secret at read time.
_LINKAGE_COLUMNS = ("site_id", "study_id", "patient_surrogate", "run_id")

# What this pipeline records that the original did not, and the reason its output is
# worth more than a table of values: whether the row can be believed. The original
# carried nothing here, so an ok=False row was a dead end — you could see it had failed
# and not why.
#
# These stay on the ROW and do not move to the run_metadata table: every one of them is a
# statement about this patient's extraction. Checked on the reference run — warnings,
# why_it_failed, the rejected citations and the seconds all differ between the two
# patients of a single file.
#
# They sit AFTER the answers, because the answers are what a reader came for. Names are
# plain rather than the internal field names, since this file is read by people who will
# never open the code.
_TRUST_COLUMNS = (
    ("errors", "why_it_failed"),
    ("warnings", "warnings"),
    # NOT "citations the model invented", which is what this column used to be called
    # and is not what the check does. It compares a cited chunk id against the chunks
    # put in front of THIS step. In a reference run it fired on a chunk from a
    # real clinic note that had been shown to another
    # step of the same run, cited approvingly there for date_of_diagnosis, supporting a
    # value that is correct. The model reached outside its window; it did not invent a
    # citation, and a column that says it did teaches a reviewer to distrust the wrong
    # thing.
    ("provenance_rejected", "cited_a_chunk_not_shown_to_this_step"),
    ("unprovenanced_value_paths", "values_with_no_evidence"),
    ("steps_that_errored", "steps_that_errored"),
    ("validation_rules_that_did_not_run", "clinical_rules_not_run"),
    ("total_elapsed_s", "seconds"),
)

# What happened to the answer on its way through the recipe — a different question from
# whether the answer is any good, and one a reader cannot otherwise ask of a row. A
# recipe's steps run in order and the LAST answer wins, so a step that completes while
# returning nothing overwrites one that found something, and the row is then
# indistinguishable from a chart that is silent. extract.py records the trace and says
# what happened on the run that prompted it.
#
# Not a judgement about whether what was found was right: only that it was found and is
# not in the answer, which is a fact about the machinery.
_STEP_TRACE_COLUMNS = (
    "answer_came_from_step",
    "steps_that_returned_nothing",
    "earlier_steps_that_found_more",
)

# Anything the model returned that the recipe's schema never declared. One column, not
# one per key: see sealed_recipe_output_columns for why an undeclared key must not widen
# the header. Always present, usually empty.
_UNDECLARED_COLUMN = "fields_not_in_the_recipe_schema"

# Per-run, per-recipe facts, stated once in the run_metadata table instead of retyped
# on every row of every file. `model` is this pipeline's answer to the original's
# llm_key; it is a hash of the resolved provider config, so it is looked up rather than
# grouped by, which is why it reads better here than repeated 42 times in a file.
_RUN_METADATA_COLUMNS = (
    "run_id", "var_name", "recipe_version", "recipe_hash", "schema_hash",
    "model", "code", "site_id", "study_id", "columns_came_from",
    "n_rows", "n_patients",
)


def _as_cell(value: Any) -> Any:
    """One value as it should appear in the file.

    Python's True/False are written verbatim by csv.writer, and R's read.csv gives you
    a character column: `sum(d$ok)` errors, `subset(d, ok)` errors. The pipeline this
    replaced stored a real boolean in xlsx and did not have the problem. TRUE/FALSE
    reads as logical in R and is still truthy-parseable everywhere else."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return value


def _flatten_for_a_cell(value: Any) -> Any:
    """A list or dict column rendered so a spreadsheet shows something readable."""
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return "; ".join(json.dumps(item, sort_keys=True) for item in value)
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _rejected_citations(payload: dict) -> str:
    """The rejected pointers, each said with the value it was meant to support.

    Rendered as ``<path>=<chunk id>``. The path was being dropped, which left a reader
    holding a chunk id and a row of a dozen values with no way to tell which one the
    complaint was about. Neither part contains '=', so the cell still splits."""
    entries = payload.get("provenance_rejected") or []
    if not isinstance(entries, list):
        return _flatten_for_a_cell(entries)
    return "; ".join(
        f"{entry.get('path') or '?'}={entry.get('chunk_id') or '?'}"
        if isinstance(entry, dict) else str(entry)
        for entry in entries
    )


def _rejected_by_column(payload: dict, row_axis: str | None) -> dict[tuple[str, int | None], str]:
    """Which column had its pointer nulled, and what it had cited.

    The gate records ``data.stage_at_diagnosis.evidence_chunk_id``; the table's column
    for that is ``stage_at_diagnosis.evidence_chunk_id``. For a row-mapped recipe the
    gate says ``data.rows[3].evidence_chunk_id`` and the column is ``evidence_chunk_id``
    on row 3, so the row index is carried alongside."""
    rejected: dict[tuple[str, int | None], str] = {}
    for entry in payload.get("provenance_rejected") or []:
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        path = str(entry["path"]).removeprefix("data.")
        index: int | None = None
        if row_axis and path.startswith(f"{row_axis}["):
            head, _, tail = path.partition("]")
            try:
                index = int(head[len(row_axis) + 1:])
            except ValueError:
                continue
            path = tail.removeprefix(".")
        rejected[(path, index)] = str(entry.get("chunk_id") or "")
    return rejected


def _trust_columns(payload: dict) -> dict[str, Any]:
    """The companion columns: can this row be believed."""
    columns: dict[str, Any] = {}
    for field, column in _TRUST_COLUMNS:
        if field == "provenance_rejected":
            columns[column] = _rejected_citations(payload)
        else:
            columns[column] = _flatten_for_a_cell(payload.get(field))
    return {**columns, **_step_trace_columns(payload)}


def _step_trace_columns(payload: dict) -> dict[str, Any]:
    """Which step the answer came from, and what the other steps found.

    ``earlier_steps_that_found_more`` is the one worth reading: a step that ran before
    the answering step and found more than the answer carries. It is a plain comparison
    of counts, not an opinion about which was right — the recipe's answer stands, and
    the row says what it cost."""
    found: dict[str, int] = payload.get("content_found_by_step") or {}
    answered_by = payload.get("answer_came_from_step")
    in_the_answer = found.get(str(answered_by), 0) if answered_by else 0

    nothing = [step for step, n in found.items() if not n]
    # Only when the answer carries NOTHING. A later step condensing what an earlier one
    # gathered is ordinary work, not a loss: race_ethnicity's harmonize turns seven raw
    # demographics fields into two canonical ones, and counting leaves called that a
    # step finding more than the answer holds. The case worth a reader's attention is
    # the one this column was built for — the row looks like a silent chart and an
    # earlier step found something.
    earlier = list(found)[: list(found).index(str(answered_by))] if answered_by in found else []
    outfound = ([f"{step} ({found[step]} items)" for step in earlier if found[step]]
                if not in_the_answer else [])
    return {
        "answer_came_from_step": answered_by,
        "steps_that_returned_nothing": "; ".join(nothing),
        "earlier_steps_that_found_more": "; ".join(outfound),
    }


def _passages_shown_to(patient_dir: Path) -> dict[str, dict[str, Any]]:
    """chunk id -> the passage the model was actually shown, for one patient.

    The row carries a chunk id, which is a receipt and not evidence. Following it meant
    opening prepared_evidence_text/<variable>/<step>/evidence_blocks.json — and the row
    does not name the step, so you grep every one — or, when the cited chunk was not in
    that step's window, two parquet reads and an undocumented positional join to slice
    the text out by character offset. Eight steps for one value.

    The pipeline cannot know whether a value is right; there is no ground truth here.
    What it can do is put the passage next to the value so the decision is a person's
    and takes five seconds."""
    passages: dict[str, dict[str, Any]] = {}
    for blocks_file in sorted(patient_dir.glob("prepared_evidence_text/*/*/evidence_blocks.json")):
        try:
            blocks = json.loads(blocks_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for block in blocks if isinstance(blocks, list) else []:
            chunk_id = block.get("chunk_id")
            # First writer wins: the same chunk shown to two steps is the same text, and
            # the earlier step is the one that found it.
            if chunk_id and chunk_id not in passages:
                passages[str(chunk_id)] = block
    return passages


def _describe_source(block: dict[str, Any]) -> str:
    """Where a passage came from, in one cell: file, record, date, kind.

    A diagnosis date lifted from a note written a decade later is a thing a reviewer
    should be able to see without opening anything."""
    parts = [str(block.get(key)) for key in ("source_file", "document_date", "doc_type")
             if block.get(key) not in (None, "", "None")]
    record = block.get("record_index")
    if record not in (None, "", "None"):
        parts.insert(1, f"record {record}")
    # ASCII, deliberately. A middle dot is UTF-8, and a CSV opened by double-click in
    # Excel on macOS is decoded as the legacy system encoding, which turns it into "¬∑"
    # in every source cell. Chart text carries its own non-ASCII (bullets, degree signs,
    # ® and © are all in these reports) and that has to survive as written — but there
    # is no reason for the separator to add to it.
    return " | ".join(parts)


def _is_evidence_pointer(column: str) -> bool:
    return "evidence_chunk_id" in column.rsplit(".", 1)[-1]


def _evidence_base(column: str) -> str:
    """The name a pointer's passage columns hang off.

    Only the singular suffix is stripped: breast_receptors declares BOTH
    evidence_chunk_id and evidence_chunk_ids at the same level, so stripping the plural
    too would give them one shared pair of columns."""
    return column[: -len("_chunk_id")] if column.endswith("_chunk_id") else column


def _evidence_for(
    value: Any, passages: dict[str, dict[str, Any]], rejected_chunk: str | None
) -> tuple[str, str]:
    """The passage text and source behind one pointer cell.

    A pointer the provenance gate nulled used to leave the value with no evidence
    columns at all — stage_at_diagnosis.overall_group = I sat in the reference run with
    nothing beside it, which reads as "nobody asked" rather than "the citation was
    thrown out". Now it says which."""
    if rejected_chunk and not value:
        return f"(citation rejected: {rejected_chunk} was not shown to this step)", ""
    if not isinstance(value, str) or not value:
        return "", ""
    # A plural pointer arrives as the joined list; every id is resolved rather than the
    # whole string being looked up, which never matched and so claimed every list of
    # citations had never been shown.
    chunk_ids = [part.strip() for part in value.split(";") if part.strip()]
    blocks = [(chunk_id, passages.get(chunk_id)) for chunk_id in chunk_ids]
    texts = [
        # Says which of the two it is. A pointer to a passage that was never shown
        # is a different problem from one whose text we simply did not keep.
        block.get("text") if block else "(passage not among those shown to this step)"
        for _, block in blocks
    ]
    sources = [_describe_source(block) for _, block in blocks if block]
    return " ;; ".join(str(text) for text in texts), " ;; ".join(sources)


def _any_value_found(flat: dict[str, Any]) -> bool:
    """Whether this row actually carries a finding, as opposed to having run cleanly.

    Judged the same way step 8 decides what needs provenance: a rationale, a certainty
    score, an evidence pointer and a null placeholder are not findings."""
    from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
        _is_value_bearing_leaf,
    )

    return any(
        # An empty cell is not a finding. A list flattens to a joined string, so an
        # empty list arrives here as "" — and step 8's rule, which is written against
        # raw data where an empty list is a list, counts that as a value. A recipe
        # answering with an empty list would have read any_value_found TRUE over a row
        # with nothing in it, which inverts the one column that exists to stop exactly
        # that misreading.
        value != ""
        and _is_value_bearing_leaf(column.rsplit(".", 1)[-1], value)
        for column, value in flat.items()
    )


def _flatten(node: Any, prefix: str = "", stop_at: frozenset[str] = frozenset()) -> dict[str, Any]:
    """A nested result as flat ``a.b.c`` columns. Lists are left for the caller.

    ``stop_at`` names columns the schema declares as leaves. A recipe may declare a
    field an object without saying what is in it — stage's pretreatment_path_stage is
    ``{"type": "object"}`` and nothing more — and that field gets one column of its own.
    Descending into it anyway produced dotted keys the header has no place for, so the
    declared column sat empty while its contents went to the undeclared bucket, which
    says the recipe never named the field. It named it; it just did not describe it."""
    flat: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{prefix}.{key}" if prefix else key
            if here in stop_at:
                flat[here] = _flatten_for_a_cell(value)
            elif isinstance(value, dict):
                flat.update(_flatten(value, here, stop_at))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                continue  # a list of records is its own ROWS, handled by the caller
            elif isinstance(value, list):
                # A list of scalars is a value, not a row axis. Skipping it dropped
                # whole columns of evidence pointers — the one thing this file exists
                # to carry.
                flat[here] = _flatten_for_a_cell(value)
            else:
                flat[here] = value
    return flat


def _record_list_fields(data: dict) -> list[str]:
    """Every field holding a list of records — the candidates for the row axis."""
    return [key for key, value in (data or {}).items()
            if isinstance(value, list) and value and isinstance(value[0], dict)]


def _data_root_of(run_root: Path) -> Path | None:
    """The data root a run sits under, or None if this is not a run in a project tree.

    ``<data root>/CONTAINS_PHI/pipeline_run_receipts/<run id>`` is the only shape the
    surrogate secret can be found from; a run folder handed over on its own has no
    secret to mint with, and an empty linkage column is better than a wrong one."""
    run_root = Path(run_root)
    return run_root.parents[2] if run_root.parent.name == "pipeline_run_receipts" else None


def _patient_surrogates(run_root: Path, patient_ids: list[str]) -> dict[str, str]:
    """The NO_PHI surrogate for each patient, in the domain extraction_outcome uses.

    Minted the same way rather than looked up: nothing on the PHI side stores the
    mapping, and re-deriving it here with the same (scope, domain, field) means a
    values table joins straight onto the exhaust without anyone handling the HMAC."""
    data_root = _data_root_of(run_root)
    if data_root is None:
        return {}
    from jr_pipeline.runtime_infrastructure.exhaust.secret_lifecycle import run_secret
    from jr_pipeline.runtime_infrastructure.exhaust.surrogates import (
        LinkageScope,
        make_surrogate,
    )

    try:
        secret = run_secret(Path(run_root).name, data_root)
    except Exception:  # noqa: BLE001 — fail-closed secret layer; no key, no linkage
        return {}
    return {
        patient_id: make_surrogate(
            secret,
            scope=LinkageScope.RUN.value,
            record_type="extraction_outcome",
            output_field="patient_surrogate",
            raw=patient_id,
        )
        for patient_id in patient_ids
    }


class RecipeTable:
    """One recipe's answers: the rows, the header they are written under, and the
    per-run facts that were lifted off the rows into the run_metadata table."""

    def __init__(self, variable: str) -> None:
        self.variable = variable
        self.rows: list[dict[str, Any]] = []
        self.columns: tuple[str, ...] = ()
        self.schema_columns: RecipeColumns | None = None
        self.provenance: dict[str, Any] = {}
        self.columns_came_from = "what_the_model_returned"

    def __len__(self) -> int:
        return len(self.rows)


def _schema_for(run_root: Path, variable: str, payload: dict, produced_by: dict) -> tuple[
    RecipeColumns | None, str, str
]:
    """The recipe's declared columns, how they were found, and the schema's hash.

    A bundle that no longer holds the schema the result names is not a bundle we can
    derive a promise from — ``junior verify`` fails such a run for the same reason —
    so the header falls back to the returned keys and the table says which it got."""
    recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
    version = str((recipe or {}).get("version") or "")
    expected = produced_by.get("schema_hash")
    if not version:
        return None, "what_the_model_returned", str(expected or "")
    schema, path = sealed_schema_for(run_root, variable, version)
    if schema is None or path is None:
        return None, "what_the_model_returned", str(expected or "")
    if not sealed_schema_matches(path, expected):
        return None, "a_schema_the_run_no_longer_matches", str(expected or "")
    return columns_from_schema(schema), "the_recipes_sealed_output_schema", str(expected or "")


def read_run_values(run_root: Path) -> dict[str, RecipeTable]:
    """Every result in a run, grouped by variable, one row per patient — or per record
    for a recipe that maps over a table.

    Reads what is on disk rather than re-deriving anything, so a half-finished run
    reports what it has. A variable that FAILED is present with ``ok`` false: dropping it
    would make a broken extraction look like a chart with nothing in it, which is the
    distinction the whole pipeline exists to preserve."""
    run_root = Path(run_root)
    tables: dict[str, RecipeTable] = {}
    passages_by_patient: dict[str, dict[str, Any]] = {}
    result_files = sorted(run_root.glob("patients/*/extract/*/result.json"))
    surrogates = _patient_surrogates(
        run_root, sorted({path.parts[-4] for path in result_files})
    )
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        site_id,
        study_id,
    )

    linkage_site, linkage_study = site_id(), study_id()

    for result_file in result_files:
        patient_id, variable = result_file.parts[-4], result_file.parts[-2]
        # Read once per patient, not once per variable: every variable of a patient
        # cites into the same set of passages.
        if patient_id not in passages_by_patient:
            passages_by_patient[patient_id] = _passages_shown_to(result_file.parents[2])
        passages = passages_by_patient[patient_id]
        try:
            envelope = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload = envelope.get("payload") or {}
        produced_by = envelope.get("produced_by") or {}
        data = payload.get("data") or {}

        table = tables.setdefault(variable, RecipeTable(variable))
        if not table.provenance:
            schema_columns, came_from, schema_hash = _schema_for(
                run_root, variable, payload, produced_by
            )
            table.schema_columns = schema_columns
            table.columns_came_from = came_from
            recipe = payload.get("recipe") if isinstance(payload.get("recipe"), dict) else {}
            table.provenance = {
                "run_id": produced_by.get("run_id") or run_root.name,
                "var_name": variable,
                "recipe_version": (recipe or {}).get("version"),
                "recipe_hash": produced_by.get("recipe_hash"),
                "schema_hash": schema_hash,
                "model": produced_by.get("provider_config_hash"),
                "code": produced_by.get("code_lock_hash"),
                "site_id": linkage_site,
                "study_id": linkage_study,
                "columns_came_from": came_from,
            }

        common = {
            "patient_id": patient_id, "var_name": variable, "ok": payload.get("ok"),
            "site_id": linkage_site, "study_id": linkage_study,
            "patient_surrogate": surrogates.get(patient_id, ""),
            "run_id": produced_by.get("run_id") or run_root.name,
        }
        # Appended to every row, after the answers. A row-mapped recipe repeats them on
        # each of its rows, which is right: each row is judged and traced the same way.
        trust = {**_trust_columns(payload), "extracted_at": produced_by.get("created_at")}

        declared = table.schema_columns
        # Columns the schema declares as leaves, so a field it names without describing
        # keeps its own cell instead of being split into keys the header cannot hold.
        stop_at = frozenset(declared.patient_level) if declared else frozenset()
        row_axis = declared.row_axis_field if declared else None
        candidates = _record_list_fields(data)
        if row_axis is None and len(candidates) == 1:
            row_axis = candidates[0]
        rejected = _rejected_by_column(payload, row_axis)

        if len(candidates) > 1 and (declared is None or declared.row_axis_field is None):
            # Two lists of records and no way to know which is the row axis. This used
            # to take whichever sorted first and silently discard the other; a recipe
            # that grew a second list would have every row change meaning with no error.
            # One row per patient, every list rendered into a cell, and the ambiguity
            # named where a reader will see it.
            flat = _flatten(data, stop_at=stop_at)
            table.rows.append({
                **common, "any_value_found": _any_value_found(flat), **flat,
                "row_axis_ambiguous": "; ".join(sorted(candidates)),
                **{field: _flatten_for_a_cell(data[field]) for field in candidates},
                **trust,
            })
            continue
        if row_axis is None or not isinstance(data.get(row_axis), list):
            flat = _flatten(data, stop_at=stop_at)
            row = {**common, "any_value_found": _any_value_found(flat), **flat}
            table.rows.append({**row, **_evidence_columns(row, passages, rejected, None), **trust})
            continue
        # A recipe that maps over a table answers with a list. Each record is its own
        # row, carrying the patient-level fields alongside so the row stands alone.
        alongside = _flatten({k: v for k, v in data.items() if k != row_axis}, stop_at=stop_at)
        for index, record in enumerate(data[row_axis]):
            flat = _flatten(record, stop_at=frozenset(declared.per_row) if declared else frozenset())
            row = {**common, "any_value_found": _any_value_found(flat),
                   "row": index, **alongside, **flat}
            table.rows.append(
                {**row, **_evidence_columns(row, passages, rejected, index), **trust}
            )
        if not data[row_axis]:
            row = {**common, "any_value_found": False, **alongside}
            table.rows.append({**row, **_evidence_columns(row, passages, rejected, None), **trust})

    for table in tables.values():
        table.columns = _header_for(table)
    return tables


def _evidence_columns(
    row: dict[str, Any],
    passages: dict[str, dict[str, Any]],
    rejected: dict[tuple[str, int | None], str],
    row_index: int | None,
) -> dict[str, Any]:
    """For every evidence pointer in a row, the passage it points at.

    Named off the pointer it belongs to, so dob_evidence_chunk_id gets dob_evidence_text
    beside it and a reader never has to work out which value a quote supports."""
    found: dict[str, Any] = {}
    pointers = {column for column in row if _is_evidence_pointer(column)}
    # A pointer the gate nulled is no longer in the row at all, so the rejections are
    # walked too — otherwise the value it supported gets no evidence column and reads
    # as one nobody asked about.
    pointers |= {column for (column, index) in rejected if index == row_index}
    for column in pointers:
        base = _evidence_base(column)
        text, source = _evidence_for(
            row.get(column), passages, rejected.get((column, row_index))
        )
        found[f"{base}_text"] = text
        found[f"{base}_source"] = source
    return found


def _header_for(table: RecipeTable) -> tuple[str, ...]:
    """The fixed column order this recipe's rows are written under.

    From the sealed schema when the run still holds it, so the same recipe gives the
    same header over any cohort and two files stack. Without one, the returned keys in
    sorted order — still stable within a file, still not a promise, which is why
    the run_metadata table records which of the two a reader is holding."""
    header: list[str] = list(_LEADING_COLUMNS)
    if any("row" in row for row in table.rows):
        header.append("row")
    header += _LINKAGE_COLUMNS

    declared = table.schema_columns
    if declared is not None:
        answers: list[str] = [*declared.patient_level, *declared.per_row]
    else:
        fixed = set(header) | {name for _, name in _TRUST_COLUMNS} | set(_STEP_TRACE_COLUMNS) | {
            "extracted_at", "row_axis_ambiguous", _UNDECLARED_COLUMN
        }
        answers = sorted(
            {column for row in table.rows for column in row
             if column not in fixed and not column.endswith(("_text", "_source"))}
        )
    for column in answers:
        header.append(column)
        if _is_evidence_pointer(column):
            base = _evidence_base(column)
            header += [f"{base}_text", f"{base}_source"]

    if any("row_axis_ambiguous" in row for row in table.rows):
        header.append("row_axis_ambiguous")
    header.append(_UNDECLARED_COLUMN)
    header += [name for _, name in _TRUST_COLUMNS]
    header += list(_STEP_TRACE_COLUMNS)
    header.append("extracted_at")
    # dict.fromkeys rather than a set: a recipe declaring a field twice through two
    # oneOf branches must not get two columns, and the order has to survive.
    return tuple(dict.fromkeys(header))


def _declared_parents(columns: tuple[str, ...]) -> set[str]:
    """Every dotted path that has columns UNDER it: `clinical_tnm` given `clinical_tnm.cT`.

    A field declared nullable-object has no column of its own (see
    sealed_recipe_output_columns._without_null), so answering it with null leaves a bare
    key in the flattened row with nowhere to go. It is not an undeclared field: the
    recipe declares it, precisely as nullable, and its emptiness is already said by its
    empty sub-columns."""
    parents: set[str] = set()
    for column in columns:
        parts = column.split(".")
        for depth in range(1, len(parts)):
            parents.add(".".join(parts[:depth]))
    return parents


def _laid_out(table: RecipeTable) -> list[dict[str, Any]]:
    """Each row keyed by the header, with whatever the header has no place for gathered
    into the undeclared column.

    Shared by both renderers rather than done inside one: the workbook is what a run
    writes and the CSV is what the workbench hands back, and a rule applied in only one
    of them is a rule that holds for only one of them."""
    known = set(table.columns)
    nullable_parents = _declared_parents(table.columns)
    laid_out: list[dict[str, Any]] = []
    for row in table.rows:
        undeclared = {
            column: value for column, value in row.items()
            if column not in known
            # A non-null value under a declared parent IS worth reporting: the recipe
            # promised an object there and got something else.
            and not (value is None and column in nullable_parents)
        }
        cells = {column: value for column, value in row.items() if column in known}
        cells[_UNDECLARED_COLUMN] = (
            json.dumps(undeclared, sort_keys=True, default=str) if undeclared else ""
        )
        laid_out.append(cells)
    return laid_out


def _as_csv(table: RecipeTable) -> str:
    buffer = io.StringIO()
    # \n, not csv's default \r\n: the file is written through write_text with no
    # translation, so the default left a trailing \r on the last column of every row for
    # anyone using cut or awk.
    writer = csv.DictWriter(
        buffer, fieldnames=list(table.columns), restval="", lineterminator="\n"
    )
    writer.writeheader()
    for cells in _laid_out(table):
        writer.writerow({column: _as_cell(value) for column, value in cells.items()})
    return buffer.getvalue()


def recipe_tables(run_root: Path) -> dict[str, str]:
    """One CSV per variable, keyed by variable name. The in-memory view of the same
    rows and header the workbook is written from."""
    return {variable: _as_csv(table) for variable, table in read_run_values(run_root).items()}


def _run_metadata_rows(tables: dict[str, RecipeTable]) -> list[dict[str, Any]]:
    """The per-run, per-recipe facts, once each instead of on every row."""
    return [
        {
            **{column: tables[variable].provenance.get(column)
               for column in _RUN_METADATA_COLUMNS},
            "n_rows": len(tables[variable].rows),
            "n_patients": len({row.get("patient_id") for row in tables[variable].rows}),
        }
        for variable in sorted(tables)
    ]


# Excel refuses a longer cell and openpyxl raises rather than writing a corrupt file.
# Only a cited passage ever comes close; the run's own evidence artifacts keep the whole
# thing, so the cell says where it stopped rather than failing the write.
_LONGEST_CELL_A_WORKBOOK_HOLDS = 32_767


def _as_workbook_cell(value: Any) -> Any:
    """One value as it should sit in a spreadsheet cell.

    Booleans and numbers stay themselves — the TRUE/FALSE spelling exists only because
    CSV has no way to say "boolean", and a reader coercing it back is a reader who can
    get it wrong."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = ILLEGAL_CHARACTERS_RE.sub("", str(value))
    if len(text) > _LONGEST_CELL_A_WORKBOOK_HOLDS:
        cut = _LONGEST_CELL_A_WORKBOOK_HOLDS - 80
        return f"{text[:cut]}\n[... {len(text) - cut} more characters; the passage is " \
               "whole in the run's evidence artifacts ...]"
    return text


def _write_sheet(worksheet: Any, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """One table onto one sheet, header first."""
    worksheet.append(list(columns))
    for row in rows:
        worksheet.append([_as_workbook_cell(row.get(column)) for column in columns])
        for cell in worksheet[worksheet.max_row]:
            # openpyxl reads a leading '=' as a formula and writes the cell as one. A
            # pathology report can begin with anything, and a chart passage silently
            # becoming a live formula in the file a clinician opens is not acceptable;
            # every text cell is pinned to text.
            if isinstance(cell.value, str):
                cell.data_type = "s"
    # A 45-column table scrolled sideways loses which patient a row belongs to.
    worksheet.freeze_panes = "B2"


def write_values_tables(run_root: Path) -> list[Path]:
    """Write the run's answers into its output folder, and return what was written.

    Named by the layout module rather than here: three places were spelling these
    filenames independently, and a name only one of them changes is a name that drifts."""
    import os
    from uuid import uuid4

    from openpyxl import Workbook

    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        ANSWER_TABLE_SUFFIX,
        RUN_METADATA_TABLE_NAME,
        recipe_table_name,
        run_output_dir,
    )

    run_root = Path(run_root)
    output = run_output_dir(run_root)
    tables = read_run_values(run_root)
    if not tables:
        return []
    output.mkdir(parents=True, exist_ok=True)

    # The sheet is named for what the table IS, carried alongside rather than picked
    # back out of the filename: deriving it by splitting the name meant the tab read
    # "date_of_birth_results.xlsx" the moment the filename stopped carrying a run id.
    sheets: list[tuple[str, str, tuple[str, ...], list[dict[str, Any]]]] = [
        (recipe_table_name(run_root.name, variable), variable, table.columns,
         _laid_out(table))
        for variable, table in sorted(tables.items())
    ]
    sheets.append((RUN_METADATA_TABLE_NAME, "run_metadata", _RUN_METADATA_COLUMNS,
                   _run_metadata_rows(tables)))

    written: list[Path] = []
    for name, sheet_title, columns, rows in sheets:
        path = output / name
        workbook = Workbook()
        # Excel caps a sheet name at 31 characters and refuses []:*?/\ in one. A recipe
        # name is already constrained to a safe folder name, so only the length bites.
        workbook.active.title = sheet_title[:31]
        _write_sheet(workbook.active, tuple(columns), rows)
        # tmp + os.replace, like every other writer in this repo. This is the file that
        # gets downloaded and emailed, and it was the only one written in place — a
        # reader could see it half-finished, and two close-outs could interleave.
        tmp = output / f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
        try:
            workbook.save(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        written.append(path)

    # Anything this pass did not write is from a variable that is no longer extracted.
    # Left in place it keeps being counted as one of "the answers" by the run index and
    # keeps being read by whoever opens the folder.
    keep = {path.name for path in written}
    for stale in sorted(output.glob(f"*{ANSWER_TABLE_SUFFIX}")):
        if stale.name not in keep:
            stale.unlink()
    return written


def values_as_csv(run_root: Path, *, shape: str = "wide") -> str:
    """The WHOLE run as a single table, for a download that has to be one file.

    Distinct from the per-recipe tables written to disk, and needed alongside them: the
    workbench offers one download, and a reader who wants everything at once cannot be
    handed five files. Kept in two shapes because the honest answer depends on what is
    being asked.

    ``long`` is one row per patient/variable/field. Lossless, never widens, and the only
    shape that survives a recipe answering with a list.

    ``wide`` is one row per patient with a column per field across every recipe. What an
    analysis wants, at the cost of sparsity — recipes do not share fields, so most cells
    are empty, and a row-mapped recipe is deliberately left out of it rather than
    contributing rows[0..16] columns."""
    tables = read_run_values(run_root)
    buffer = io.StringIO()

    if shape == "long":
        writer = csv.DictWriter(
            buffer, fieldnames=["patient_id", "variable", "field", "value", "ok"]
        )
        writer.writeheader()
        for variable, table in sorted(tables.items()):
            for row in table.rows:
                for field, value in row.items():
                    if field in _LEADING_COLUMNS or field == "row":
                        continue
                    writer.writerow({
                        "patient_id": row.get("patient_id"), "variable": variable,
                        "field": field, "value": value, "ok": row.get("ok"),
                    })
        return buffer.getvalue()

    columns: list[str] = []
    per_patient: dict[str, dict[str, Any]] = {}
    linkage: dict[str, dict[str, Any]] = {}
    failed: dict[str, list[str]] = {}
    for variable, table in sorted(tables.items()):
        # One row per patient only. A recipe that answers with a list has no single row
        # to contribute here, and flattening it produced 285 columns for two patients.
        if any("row" in row for row in table.rows):
            continue
        for row in table.rows:
            patient_id = str(row.get("patient_id"))
            per_patient.setdefault(patient_id, {})
            # The same for every recipe of a run, so they go at the front once rather
            # than becoming date_of_birth.site_id beside date_of_death.site_id.
            linkage.setdefault(patient_id, {column: row.get(column, "")
                                            for column in _LINKAGE_COLUMNS})
            if not row.get("ok") and variable not in failed.get(patient_id, []):
                failed.setdefault(patient_id, []).append(variable)
            for field, value in row.items():
                if field in _LEADING_COLUMNS or field in _LINKAGE_COLUMNS:
                    continue
                column = f"{variable}.{field}"
                if column not in columns:
                    columns.append(column)
                per_patient[patient_id][column] = value

    writer = csv.writer(buffer)
    writer.writerow(["patient_id", *_LINKAGE_COLUMNS, *columns, "variables_that_failed"])
    for patient_id in sorted(per_patient):
        row = per_patient[patient_id]
        writer.writerow(
            [patient_id]
            + [linkage.get(patient_id, {}).get(column, "") for column in _LINKAGE_COLUMNS]
            + [row.get(column, "") for column in columns]
            # Named, never left blank: an empty cell reads as "not in the chart", which
            # is the one thing a failed extraction must not be mistaken for.
            + ["; ".join(sorted(failed.get(patient_id, [])))]
        )
    return buffer.getvalue()
