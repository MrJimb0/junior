"""A reviewer's judgement must never be recorded against a patient they never saw.

Three failures would keep that reachable and each is pinned here:
  * a selected run whose patient folders are gone (or whose result.json files all fail
    to parse) produces an empty row list, which must not read as "no run selected";
  * the writers must refuse from any view that is not a real run;
  * a stale variable selection — the reviewer's pick survives a run switch because
    Shiny drops a None ``selected`` — must not write a confirmation of a value that
    has no row.

Also pinned: evidence text that cannot be resolved must say why rather than render as
an empty quote (an empty quote reads as "the chart said nothing here"), and an exported
CSV must name its run and whether it is real.
"""
from __future__ import annotations

import json
from pathlib import Path

import app as review_app
import polars as pl
import pytest
import run_results

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore

RUN_ID = "20260101_010101_aa"
PATIENT_ID = "Patient_A"

_ENVELOPE_WITH_EVIDENCE = {
    "payload": {
        "variable": "date_of_birth",
        "ok": True,
        "data": {
            "date_of_birth": {"type": "string", "value": "1970-01-01"},
            "date_of_birth_evidence_chunk_id": "Patient_A:notes:3",
        },
        "recipe": {"name": "demographics", "version": "1"},
    }
}


def _write_extract(dr: Path, envelope: dict, variable: str = "date_of_birth") -> None:
    var_dir = extract_output_dir(phi_patient_run_dir(RUN_ID, PATIENT_ID, dr)) / variable
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / "result.json").write_text(json.dumps(envelope), encoding="utf-8")


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Point the loader at an empty tmp data root for the duration of one test."""
    monkeypatch.setattr(run_results, "DATA_ROOT", tmp_path)
    return tmp_path


class _ChunkStoreRaising:
    """Stands in for PatientChunkStore, failing the way the real one fails."""

    def __init__(self, failure: Exception) -> None:
        self._failure = failure

    def text_for(self, chunk_id: str) -> str:
        raise self._failure


# ── the selected run failed to load ─────────────────────────────────────────────

def test_a_selected_run_with_no_readable_output_keeps_its_own_identity(data_root):
    view = review_app._load_view(RUN_ID, PATIENT_ID)

    assert view.rows == [], "showed rows for a run that produced none"
    assert view.patient_id == PATIENT_ID
    assert view.run_id == RUN_ID, "dropped the run the reviewer actually selected"
    assert view.is_real_run is False
    assert view.nothing_selected is False


def test_the_selected_run_and_patient_are_named_in_what_the_reviewer_reads(data_root):
    label = review_app._load_view(RUN_ID, PATIENT_ID).source_label

    assert RUN_ID in label and PATIENT_ID in label
    assert "NO READABLE EXTRACT OUTPUT" in label


def test_a_run_whose_patient_folders_are_gone_is_not_the_empty_state(data_root):
    # The reachable path: a run whose patient dirs are gone leaves the patient select
    # empty, so the app is asked for run + "". A run WAS selected, so this is the
    # reviewer's own pick failing — not a boot with nothing picked yet.
    view = review_app._load_view(RUN_ID, "")

    assert view.nothing_selected is False, "a failed pick read as nothing selected"
    assert view.rows == []
    assert view.run_id == RUN_ID
    assert "no patient selected" in view.source_label


def test_only_a_boot_with_nothing_selected_is_the_empty_state(data_root):
    view = review_app._load_view(None, None)

    assert view.nothing_selected is True
    assert view.rows == []


def test_a_loader_crash_is_reported_not_swallowed_into_the_empty_state(data_root, monkeypatch):
    def _explode(run_id, patient_id):
        raise OSError("cannot list this run's patient folders")

    monkeypatch.setattr(run_results, "load_run_variables", _explode)

    view = review_app._load_view(RUN_ID, PATIENT_ID)

    assert view.nothing_selected is False
    assert "cannot list this run's patient folders" in view.source_label


# ── no irreversible write from a view that is not a real run ────────────────────

def test_no_feedback_can_be_written_when_nothing_is_selected():
    refusal = review_app._why_feedback_cannot_be_saved(review_app._nothing_selected_view())

    assert refusal != "", "the empty state left the feedback writers open"
    assert "No run is selected" in refusal
    assert "Refresh runs" in refusal, "does not say what to do instead"


def test_no_feedback_can_be_written_from_a_run_that_produced_nothing(data_root):
    refusal = review_app._why_feedback_cannot_be_saved(review_app._load_view(RUN_ID, PATIENT_ID))

    assert refusal != ""
    assert RUN_ID in refusal and PATIENT_ID in refusal
    assert "Re-run extract" in refusal, "does not say what to do instead"


def test_a_real_run_stays_writable(data_root):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)

    view = review_app._load_view(RUN_ID, PATIENT_ID)

    assert view.is_real_run is True
    assert review_app._why_feedback_cannot_be_saved(view) == "", (
        "the guard blocked the one case feedback capture exists for"
    )


def test_every_feedback_writer_consults_the_same_guard():
    """The guard is only worth having if all four writers ask it. Tie the count to the
    source so a fifth writer added later cannot quietly skip it."""
    source = (Path(review_app.__file__).parent / "app.py").read_text(encoding="utf-8")
    writers = ("write_correction(", "write_confirmation(", "write_sampling_frame(",
               "write_chunk_relevance(")

    assert all(name in source for name in writers)
    # The whole idiom, not just the call: the panels that explain the state to the
    # reviewer ask the same helper for its sentence, and only a writer binds the answer
    # to `refusal` and returns on it.
    assert source.count("refusal = _why_feedback_cannot_be_saved(view)") == len(writers), (
        "a feedback writer does not check whether the active view is a real run"
    )


# ── no record for a variable that has no row ────────────────────────────────────

def test_a_variable_with_no_row_is_refused_rather_than_confirmed(data_root):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    view = review_app._load_view(RUN_ID, PATIENT_ID)

    # The stale selection: the reviewer's previous variable is not in this run.
    assert "stage" not in view.rows_by_variable
    reason = review_app._no_row_for_variable_reason("stage", view)

    assert "stage" in reason and PATIENT_ID in reason and RUN_ID in reason
    assert "Pick a variable" in reason, "does not say what to do instead"


def test_the_confirm_and_correct_writers_refuse_a_missing_row():
    """Both would otherwise write unconditionally: confirm recording agreed=true with
    reviewed_value=None, correct recording original_value=None — an agreement with, and
    a correction of, a value the run never produced."""
    source = (Path(review_app.__file__).parent / "app.py").read_text(encoding="utf-8")

    assert source.count("_no_row_for_variable_reason(variable, view)") == 3, (
        "correction, confirmation and chunk-relevance must each refuse a missing row"
    )
    assert "row.value if row else None" not in source
    assert "row.model if row else None" not in source


# ── an unresolvable evidence quote must not render as an empty string ───────────

def test_a_chart_that_changed_since_embed_says_so_instead_of_quoting_nothing(data_root, monkeypatch):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    changed = RuntimeError(
        "source row for 'Patient_A:notes:3' changed since embed (content-hash mismatch): "
        "chunk offsets would slice the wrong text. Re-run embed for this patient."
    )
    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: _ChunkStoreRaising(changed))

    row = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0]

    assert row.evidence_quote != "", "an unreadable quote exported as an empty cell"
    assert "content-hash mismatch" in row.evidence_quote
    assert row.evidence[0].source_changed_since_embed is True, (
        "this run's evidence is untrustworthy and nothing says so"
    )
    assert row.evidence_text_is_readable is False


def test_an_unknown_chunk_id_says_which_step_to_re_run(data_root, monkeypatch):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    monkeypatch.setattr(
        run_results, "_chunk_store",
        lambda *_: _ChunkStoreRaising(KeyError("Patient_A:notes:3")),
    )

    row = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0]

    assert "not in the patient's chunk index" in row.evidence_quote
    assert "re-run embed" in row.evidence_quote
    assert row.evidence[0].source_changed_since_embed is False, (
        "a missing chunk id is not evidence that the chart changed"
    )


def test_a_row_that_is_no_longer_there_is_not_the_same_as_a_chart_that_said_nothing(
    data_root, monkeypatch,
):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    monkeypatch.setattr(
        run_results, "_chunk_store",
        lambda *_: _ChunkStoreRaising(IndexError("row_id 9 out of range for notes.csv")),
    )

    row = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0]

    assert "no longer there" in row.evidence_quote


def test_a_chunk_store_that_could_not_be_opened_is_not_a_silent_blank(data_root, monkeypatch):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: None)

    row = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0]

    assert row.evidence_quote != ""
    assert "chunk store could not be opened" in row.evidence_quote


def test_a_chunk_that_resolves_to_no_text_says_what_to_do_about_it(data_root, monkeypatch):
    """A chunk pointing at a numeric column, or at a table row that renders empty, is the
    one unreadable-evidence case no re-run can fix — so it is the one that would name
    nothing to do. It has to say what happened and what to check, in the Review table
    and in the exported CSV both."""
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)

    class _StoreWithNothingToQuote:
        def text_for(self, chunk_id):
            return ""

    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: _StoreWithNothingToQuote())

    quote = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0].evidence_quote

    assert "no text to quote" in quote, "does not say what happened"
    assert "confirm this value against" in quote, "does not say what to do next"


# ── a chart that changed under a finished run: which stage, and what clears it ──────

TABLE_CHUNK_ID = f"{PATIENT_ID}:demographics:0:TABLE"


def _envelope_citing(chunk_id: str) -> dict:
    envelope = json.loads(json.dumps(_ENVELOPE_WITH_EVIDENCE))
    envelope["payload"]["data"]["date_of_birth_evidence_chunk_id"] = chunk_id
    return envelope


def _store_whose_source_row_changed_since_embed(patient_root: Path) -> PatientChunkStore:
    """A real chunk store in the state embed's per-row hash catches: the chunk index still
    carries the hash of the text embed read, the parquet now holds different text."""
    (patient_root / "structured").mkdir(parents=True)
    pl.DataFrame({
        "chunk_id": ["Patient_A:notes:3"], "source_file": ["notes.csv"],
        "source_column": ["note_text"], "row_id": [0], "char_start": [0], "char_end": [4],
        "source_text_sha256": ["sha256:the-text-embed-hashed"],
    }).write_parquet(patient_root / "chunk_index.parquet")
    pl.DataFrame({"note_text": ["edited since embed read it"]}).write_parquet(
        patient_root / "structured" / "notes.parquet")
    return PatientChunkStore(patient_root=patient_root)


def _store_whose_structured_table_changed_since_ingest(patient_root: Path) -> PatientChunkStore:
    """The other one: a :TABLE value is read straight out of the parquet, and the parquet no
    longer hashes to what ingest recorded in the sidecar beside it."""
    (patient_root / "structured").mkdir(parents=True)
    pl.DataFrame({"chunk_id": []}, schema={"chunk_id": pl.Utf8}).write_parquet(
        patient_root / "chunk_index.parquet")
    pl.DataFrame({"birth_date": ["1970-01-01"]}).write_parquet(
        patient_root / "structured" / "demographics.parquet")
    (patient_root / "structured" / "demographics.parquet.meta.json").write_text(
        json.dumps({"payload": {"source_file": "demographics.csv",
                                "parquet_content_hash": "sha256:the-bytes-ingest-indexed"}}),
        encoding="utf-8",
    )
    return PatientChunkStore(patient_root=patient_root)


def test_a_table_that_changed_since_ingest_is_not_told_to_re_run_embed(
    data_root, tmp_path, monkeypatch,
):
    """Two different integrity failures must not share one remedy. For a structured
    table that changed since INGEST, "re-run embed" cannot work — embed reads that
    table, so it would re-cut the chunks out of the same wrong bytes and report
    success. The operator watches the warning survive a re-run they were told would
    clear it."""
    _write_extract(data_root, _envelope_citing(TABLE_CHUNK_ID))
    store = _store_whose_structured_table_changed_since_ingest(tmp_path / "patient")
    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: store)

    chunk = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0].evidence[0]

    assert chunk.source_changed_since_embed is True, "the run's evidence is stale and nothing says so"
    assert chunk.chart_changed_next_step == "re-run ingest for this patient, then embed and extract"


def test_a_row_that_changed_since_embed_still_gets_the_embed_remedy(
    data_root, tmp_path, monkeypatch,
):
    """The counterpart — telling the two apart must not have cost the case that was already
    right, and re-ingesting to fix a chunk offset would be a wasted afternoon."""
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    store = _store_whose_source_row_changed_since_embed(tmp_path / "patient")
    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: store)

    chunk = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0].evidence[0]

    assert chunk.source_changed_since_embed is True
    assert chunk.chart_changed_next_step == "re-run embed for this patient, then extract"


def test_a_chunk_store_failure_that_is_not_a_changed_chart_is_not_given_a_re_run_to_do(
    data_root, monkeypatch,
):
    """A failure that is neither of the two integrity checks has to keep its own message
    and claim no remedy, rather than send the reviewer off to re-run a stage that was
    never the problem."""
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    monkeypatch.setattr(
        run_results, "_chunk_store",
        lambda *_: _ChunkStoreRaising(RuntimeError("the chunk index could not be opened")),
    )

    chunk = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0].evidence[0]

    assert chunk.source_changed_since_embed is False
    assert chunk.chart_changed_next_step == ""
    assert "could not be opened" in chunk.text_unavailable_reason


def test_a_resolvable_quote_is_still_just_the_quote(data_root, monkeypatch):
    """The counterpart — the reasons must not have cost the normal case."""
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)

    class _Store:
        def text_for(self, chunk_id):
            return "Patient reports onset in March."

    monkeypatch.setattr(run_results, "_chunk_store", lambda *_: _Store())

    row = run_results.load_run_variables(RUN_ID, PATIENT_ID)[0]

    assert row.evidence_quote == "Patient reports onset in March."
    assert row.evidence_text_is_readable is True


# ── the CSV export carries its own provenance ───────────────────────────────────

def test_an_empty_state_export_names_itself_in_every_part_that_leaves_the_app():
    view = review_app._nothing_selected_view()

    csv_text = run_results.rows_as_csv(
        view.rows, view.patient_id, run_id=view.export_run_id, source=view.export_source,
    )
    header, *body = csv_text.splitlines()

    assert header.startswith("run_id,source,patient_id,")
    assert body == [], "an empty state exported rows"
    assert "NO_RUN" in view.export_source
    assert "NO_RUN" in review_app._csv_export_name(view)


def test_a_real_export_names_its_run_in_every_row(data_root):
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    view = review_app._load_view(RUN_ID, PATIENT_ID)

    csv_text = run_results.rows_as_csv(
        view.rows, view.patient_id, run_id=view.export_run_id, source=view.export_source,
    )
    _header, *body = csv_text.splitlines()

    assert body, "a real run exported no rows"
    for line in body:
        assert line.startswith(f"{RUN_ID},real_pipeline_run,{PATIENT_ID},")


def test_the_downloaded_filename_makes_the_no_output_case_obvious(data_root):
    unreadable_name = review_app._csv_export_name(review_app._load_view(RUN_ID, PATIENT_ID))
    _write_extract(data_root, _ENVELOPE_WITH_EVIDENCE)
    real_name = review_app._csv_export_name(review_app._load_view(RUN_ID, PATIENT_ID))

    assert "NO_OUTPUT" in unreadable_name
    assert real_name == f"junior_extraction_{RUN_ID}_{PATIENT_ID}.csv"
    assert "NO_OUTPUT" not in real_name and "NO_RUN" not in real_name
