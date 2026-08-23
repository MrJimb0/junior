"""Ingest decides the chart-field mapping permanently, so it has to say so.

`resolve_metadata_columns_for` runs at ingest and records its answer in each file's
sidecar; embed reads that and re-derives only when it is absent or names a column the
table does not have. A field resolved to None is settled, not missing. The CLI used to
filter these advisories out of the ingest report on the stated reasoning that chunk
metadata is attached at embed — which is not what this pipeline does, and which made the
one place that stayed quiet the one place where quiet is permanent.

The other half of the old reasoning was right and is tested here too: a lab table has no
author because lab tables do not have authors, and a report that says so once per file
per field per patient is one nobody reads. Only files carrying free text are reported.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apps_and_interfaces.command_line_interface import (
    _report_what_ingest_is_about_to_settle,
)


class _Advisory:
    """The shape `_report_what_ingest_is_about_to_settle` reads off a PreflightIssue."""

    def __init__(self, detail: str, fields: tuple[str, ...] = (), text: bool = False):
        self.detail = detail
        self.fields = fields
        self.file_contributes_text = text


class _Preflight:
    def __init__(self, advisories: list[_Advisory]):
        self.advisories = advisories


def _reported(advisories: list[_Advisory], cfg: dict, capsys) -> str:
    _report_what_ingest_is_about_to_settle(_Preflight(advisories), cfg)
    return capsys.readouterr().err


A_MAP = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}


def test_an_unmapped_field_on_a_text_file_is_reported(capsys):
    """The case that was silently dropped, and the only one that reaches retrieval."""
    reported = _reported(
        [_Advisory("no column found for: specialty", ("specialty",), text=True)],
        A_MAP, capsys,
    )

    assert "specialty" in reported
    assert "1 file(s)" in reported


def test_a_table_with_no_free_text_is_not_reported(capsys):
    """A lab table has no author because lab tables do not have authors.

    On the shipped example cohort this filter is the difference between one line and
    eight, and the eight are all about medication and lab tables."""
    reported = _reported(
        [
            _Advisory("no column found for: author", ("author", "title"), text=False),
            _Advisory("no column found for: age", ("age",), text=False),
        ],
        A_MAP, capsys,
    )

    assert reported.strip() == "", f"reported something about a table with no text:\n{reported}"


def test_fields_are_grouped_not_listed_per_file(capsys):
    """A 500-patient cohort has the same few unmapped fields 500 times over."""
    reported = _reported(
        [_Advisory("no column found for: specialty", ("specialty",), text=True)] * 500,
        A_MAP, capsys,
    )

    assert "500 file(s)" in reported
    assert len(reported.strip().splitlines()) <= 2, reported


def test_only_the_worst_three_fields_are_named(capsys):
    reported = _reported(
        [
            _Advisory("x", ("a",), text=True), _Advisory("x", ("a",), text=True),
            _Advisory("x", ("b",), text=True), _Advisory("x", ("c",), text=True),
            _Advisory("x", ("d",), text=True), _Advisory("x", ("e",), text=True),
        ],
        A_MAP, capsys,
    )

    assert "2 more field(s)" in reported


def test_malformation_is_still_reported(capsys):
    """What the function reported before, and must go on reporting: an advisory with no
    fields is a column or a date the parquet itself will carry wrong."""
    reported = _reported(
        [_Advisory("columns ['Date', 'date'] differ only by case", (), text=True)],
        A_MAP, capsys,
    )

    assert "differ only by case" in reported


# --- the no-map block -------------------------------------------------------------

def test_no_column_map_says_so_in_one_line(capsys):
    """The log line, for the runs nobody is watching.

    The full explanation belongs at the confirmation, where somebody is about to decide.
    A paragraph here would be that same warning scrolling past before the question —
    which is exactly what it did, and why it went unread."""
    reported = _reported(
        [_Advisory("no column found for: document_date", ("document_date",), text=True)],
        {}, capsys,
    )

    assert "no column map is set" in reported
    assert len(reported.strip().splitlines()) == 2, reported


def test_a_mapped_project_is_not_told_it_has_no_map(capsys):
    """With the shipped example map set, the example cohort leaves the same eight
    fields unresolved as with no map at all. So unresolved fields cannot be the
    trigger: an alarm on every run of every correctly-configured cohort is one the
    operator learns to scroll past."""
    reported = _reported(
        [_Advisory("no column found for: specialty", ("specialty",), text=True)],
        A_MAP, capsys,
    )

    assert "no column map" not in reported


def test_nothing_unmapped_means_no_block_even_without_a_map(capsys):
    """Silence is the right output when there is nothing to say."""
    reported = _reported([], {}, capsys)

    assert reported.strip() == ""


# --- the confirmation itself ------------------------------------------------------

@pytest.fixture
def someone_at_the_keyboard(monkeypatch):
    """`is_interactive` is False under pytest, which would skip every confirmation."""
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: True
    )


def _asked(cfg: dict, tmp_path: Path, monkeypatch, capsys, typed: str = "") -> tuple[bool, str]:
    from apps_and_interfaces.command_line_interface import _confirm_column_mapping

    monkeypatch.setattr("click.termui.visible_prompt_func", lambda _p="": typed)
    went_ahead = _confirm_column_mapping(cfg, tmp_path, 9)
    return went_ahead, capsys.readouterr().out


def test_ingest_asks_before_it_settles_the_mapping(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    """The thing embed and extract always had and ingest never did.

    Ingest is the stage that freezes the mapping, so it is the one that most needed
    a confirmation, and it was the only one running straight through."""
    _, shown = _asked({"source_root": "/charts"}, tmp_path, monkeypatch, capsys, "n")

    assert "About to ingest" in shown
    assert "9 from /charts" in shown


def test_enter_declines_when_no_column_map_is_set(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    """An unset map is not a choice anyone made — a project inherits it from the
    bundled settings — so the path of least resistance must not confirm it."""
    went_ahead, shown = _asked({"source_root": "/charts"}, tmp_path, monkeypatch, capsys)

    assert went_ahead is False
    assert "no column map" in shown
    assert "--force" in shown, "the only instruction that actually works is missing"
    assert "junior columns" in shown


def test_enter_accepts_when_a_column_map_is_set(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    """With a map this is an ordinary confirmation, like embed's. Making a correctly
    configured cohort answer an alarm every run is how an alarm gets ignored."""
    went_ahead, shown = _asked(
        {"source_root": "/charts", "chart_columns_file": "/maps/site_map.yaml"},
        tmp_path, monkeypatch, capsys,
    )

    assert went_ahead is True
    assert "site_map.yaml" in shown
    assert "no column map" not in shown


def test_a_scripted_run_is_never_asked_but_is_still_told(tmp_path, monkeypatch, capsys):
    """No keyboard, so no question — a SLURM array task must not block on stdin.

    It is still TOLD. What a stage is about to use is not a function of who is watching,
    and the workbench runs these very commands with their output piped into its log
    panel. Bailing out before the display, rather than before the question, is why
    somebody clicking Ingest or Embed there never saw the column mapping or the
    embedding model being locked into their run. Reported exactly that way."""
    from apps_and_interfaces.command_line_interface import _confirm_column_mapping

    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: False
    )
    # A question would read a stdin that is not there. Blocking is the failure this
    # guards against, so asking at all fails the test rather than hanging it.
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.click.confirm",
        lambda *a, **k: pytest.fail("a scripted run was asked a question"),
    )

    assert _confirm_column_mapping({"source_root": "/charts"}, tmp_path, 9) is True
    assert "About to ingest" in capsys.readouterr().out, (
        "a scripted run is told nothing about what it is settling"
    )


def test_a_run_that_already_ingested_is_not_re_asked(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    """A plain re-run cannot change the answer — the ingest cache is keyed on whether
    the source files changed — so asking again would imply it could."""
    from apps_and_interfaces.command_line_interface import _confirm_column_mapping

    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.read_stage_progress",
        lambda *a, **k: type("P", (), {"completed": 9})(),
        raising=False,
    )
    import jr_pipeline.runtime_infrastructure.run_progress as run_progress

    monkeypatch.setattr(
        run_progress, "read_stage_progress",
        lambda *a, **k: type("P", (), {"completed": 9})(),
    )

    assert _confirm_column_mapping({"source_root": "/charts"}, tmp_path, 9) is True
    assert "About to ingest" not in capsys.readouterr().out


# --- re-running after changing the map --------------------------------------------

def _an_ingested_run(run_root: Path, recorded: dict, columns: list[str]) -> None:
    import json

    structured = run_root / "patients" / "PT1" / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    (structured / "labs.parquet.meta.json").write_text(
        json.dumps({"payload": {
            "metadata_columns": recorded,
            "columns": [{"name": c} for c in columns],
        }}),
        encoding="utf-8",
    )


def _a_map(tmp_path: Path, entry: dict) -> str:
    path = tmp_path / "map.yaml"
    path.write_text(yaml.safe_dump({"chunk_metadata_columns": {"labs": entry}}), encoding="utf-8")
    return str(path)


def test_changing_the_map_then_re_running_ingest_is_caught(tmp_path, monkeypatch, capsys,
                                                           someone_at_the_keyboard):
    """The trap: set a map, run ingest, get a tick, change nothing.

    The ingest cache is keyed on whether the source files changed. The mapping is not
    part of that check, so every patient is reused and the new map is never recorded —
    and the run reports success."""
    from apps_and_interfaces.command_line_interface import _confirm_column_mapping

    _an_ingested_run(tmp_path, {"document_date": None, "author": None},
                     ["result_date", "authorizing_provider", "text"])
    monkeypatch.setattr(
        "jr_pipeline.runtime_infrastructure.run_progress.read_stage_progress",
        lambda *a, **k: type("P", (), {"completed": 1})(),
    )
    monkeypatch.setattr("click.termui.visible_prompt_func", lambda _p="": "n")

    cfg = {
        "source_root": "/charts",
        "chart_columns_file": _a_map(tmp_path, {
            "document_date": "result_date", "author": "authorizing_provider",
        }),
    }
    went_ahead = _confirm_column_mapping(cfg, tmp_path, 1)
    shown = capsys.readouterr().out

    assert went_ahead is False
    assert "different column mapping" in shown
    assert "result_date" in shown and "authorizing_provider" in shown
    assert "--force" in shown, "the only thing that applies it is unnamed"


def test_re_running_ingest_with_the_same_map_asks_nothing(tmp_path, monkeypatch, capsys,
                                                          someone_at_the_keyboard):
    """A re-run for a late-arriving patient must not interrogate the operator."""
    from apps_and_interfaces.command_line_interface import _confirm_column_mapping

    entry = {"document_date": "result_date", "author": "authorizing_provider"}
    _an_ingested_run(tmp_path, {"document_date": "result_date",
                                "author": "authorizing_provider"},
                     ["result_date", "authorizing_provider", "text"])
    monkeypatch.setattr(
        "jr_pipeline.runtime_infrastructure.run_progress.read_stage_progress",
        lambda *a, **k: type("P", (), {"completed": 1})(),
    )

    cfg = {"source_root": "/charts", "chart_columns_file": _a_map(tmp_path, entry)}

    assert _confirm_column_mapping(cfg, tmp_path, 1) is True
    assert capsys.readouterr().out.strip() == ""


# --- embed asks the same question about its own half ------------------------------

def _asked_to_embed(cfg: dict, tmp_path: Path, monkeypatch, capsys, typed: str = "") -> tuple:
    from apps_and_interfaces.command_line_interface import _confirm_embedding_model

    monkeypatch.setattr("click.termui.visible_prompt_func", lambda _p="": typed)
    went_ahead = _confirm_embedding_model(cfg, tmp_path, 9)
    return went_ahead, capsys.readouterr().out


def test_embed_enter_declines_when_nothing_names_the_text_columns(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    """Ingest settles the metadata half of the map; embed settles the text half.

    Which column of each file gets embedded IS the corpus. With no map only a column
    literally named `text` is looked for, so a table whose narrative sits under another
    name is silently left out. That is the last gate before the slow step, so Enter must
    not take it."""
    went_ahead, shown = _asked_to_embed(
        {"encoder": {"model_id": str(tmp_path / "model")}}, tmp_path, monkeypatch, capsys
    )

    assert went_ahead is False
    assert "no column map" in shown
    assert "`text`" in shown, "it has to name the column it will actually look for"
    assert "junior columns" in shown
    assert "chart_columns_file" in shown, "no pointer to what to edit"


def test_embed_enter_accepts_when_the_text_columns_are_named(
    tmp_path, monkeypatch, capsys, someone_at_the_keyboard
):
    a_map = tmp_path / "map.yaml"
    a_map.write_text(
        yaml.safe_dump({"chunk_metadata_columns": {"clinical_note": {"text_columns": ["text"]}}}),
        encoding="utf-8",
    )
    went_ahead, shown = _asked_to_embed(
        {"encoder": {"model_id": str(tmp_path / "model")}, "chart_columns_file": str(a_map)},
        tmp_path, monkeypatch, capsys,
    )

    assert went_ahead is True
    assert "No column map set" not in shown
    assert "clinical_note.text" in shown
    # The map is the file you edit to change that list, so the screen has to name it.
    assert "map.yaml" in shown


def test_embed_names_the_tables_it_will_actually_read(tmp_path, monkeypatch, capsys,
                                                      someone_at_the_keyboard):
    """With no map, what gets embedded is knowable rather than describable.

    Ingest recorded every table's columns in its sidecar, so embed can list the tables
    that have a text column instead of saying it will work it out per file."""
    import json

    structured = tmp_path / "patients" / "PT1" / "structured"
    structured.mkdir(parents=True)
    for stem, columns in (("clinical_note", ["date", "text"]), ("labs", ["result", "value"])):
        (structured / f"{stem}.parquet.meta.json").write_text(
            json.dumps({"payload": {"columns": [{"name": c} for c in columns]}}),
            encoding="utf-8",
        )

    _, shown = _asked_to_embed(
        {"encoder": {"model_id": str(tmp_path / "model")}}, tmp_path, monkeypatch, capsys
    )

    columns_block = shown.split("    model")[0]
    assert "clinical_note.text" in columns_block, shown
    assert "labs" not in columns_block, "labs has no text column, so it must not be listed"


# --- pinning a map after the fact -------------------------------------------------

def _a_run_under(output_root: Path, run_id: str = "20260101_000000_aa") -> None:
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        SENSITIVE_LABEL,
        pipeline_run_receipts_root,
    )

    (output_root / SENSITIVE_LABEL / pipeline_run_receipts_root().name / run_id).mkdir(
        parents=True, exist_ok=True
    )


def test_pinning_a_map_warns_that_existing_runs_keep_theirs(tmp_path, capsys):
    """Verified end to end: with a map newly set, a plain re-ingest leaves the sidecar's
    document_date at None and still prints a tick. Only --force rewrites it."""
    from apps_and_interfaces.command_line_interface import _say_what_this_map_cannot_reach

    _a_run_under(tmp_path)
    _say_what_this_map_cannot_reach({"output_root": str(tmp_path)})
    reported = capsys.readouterr().out

    assert "1 run(s)" in reported
    assert "--force" in reported, "a plain re-run is a no-op; saying otherwise misleads"


def test_a_project_with_no_runs_yet_is_told_nothing(tmp_path, capsys):
    from apps_and_interfaces.command_line_interface import _say_what_this_map_cannot_reach

    _say_what_this_map_cannot_reach({"output_root": str(tmp_path)})

    assert capsys.readouterr().out.strip() == ""


def test_a_project_that_does_not_say_where_it_writes_is_not_guessed_at(tmp_path, capsys):
    """Falling back to the ambient data root would answer about somebody else's runs.

    A developer checkout's ./data holds dozens of receipts, so a fallback would fire this
    warning on every `junior columns` run in the repo."""
    from apps_and_interfaces.command_line_interface import _say_what_this_map_cannot_reach

    _say_what_this_map_cannot_reach({"project": "no_output_root_here"})

    assert capsys.readouterr().out.strip() == ""


# --- the shipped example, end to end ----------------------------------------------

@pytest.mark.parametrize("column_map", [None, "deployment/example_site/example_export_column_map.yaml"])
def test_the_shipped_example_reports_one_field_not_eight(tmp_path, column_map, capsys):
    """The front door: README tells a new user to run `junior ingest` on a fresh clone,
    which lands on examples/ with no map. Whatever this prints, a stranger reads first."""
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    repo = Path(__file__).resolve().parents[2]
    examples = repo / "examples"
    patients = [p.name for p in sorted(examples.iterdir()) if p.is_dir()][:3]
    if not patients:
        pytest.skip("no example patient folders in this checkout")

    cfg = {
        "source_root": str(examples),
        "run_id": "20260101_000000_aa",
        "output_root": str(tmp_path),
    }
    if column_map:
        cfg["chart_columns_file"] = str(repo / column_map)

    preflight = preflight_patients(cfg=cfg, patients=patients)
    reported = _reported(list(preflight.advisories), cfg, capsys)

    named_fields = [line for line in reported.splitlines() if "no column holds" in line]
    assert len(named_fields) == 1, (
        f"the example cohort should name one field, not {len(named_fields)}:\n{reported}"
    )
    assert "specialty" in reported
