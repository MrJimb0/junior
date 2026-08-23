"""The confirmations are reached by the stage commands, not just callable in isolation.

Every other test of these gates calls them directly. That leaves the wiring untested,
and the wiring is where they live: `is_interactive()` is `sys.stdin.isatty()`, which is
False under pytest, so every CliRunner test in the suite takes the nobody-at-the-keyboard
path and the gates return True before doing anything. The whole `if name == "ingest" …`
block could be deleted from `_common_step` and the suite would stay green — verified, it
does. These tests patch someone in at the keyboard so the wiring is exercised.
"""
from __future__ import annotations

import pytest
import yaml
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main

CHART = "patient_id,result_date,authorizing_provider,text\nP1,2026-01-02,Dr Who,a note\n"


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """A project with one patient, and someone at the keyboard."""
    charts = tmp_path / "charts" / "P1"
    charts.mkdir(parents=True)
    (charts / "labs.csv").write_text(CHART, encoding="utf-8")
    project = tmp_path / "study"
    project.mkdir()
    (project / "junior_study.yaml").write_text(yaml.safe_dump({
        "project": "study", "run_id": None,
        "source_root": str(tmp_path / "charts"), "output_root": str(project),
    }), encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: True
    )
    return project


def _run(args: list[str], typed: str):
    return CliRunner().invoke(main, args, input=typed)


def test_ingest_asks_and_declining_writes_nothing(cohort):
    """The wiring: the gate is reached, and answering no stops the stage."""
    result = _run(["ingest"], "n\n")

    assert "About to ingest" in result.output, result.output
    assert "no column map" in result.output
    assert "cancelled — nothing was ingested" in result.output
    # And the run folder is not created before the answer — the reason ensure_layout
    # moved below the confirmations.
    assert not (cohort / "CONTAINS_PHI").exists(), sorted(p.name for p in cohort.iterdir())


def test_ingest_enter_declines_when_no_map_is_set(cohort):
    result = _run(["ingest"], "\n")

    assert "cancelled — nothing was ingested" in result.output, result.output
    assert not (cohort / "CONTAINS_PHI").exists()


def test_accepting_runs_it_and_creates_the_run(cohort):
    result = _run(["ingest"], "y\n")

    assert result.exit_code == 0, result.output
    assert "ingest finished" in result.output
    assert (cohort / "CONTAINS_PHI" / "pipeline_run_receipts").is_dir()


def test_one_ingested_patient_does_not_silence_the_gate_for_the_rest(cohort, tmp_path):
    """`completed` is a count. Treating it as "this run is ingested" meant trying one
    patient first turned the gate off for every patient still undecided."""
    assert _run(["ingest", "--patient", "P1"], "").exit_code == 0
    second = tmp_path / "charts" / "P2"
    second.mkdir()
    (second / "labs.csv").write_text(CHART.replace("P1", "P2"), encoding="utf-8")

    result = _run(["ingest"], "n\n")

    assert "About to ingest" in result.output, result.output
    assert "cancelled — nothing was ingested" in result.output


def test_a_single_patient_run_is_never_asked(cohort):
    """--patient is the SLURM array path; it must not block on stdin."""
    result = _run(["ingest", "--patient", "P1"], "")

    assert result.exit_code == 0, result.output
    assert "About to ingest" not in result.output


def test_embed_gate_is_reached_too(cohort):
    assert _run(["ingest"], "y\n").exit_code == 0
    # Embed refuses outright with no encoder, and again when the encoder names a folder
    # that is not there, so this project needs one that exists for the gate itself to be
    # reachable. It need not hold real weights: the question comes before anything loads.
    (cohort / "model").mkdir()
    settings = cohort / "junior_study.yaml"
    settings.write_text(settings.read_text() + f"encoder:\n  model_id: {cohort / 'model'}\n")

    result = _run(["embed"], "n\n")

    assert "About to embed" in result.output, result.output
    assert "nothing was embedded" in result.output


def test_embed_refuses_up_front_with_no_encoder(cohort):
    """A project with no model fails once, about the project — not once per patient."""
    assert _run(["ingest"], "y\n").exit_code == 0

    result = _run(["embed"], "y\n")

    assert "no embedding model set" in result.output, result.output
    assert "encoder.model_id" in result.output


def test_changing_the_map_after_ingest_is_caught_through_the_command(cohort, tmp_path):
    """End to end: the drift warning fires from `junior ingest`, not just in isolation."""
    assert _run(["ingest"], "y\n").exit_code == 0
    a_map = cohort / "m.yaml"
    a_map.write_text(yaml.safe_dump({"chunk_metadata_columns": {"labs": {
        "text_columns": ["text"], "document_date": "result_date",
        "author": "authorizing_provider",
    }}}), encoding="utf-8")
    settings = cohort / "junior_study.yaml"
    settings.write_text(settings.read_text() + f"chart_columns_file: {a_map}\n")

    result = _run(["ingest"], "n\n")

    assert "different column mapping" in result.output, result.output
    assert "result_date" in result.output
    assert "--force" in result.output


def test_a_setting_pointing_at_nothing_is_one_message_not_one_per_patient(cohort):
    """A stage reads the column map, the model, the recipes and the allow-list once for
    the whole cohort, so a missing one fails identically for every patient. The
    per-patient loop reported it as "skipping <patient>", or "1 of 500 failed", and
    offered to re-run that patient — a remedy that cannot work, multiplied by the
    cohort."""
    settings = cohort / "junior_study.yaml"
    settings.write_text(settings.read_text() + f"chart_columns_file: {cohort / 'gone.yaml'}\n")

    result = _run(["ingest"], "y\n")

    assert result.exit_code != 0
    assert "column map is not there" in result.output, result.output
    assert "chart_columns_file:" in result.output, "the setting to fix is unnamed"
    assert "skipping" not in result.output, "blamed a patient for a project setting"
    assert "--patient" not in result.output, "offered a per-patient remedy"
    # Naive conjugation produced "embeded" and "retrieveed"; the sentence avoids it.
    assert "so nothing ran" in result.output


def test_a_missing_model_folder_is_caught_before_any_patient(cohort):
    settings = cohort / "junior_study.yaml"
    settings.write_text(
        settings.read_text() + f"encoder:\n  model_id: {cohort / 'no_model'}\n")
    assert _run(["ingest"], "y\n").exit_code == 0

    result = _run(["embed"], "y\n")

    assert "embedding model folder is not there" in result.output, result.output
    assert "patient(s) failed" not in result.output, "reported as a patient failure"


def test_what_you_type_during_a_long_stage_is_not_run_as_a_command(monkeypatch):
    """A stage can take minutes and draws its progress in place, so a keypress lands in
    the terminal buffer — and the next thing to read stdin is the prompt. Enter pressed
    out of impatience is harmless; a half-typed word arriving as a command is not."""
    import apps_and_interfaces.interactive_session as session

    flushed: list[int] = []
    monkeypatch.setattr(session, "is_interactive", lambda: True)

    class _Termios:
        error = OSError
        TCIFLUSH = 1

        @staticmethod
        def tcflush(_stream, how):
            flushed.append(how)

    monkeypatch.setitem(__import__("sys").modules, "termios", _Termios)

    session._forget_what_was_typed_while_it_ran()

    assert flushed == [_Termios.TCIFLUSH], "type-ahead was left for the next prompt"


def test_a_scripted_session_keeps_its_piped_input(monkeypatch):
    """Draining a pipe would eat the rest of the script."""
    import apps_and_interfaces.interactive_session as session

    flushed: list[int] = []
    monkeypatch.setattr(session, "is_interactive", lambda: False)

    class _Termios:
        error = OSError
        TCIFLUSH = 1

        @staticmethod
        def tcflush(_stream, how):
            flushed.append(how)

    monkeypatch.setitem(__import__("sys").modules, "termios", _Termios)

    session._forget_what_was_typed_while_it_ran()

    assert flushed == []


def test_the_prompt_discards_type_ahead_after_every_command():
    """Wired into the loop, not merely callable."""
    import inspect

    import apps_and_interfaces.interactive_session as session

    body = inspect.getsource(session._loop)
    assert "_forget_what_was_typed_while_it_ran()" in body
    assert body.index("_dispatch(group, arguments)") < body.index(
        "_forget_what_was_typed_while_it_ran()"
    ), "must discard AFTER the command has run, not before"


def test_ingesting_without_a_column_map_asks_a_different_question(monkeypatch):
    """Without a map the question says so and names the way out.

    Carrying that only in the default — [y/N] rather than [Y/n] — asks somebody to
    notice a capital letter and infer a recommendation from it. They will not, and
    ingest is the point of no return for this choice: it resolves each chart field
    against a file's real header and records the answer per file, and every later stage
    reads what was recorded rather than deciding again."""
    from pathlib import Path
    from unittest import mock

    from apps_and_interfaces import command_line_interface as cli

    asked = {}

    def _capture(text, default=False):
        asked["text"], asked["default"] = text, default
        return False

    with mock.patch.object(cli, "is_interactive", lambda: True), \
         mock.patch.object(cli.click, "confirm", _capture):
        cli._confirm_column_mapping({"source_root": "/charts"}, Path("/tmp/run_x"), 2)
        assert "Are you sure" in asked["text"]
        assert asked["default"] is False
        without_a_map = asked["text"]

        cli._confirm_column_mapping(
            {"source_root": "/charts", "chart_columns_file": "map.yaml"},
            Path("/tmp/run_x"), 2,
        )
        assert asked["default"] is True
        assert "Are you sure" not in asked["text"], (
            "an ordinary confirmation should not sound like a warning"
        )
        assert without_a_map != asked["text"]


def test_it_names_the_command_that_writes_a_column_map(capsys, monkeypatch):
    """A warning that does not say what to do instead is a warning people learn to
    press through."""
    from pathlib import Path
    from unittest import mock

    from apps_and_interfaces import command_line_interface as cli

    with mock.patch.object(cli, "is_interactive", lambda: True), \
         mock.patch.object(cli.click, "confirm", lambda *a, **k: False):
        cli._confirm_column_mapping({"source_root": "/charts"}, Path("/tmp/run_x"), 2)

    assert "junior columns" in capsys.readouterr().out
