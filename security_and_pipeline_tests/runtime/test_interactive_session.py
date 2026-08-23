"""The `junior` prompt is a way of reaching the CLI, not a second CLI.

Every line typed at the prompt goes through the same click group as `junior <command>`,
so a command added to the CLI is reachable from the prompt with no extra wiring. These
tests pin that, and pin that one bad command does not end the session — a prompt that
exits on a typo is worse than no prompt.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main
from apps_and_interfaces.interactive_session import start


def _run_session(monkeypatch, capsys, lines: list[str]) -> str:
    """Drive a session over `lines`, then quit, and return everything it printed."""
    remaining = iter([*lines, "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))
    start(main, "0.0.1")
    return capsys.readouterr().out


def test_prompt_dispatches_into_the_same_command_group(monkeypatch, capsys):
    out = _run_session(monkeypatch, capsys, ["run --help"])
    # This help text belongs to the `run` command, not to a prompt-local copy of it.
    assert "PATIENT_FOLDERS" in out
    assert "--config" in out


# The `help` listing is how these tests tell a live session from a dead one. Derived
# from the CLI's own phase titles rather than spelled out, so renaming a section does
# not read as the session having died.
def _help_marker() -> str:
    from apps_and_interfaces.command_line_interface import PIPELINE_PHASES

    return next(iter(PIPELINE_PHASES)).upper()


def test_unknown_command_does_not_end_the_session(monkeypatch, capsys):
    out = _run_session(monkeypatch, capsys, ["definitely-not-a-command", "help"])
    assert _help_marker() in out, "session died before reaching the second line"


def test_unbalanced_quotes_do_not_end_the_session(monkeypatch, capsys):
    out = _run_session(monkeypatch, capsys, ['run "oops', "help"])
    # Plain english, not shlex's "No closing quotation" — the person at this prompt
    # is a researcher, and the library's wording names a parser they never invoked.
    assert "quote" in out.lower()
    assert _help_marker() in out, "session died before reaching the second line"


def test_a_raising_command_does_not_end_the_session(monkeypatch, capsys):
    def _boom(*_a, **_k):
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(main, "main", _boom)
    out = _run_session(monkeypatch, capsys, ["run", "help"])
    assert "stage exploded" in out, "the reason was swallowed"
    # The exception class belongs in the log, not on a researcher's screen.
    assert "RuntimeError" not in out, "leaked a Python exception class to the operator"
    assert _help_marker() in out, "a failing command took the session down with it"


def test_every_advertised_command_exists_in_the_group():
    """The prompt's command list must not name something the CLI does not have."""
    advertised = {
        "new-project",
        "ingest", "embed", "index", "extract",
        "export-metadata",
        "workbench",
    }
    missing = advertised - set(main.commands)
    assert not missing, f"prompt advertises commands the CLI does not define: {sorted(missing)}"


def test_the_cli_advertises_only_what_a_run_needs():
    """`junior --help` is the run surface: set up, stages, close out, open the app.

    Analysis commands still exist but are hidden, so the listed surface stays small
    enough that a new operator can read it top to bottom."""
    from apps_and_interfaces.command_line_interface import APP_SIDE_COMMANDS

    visible = {name for name, command in main.commands.items() if not command.hidden}
    assert visible.isdisjoint(APP_SIDE_COMMANDS), (
        f"app-side commands are showing in --help: {sorted(visible & APP_SIDE_COMMANDS)}"
    )
    for stage in ("ingest", "embed", "index", "extract"):
        assert stage in visible, f"{stage} must stay visible — it is how a run advances"


def test_hidden_commands_still_run():
    """Hiding is a help-listing decision, not a removal.

    The cluster runbook and the workbench notes invoke these by name, and a cluster shell
    may be the only thing available; they must keep working."""
    from apps_and_interfaces.command_line_interface import APP_SIDE_COMMANDS

    for name in APP_SIDE_COMMANDS:
        assert name in main.commands, f"{name} was removed, not hidden"
        result = CliRunner().invoke(main, [name, "--help"])
        assert result.exit_code == 0, f"`junior {name} --help` failed: {result.output}"


def test_typing_the_program_name_at_its_own_prompt_works(monkeypatch, capsys):
    """Every error message and doc says `junior status`, and habit says it too. Inside
    the session the program name is implied, so it must be dropped, not rejected."""
    out = _run_session(monkeypatch, capsys, ["junior --version"])
    assert "No such command" not in out, out
    assert "0.0.1" in out


def test_the_bare_program_name_shows_the_list(monkeypatch, capsys):
    out = _run_session(monkeypatch, capsys, ["junior"])
    assert _help_marker() in out


class TestWhatPeopleActuallyType:
    """Each of these is a real mistyping observed at the prompt, not a hypothetical.

    The rule for adding one: it must be unambiguous — a normalization may never turn
    one valid command into a different valid command."""

    def _normalized(self, line: str) -> list[str] | None:
        import shlex

        from apps_and_interfaces.interactive_session import _normalize

        return _normalize(shlex.split(line), main)

    def test_placeholder_brackets_are_stripped(self):
        """The footer used to print `<command> --help`, so people copied the brackets."""
        assert self._normalized("<new-project> --help") == ["new-project", "--help"]

    def test_the_program_name_is_dropped(self):
        assert self._normalized("junior status") == ["status"]

    def test_a_space_instead_of_a_hyphen_still_resolves(self):
        assert self._normalized("new project --help") == ["new-project", "--help"]

    def test_a_command_and_its_argument_are_not_welded_together(self):
        """`new project` joins only because `new-project` exists. A real command
        followed by a word must not become `columns-somefolder`."""
        assert self._normalized("columns somefolder") == ["columns", "somefolder"]

    def test_a_step_number_runs_that_step(self):
        """The listing numbers the stages; typing the number is the obvious next move."""
        from apps_and_interfaces.command_line_interface import (
            PIPELINE_STAGE_DESCRIPTIONS,
        )

        first_stage = next(iter(PIPELINE_STAGE_DESCRIPTIONS))
        assert self._normalized("1") == [first_stage]
        assert self._normalized("1.") == [first_stage]

    def test_a_step_number_keeps_its_flags(self):
        assert self._normalized("1 --patient P1")[1:] == ["--patient", "P1"]

    def test_a_number_that_is_not_a_step_is_refused_not_guessed(self):
        assert self._normalized("9") is None

    def test_help_survives_the_program_name(self, monkeypatch, capsys):
        """`junior help` strips to `help`, which used to reach click as a command and
        get answered with "Did you mean 'health'?"."""
        out = _run_session(monkeypatch, capsys, ["junior help"])
        assert _help_marker() in out
        assert "health" not in out


def test_back_at_the_top_level_explains_itself(monkeypatch, capsys):
    """`back` is a real word inside a command's questions, so at the prompt it should
    say there is nowhere to go — not "No such command", which reads as a typo."""
    out = _run_session(monkeypatch, capsys, ["back"])
    assert "top level" in out
    assert "No such command" not in out


def test_an_unknown_command_points_at_help_that_works_here(monkeypatch, capsys):
    """Click's own advice is `Try 'junior --help'` — a shell command, useless to
    someone already at the prompt."""
    out = _run_session(monkeypatch, capsys, ["nonsense"])
    assert "Type help" in out


def test_ctrl_c_cancels_the_command_not_the_session(monkeypatch, capsys):
    """KeyboardInterrupt is a BaseException, so _dispatch's handlers never saw it and
    it escaped the loop — killing the prompt in the middle of a long run."""
    def _interrupt(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "main", _interrupt)
    out = _run_session(monkeypatch, capsys, ["extract", "help"])
    assert "cancelled" in out
    assert _help_marker() in out, "Ctrl-C took the session down with the command"


def test_the_stages_are_one_ordered_list(monkeypatch, capsys):
    """Working on a cohort is one sequence. Extraction was its own heading for a while,
    which read as a separate mode rather than the last step of the same job."""
    from apps_and_interfaces.command_line_interface import PIPELINE_PHASES

    (title, stages), = PIPELINE_PHASES.items()
    assert list(stages) == ["ingest", "embed", "index", "extract"]

    out = _run_session(monkeypatch, capsys, ["help"])
    assert title.upper() in out, "the prompt lost the stage heading"
    # Every stage sits under that one heading, in order. The heading shares a line with
    # the first stage rather than sitting above it, so what marks the group is the
    # heading being ON that first row — not an offset comparison against it.
    positions = [_line_for(stage, out) for stage in stages]
    assert positions == sorted(positions), "the stages are listed out of order"
    assert title.upper() in _row_for(list(stages)[0], out), (
        "the stage heading no longer marks where the group starts"
    )
    for later_stage in list(stages)[1:]:
        assert title.upper() not in _row_for(later_stage, out), (
            f"the heading is repeated on {later_stage}"
        )


def test_workbench_is_the_only_name():
    """`demo` was the command's old name, kept while run_demo.sh and the
    double-clickable app said it. Those are gone, and a hidden alias to a name
    nothing prints anymore is a trap for old muscle memory to land in silently."""
    from apps_and_interfaces.command_line_interface import main

    assert "workbench" in main.commands
    assert "demo" not in main.commands


def test_stage_numbering_matches_the_order_stages_run_in():
    """The number in each stage's own help must match its position in the pipeline."""
    from apps_and_interfaces.command_line_interface import (
        PIPELINE_STAGE_DESCRIPTIONS,
        STAGES,
    )

    # The numbered stages need not be all of STAGES — `retrieve` is dispatchable but
    # runs inside extract — but they must appear in execution order, or the numbers
    # would tell an operator to run them in an order the pipeline does not support.
    execution_order = list(STAGES)
    numbered = list(PIPELINE_STAGE_DESCRIPTIONS)
    assert numbered == [name for name in execution_order if name in set(numbered)], (
        f"numbered stages {numbered} are not in execution order {execution_order}"
    )

    for position, name in enumerate(numbered, 1):
        assert main.commands[name].help.startswith(f"Step {position}:"), (
            f"`{name}` is stage {position} but its help says {main.commands[name].help!r}"
        )

    # Every dispatchable stage still gets a command, numbered or not.
    for name in execution_order:
        assert name in main.commands, f"{name} is in STAGES but has no command"


def _row_for(command: str, output: str) -> str:
    """The listing row for a command. Fails the test if it is not listed at all."""
    for line in output.splitlines():
        fields = line.split()
        # The command name is its own field, so `extract` does not match `export-metadata`.
        if command in fields:
            return line
    raise AssertionError(f"{command!r} is not listed in:\n{output}")


def _line_for(command: str, output: str) -> int:
    return output.find(_row_for(command, output))


def test_prompt_lists_every_command_with_the_cli_numbering(monkeypatch, capsys):
    """The prompt renders from the CLI's tables, so it must show all of them, and
    show each stage under the number that stage actually has."""
    from apps_and_interfaces.command_line_interface import (
        APP_COMMANDS,
        PIPELINE_STAGE_DESCRIPTIONS,
        SHARE_COMMANDS,
        START_COMMANDS,
    )

    out = _run_session(monkeypatch, capsys, ["help"])

    # Assert on the row's fields rather than an exact format, so restyling the
    # listing does not break these — only dropping or misnumbering a command does.
    for position, (name, description) in enumerate(PIPELINE_STAGE_DESCRIPTIONS.items(), 1):
        row = _row_for(name, out)
        fields = row.split()
        assert str(position) in fields, f"{name} is not shown as step {position}: {row!r}"
        assert fields.index(str(position)) < fields.index(name), (
            f"{name}'s step number does not come before it: {row!r}"
        )
        assert description in row

    for table in (START_COMMANDS, APP_COMMANDS, SHARE_COMMANDS):
        for name, description in table.items():
            assert description in _row_for(name, out)


def test_bare_junior_opens_the_prompt_not_the_app(monkeypatch):
    """Regression: bare `junior` used to exec the Shiny launcher instead."""
    opened: list[str] = []
    monkeypatch.setattr(
        "apps_and_interfaces.interactive_session.start",
        lambda group, version: opened.append("prompt"),
    )
    monkeypatch.setattr("os.execvp", lambda *a, **k: opened.append("app"))
    CliRunner().invoke(main, [])
    assert opened == ["prompt"], f"bare `junior` did {opened or 'nothing'}"


def test_stage_output_is_quiet_by_default_but_the_log_keeps_everything(tmp_path):
    """A stage logs one event per source file per patient. Nineteen tables across a
    cohort buries the progress line; the durable record must not pay for that."""
    from jr_pipeline.runtime_infrastructure.json_event_logging import (
        clear_run_log_file,
        get_logger,
        set_console_quiet,
        set_run_log_file,
    )

    log_file = tmp_path / "run_log.jsonl"
    set_run_log_file(log_file)
    try:
        set_console_quiet(True)
        get_logger("t").info("routine_event", extra_={"stem": "labs"})
        get_logger("t").warning("something_worth_seeing")
    finally:
        set_console_quiet(False)
        clear_run_log_file()

    written = log_file.read_text(encoding="utf-8")
    assert "routine_event" in written, "quieting the terminal cost the durable record"
    assert "something_worth_seeing" in written


def test_every_stage_explains_itself():
    """A stage that prints only a clock leaves the operator guessing what it is doing
    to their data."""
    from apps_and_interfaces.command_line_interface import (
        STAGE_EXPLANATIONS,
        STAGES,
    )

    for stage in STAGES:
        assert stage in STAGE_EXPLANATIONS, f"{stage} starts with no explanation"
        assert len(STAGE_EXPLANATIONS[stage]) > 30, stage


class TestMoreOfWhatPeopleType:
    """A second round, from driving the prompt through a real terminal."""

    def _normalized(self, line: str) -> list[str] | None:
        import shlex

        from apps_and_interfaces.interactive_session import _normalize

        return _normalize(shlex.split(line), main)

    def test_a_capitalised_command_still_runs(self):
        """The session already matches HELP and QUIT regardless of case. Punishing a
        capitalised command name teaches a rule that only holds half the time."""
        assert self._normalized("Ingest --patient P1") == ["ingest", "--patient", "P1"]

    def test_dash_h_means_help(self):
        assert self._normalized("ingest -h") == ["ingest", "--help"]

    def test_case_folding_cannot_invent_a_command(self):
        """Only folds when the lowercase spelling is real, so an argument that happens
        to be capitalised is never turned into something else."""
        assert self._normalized("NotACommand")[0] == "NotACommand"


def test_workbench_does_not_replace_the_session(monkeypatch, tmp_path):
    """`os.execvp` replaces this process, so launching the app from the prompt ended
    the session for good — `quit` was never reached and there was no way back."""
    # The command refuses before it spawns anything when the workbench is not installed,
    # so without the `app` extra this asserts against a step that never runs.
    pytest.importorskip("shiny", reason="needs the app extra: pip install -e '.[app]'")

    import os as os_module

    from apps_and_interfaces import command_line_interface as cli

    called: list[str] = []
    monkeypatch.setattr(os_module, "execvp", lambda *a, **k: called.append("exec"))
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.append("subprocess"))
    monkeypatch.setattr(cli, "_in_interactive_session", lambda: True)
    monkeypatch.setattr(
        cli, "_resolve_repo_root",
        lambda: __import__("pathlib").Path(__file__).resolve().parents[2],
    )

    CliRunner().invoke(main, ["workbench"])
    assert "exec" not in called, "launching the app from the prompt still kills the session"
    assert "subprocess" in called


def test_a_warning_reaches_the_terminal_as_a_sentence(tmp_path, capsys):
    """Quieting the routine events left warnings printing as raw structlog JSON —
    so the only lines an operator was left reading were the machine-formatted ones."""
    from jr_pipeline.runtime_infrastructure.json_event_logging import (
        get_logger,
        set_console_quiet,
    )

    set_console_quiet(True)
    try:
        get_logger("t").warning("excel_mangled_column_suspected", extra_={
            "stem": "epicprom", "column": "float_answer",
            "hint": "values look like Excel scientific notation",
        })
    finally:
        set_console_quiet(False)

    reported = capsys.readouterr().err
    assert "values look like Excel scientific notation" in reported
    assert '{"' not in reported, "a warning still printed as raw JSON"
    assert "epicprom" in reported, "dropped the context that says which file"


def test_every_stage_says_what_it_did_and_what_follows():
    """Ingest is fast and produces tables, which reads like it might have been the
    whole job — so a stage has to state what it did NOT do, and what comes next."""
    from apps_and_interfaces.command_line_interface import (
        _STAGE_COMPLETION,
        PIPELINE_STAGE_DESCRIPTIONS,
        STAGES,
    )

    for stage in STAGES:
        assert stage in _STAGE_COMPLETION, f"{stage} ends without saying what it did"
    described, follows = _STAGE_COMPLETION["ingest"]
    assert "nothing has been embedded" in described
    assert follows == "embed"
    # Every named follow-on is a real stage, so the advice cannot go stale.
    for _, follows in _STAGE_COMPLETION.values():
        assert follows is None or follows in PIPELINE_STAGE_DESCRIPTIONS
