"""`junior run` resumes. `junior run --new-run` starts over. That is the whole model.

Starting over used to require inventing a run id, which is an implementation
identifier: `junior run --run-id stage-fix-2`. Worse, an invented name was not a run
the tool could find again. A bare `junior run` continues the NEWEST run and only
recognises generated ids (RUN_ID_PATTERN), so a hand-named run fell out of resume
entirely and every later command had to keep repeating --run-id.

So --run-id stops being part of the routine loop and becomes what it always was: a way
to name one particular run. Starting over is a flag, and the id is generated.
"""
from __future__ import annotations

from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main
from jr_pipeline.runtime_infrastructure.project_context import (
    RUN_ID_PATTERN,
    resolve_run_id,
)


def test_a_fresh_run_gets_an_id_resume_can_find_again():
    """The failure this closes: a hand-named run that `junior run` will not resume."""
    minted = resolve_run_id({}, None, start_fresh=True)

    assert RUN_ID_PATTERN.fullmatch(minted), (
        f"{minted!r} is not a shape `junior run` would ever continue"
    )


def test_starting_over_outranks_a_pinned_run_id():
    """`run_id:` in a config and JR_RUN_ID both say WHICH run. --new-run says a new
    one. A request to start over that landed back in a pinned run is the worst case:
    it looks like it started over and appends to the old one."""
    assert resolve_run_id({"run_id": "pinned"}) == "pinned"
    assert resolve_run_id({"run_id": "pinned"}, start_fresh=True) != "pinned"


def test_two_fresh_runs_are_two_runs():
    assert resolve_run_id({}, None, start_fresh=True) != resolve_run_id({}, None, start_fresh=True)


def test_asking_for_both_at_once_is_refused():
    """--new-run wants a run that does not exist, --run-id one that does. Honouring
    either silently would be wrong half the time."""
    result = CliRunner().invoke(main, ["ingest", "--new-run", "--run-id", "20990101_000000_ab"])

    assert result.exit_code != 0
    assert "--new-run starts a run" in result.output


def test_every_stage_can_start_a_run():
    """Each stage can start one, so a rebuild is driven a stage at a time and you see
    where it stops."""
    runner = CliRunner()
    for command in ("ingest", "embed", "index", "extract"):
        assert "--new-run" in runner.invoke(main, [command, "--help"]).output, command


def test_run_does_not_start_one():
    """`junior run` walks every stage in one go, which is the wrong shape for a
    rebuild: the expensive part is embedding, and one command hides which stage it
    stopped at. Starting over is driven from `ingest --new-run` instead."""
    assert "--new-run" not in CliRunner().invoke(main, ["run", "--help"]).output


def test_run_id_is_no_longer_advertised_as_the_way_to_start_over():
    """It read `Re-enter an existing run instead of minting a new id`, which taught
    exactly the workflow being retired."""
    out = CliRunner().invoke(main, ["run", "--help"]).output

    assert "minting a new id" not in out
    assert "Rarely needed" in out
