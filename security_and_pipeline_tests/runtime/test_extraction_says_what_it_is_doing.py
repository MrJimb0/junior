"""Extraction reports per variable, so a slow run is distinguishable from a stuck one.

It is the only stage measured in minutes and was the only one that printed nothing until
a patient finished. Twenty minutes of silence reads as a hang, and did.
"""
from __future__ import annotations

from jr_pipeline.runtime_infrastructure.extraction_progress import (
    report_nothing_to_do,
    report_patient,
    report_variables,
)


def _capture():
    lines: list[str] = []
    return lines, report_variables(write=lines.append, animated=False)


def test_a_variable_is_announced_when_it_starts_not_when_it_ends():
    """The whole point. A line printed only on completion arrives after the wait it was
    meant to explain."""
    lines, show = _capture()

    show("stage", "running", None)

    assert any("stage" in line and "running" in line for line in lines)


def test_the_five_states_are_distinguishable():
    lines, show = _capture()

    show("date_of_birth", "already complete", None)
    show("stage", "running", None)
    show("stage", "done", 222.0)
    show("date_of_diagnosis", "retrying", None)
    show("date_of_diagnosis", "failed", 41.0)

    printed = "\n".join(lines)
    for state in ("already complete", "running", "done", "retrying", "failed"):
        assert state in printed, state
    assert "3m 42s" in printed
    assert "41s" in printed


def test_already_complete_is_not_called_cached():
    """Different things. Already complete means no work was needed. Cached means the
    work ran and a stored response stood in for one model call. Calling a resumed
    variable "cached" would say the model cache is what makes resume correct — it is
    not, and a reader who believes that will clear the cache to force a redo and be
    surprised when nothing changes."""
    lines, show = _capture()

    show("stage", "already complete", None)

    assert "cached" not in "\n".join(lines)


def test_a_failure_names_the_variable_that_failed():
    """Enough to act on: which variable, on which patient. The patient comes from the
    header above it."""
    lines: list[str] = []
    report_patient(2, 2, "Patient_B", write=lines.append)
    show = report_variables(write=lines.append, animated=False)
    show("stage", "failed", 12.0)

    printed = "\n".join(lines)
    assert "Patient_B" in printed
    assert "stage" in printed and "failed" in printed


def test_a_patient_with_nothing_to_do_says_so():
    """Otherwise a resumed patient prints a header with nothing under it, which reads
    as the display having broken rather than as there being no work."""
    lines: list[str] = []

    report_nothing_to_do(write=lines.append)

    assert "nothing to do" in lines[0]


def test_no_values_are_printed():
    """Extraction output is patient data and a terminal outlives the run in scrollback
    and cluster logs. The display takes a name and a state; there is no argument it
    could print a value from."""
    import inspect

    source = inspect.getsource(report_variables)

    assert "data" not in source.replace("dataclass", "")
    lines, show = _capture()
    show("stage", "done", 1.0)
    assert lines == ["      stage                 done              1s"]


def test_both_commands_use_this_display():
    """Two displays would drift, and the one nobody watched would be the one that lied."""
    import inspect

    from apps_and_interfaces import command_line_interface as cli
    from jr_pipeline.runtime_infrastructure import cohort_runner

    assert "report_variables" in inspect.getsource(cli._run_extract)
    assert "report_variables" in inspect.getsource(cohort_runner._run_extract)


def test_a_repeat_failure_is_not_called_the_same_thing_as_a_new_one():
    """Extraction is deterministic — greedy decoding, and a response cache keyed on the
    prompt — so a variable that fails on unchanged evidence fails the same way every
    time. Calling that "failed" invites another retry that will do the same, instantly,
    which is exactly how it gets read as the tool being broken."""
    lines, show = _capture()

    show("stage", "failed again", 0.4)

    assert "failed again" in "\n".join(lines)


def test_the_states_a_variable_can_end_in():
    """A variable that reports a state not in this set leaves its line hanging open on
    a terminal, because the next line overwrites it."""
    from jr_pipeline.runtime_infrastructure.extraction_progress import (
        _STATES_THAT_END_A_VARIABLE,
    )

    assert _STATES_THAT_END_A_VARIABLE == {
        "done", "failed", "failed again", "already complete",
    }


def test_on_a_terminal_the_finishing_state_overwrites_the_running_line():
    """One line per variable on a terminal: the opening state holds its line open and
    the finishing state returns to column 0 and overwrites it. The first version
    printed the opener WITH a newline, so the overwrite sequence landed on the next,
    blank line and every variable rendered twice with a stale 'running' above it."""
    from jr_pipeline.runtime_infrastructure.extraction_progress import report_variables

    lines: list[str] = []
    show = report_variables(write=lines.append, animated=True)

    show("date_of_birth", "running", None)
    show("date_of_birth", "done", 5.0)
    show("stage", "running", None)
    show("stage", "retrying", None)
    show("stage", "failed", 8.0)

    assert not lines[0].startswith("\r"), "the first opener has nothing to overwrite"
    assert lines[1].startswith("\r\x1b[K"), "done did not return to overwrite running"
    assert not lines[2].startswith("\r"), "a fresh variable starts on its own line"
    assert lines[3].startswith("\r\x1b[K"), "retrying overwrites running in place"
    assert lines[4].startswith("\r\x1b[K"), "failed overwrites retrying in place"


def test_piped_output_keeps_both_lines_and_no_control_codes():
    """A log wants the start time, and control codes in a SLURM logfile are noise."""
    from jr_pipeline.runtime_infrastructure.extraction_progress import report_variables

    lines: list[str] = []
    show = report_variables(write=lines.append, animated=False)

    show("date_of_birth", "running", None)
    show("date_of_birth", "done", 5.0)

    assert len(lines) == 2
    assert not any("\r" in line or "\x1b" in line for line in lines)
