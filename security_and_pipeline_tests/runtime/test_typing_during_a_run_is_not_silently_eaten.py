"""Typing during a stage is discarded — and now says so.

The discard is right: a stage runs for minutes, the terminal echoes every keypress into
the middle of a progress line, and those keystrokes are still in the input buffer when
the stage ends. The next thing to read stdin is the prompt, which would take a
half-typed word as a command.

What was wrong is that it happened in silence. The keystrokes appeared on screen,
because the terminal echoed them, and then vanished with nothing said. From the outside
that is identical to the session being wedged — which is exactly how it got reported:
"why could I type all that and it locked me out".
"""
from __future__ import annotations

import sys

import pytest

from apps_and_interfaces import interactive_session


@pytest.fixture
def _at_a_terminal(monkeypatch):
    monkeypatch.setattr(interactive_session, "is_interactive", lambda: True)
    flushed: list[int] = []
    fake_termios = type(sys)("termios")
    fake_termios.TCIFLUSH = 1
    fake_termios.error = OSError
    fake_termios.tcflush = lambda _stream, which: flushed.append(which)
    monkeypatch.setitem(sys.modules, "termios", fake_termios)
    return flushed


def _pretend_input_is_waiting(monkeypatch, waiting: bool):
    fake_select = type(sys)("select")
    fake_select.select = lambda r, w, x, t: ([sys.stdin] if waiting else [], [], [])
    monkeypatch.setitem(sys.modules, "select", fake_select)


def test_it_says_so_when_it_throws_something_away(_at_a_terminal, monkeypatch, capsys):
    _pretend_input_is_waiting(monkeypatch, True)

    interactive_session._forget_what_was_typed_while_it_ran()

    out = capsys.readouterr().out
    assert "ignored what was typed" in out
    # Why there was no second command to run, which is the actual question being asked.
    assert "one command at a time" in out
    assert _at_a_terminal, "the input was reported but not actually discarded"


def test_it_stays_quiet_when_nothing_was_typed(_at_a_terminal, monkeypatch, capsys):
    """The common case is a stage finishing with nobody touching the keyboard. A note
    every time would be noise, and noise is what stops notes being read."""
    _pretend_input_is_waiting(monkeypatch, False)

    interactive_session._forget_what_was_typed_while_it_ran()

    assert capsys.readouterr().out == ""
    assert _at_a_terminal, "the buffer should still be flushed"


def test_it_still_discards_when_it_cannot_tell_what_is_waiting(
    _at_a_terminal, monkeypatch, capsys
):
    """The discard is the safety property; the message is a courtesy. If select cannot
    answer, the keystrokes still go."""
    fake_select = type(sys)("select")

    def _no(*_a, **_k):
        raise OSError("no select on this stream")

    fake_select.select = _no
    monkeypatch.setitem(sys.modules, "select", fake_select)

    interactive_session._forget_what_was_typed_while_it_ran()

    assert _at_a_terminal, "input was left in the buffer for the next prompt to read"
    assert capsys.readouterr().out == ""


def test_nothing_happens_when_not_at_a_terminal(monkeypatch, capsys):
    """Under a pipe or a test there is no terminal to flush and nobody typing."""
    monkeypatch.setattr(interactive_session, "is_interactive", lambda: False)

    interactive_session._forget_what_was_typed_while_it_ran()

    assert capsys.readouterr().out == ""


# --- said while it is still running, not eight minutes later ------------------------

def test_the_ticker_notices_typing_without_reading_it(monkeypatch):
    """The notice has to arrive while somebody is still sitting there waiting on it.

    And it must NOT consume the keystrokes: a stage can be interrupted, and the Ctrl-C
    confirmation reads stdin for its y/n. A ticker draining the buffer once a second
    would race it and eat the answer to "stop this run?"."""
    import inspect

    from apps_and_interfaces import command_line_interface as cli

    source = inspect.getsource(cli._stage_monitor)

    assert "_somebody_is_typing" in source, "the ticker does not notice typing at all"
    assert "one command at a time" in source
    # The dangerous verbs. select() asks; read/readline/tcflush would take.
    for consuming in ("sys.stdin.read", "sys.stdin.readline", "tcflush"):
        assert consuming not in source, (
            f"the ticker calls {consuming}, which would race the Ctrl-C confirmation "
            "for its keypress"
        )


def test_it_says_it_once_not_every_second(monkeypatch):
    """A line a second, for minutes, would bury the progress display it is printed
    beside."""
    import inspect

    from apps_and_interfaces import command_line_interface as cli

    source = inspect.getsource(cli._stage_monitor)
    assert "said_it_is_busy = True" in source
    assert "if not said_it_is_busy" in source
