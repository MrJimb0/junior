"""Ctrl-C can stop a stage, but not by accident.

A cohort can run for hours. Finding out two patients in that the settings were wrong
should not mean waiting it out, so stopping has to stay possible. But Ctrl-C is also what
a hand reaches for reflexively, and it is one keystroke away from discarding an afternoon
of GPU time — so it asks first, and the default answer is to keep going.

Nothing here relies on a real terminal. The handler is invoked the way the signal module
would invoke it, and `click.confirm` is answered by a stub.
"""
from __future__ import annotations

import signal

import pytest

from apps_and_interfaces.command_line_interface import _confirm_before_stopping


@pytest.fixture
def at_a_keyboard(monkeypatch):
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: True
    )


def _interrupt_now():
    """Call whatever SIGINT handler is currently installed, as the runtime would."""
    handler = signal.getsignal(signal.SIGINT)
    assert callable(handler), "no handler installed, so Ctrl-C would kill the stage"
    handler(signal.SIGINT, None)


def test_declining_lets_the_run_carry_on(monkeypatch, capsys, at_a_keyboard):
    monkeypatch.setattr("click.confirm", lambda *a, **k: False)

    with _confirm_before_stopping("ingest", 9):
        _interrupt_now()          # must NOT raise
        carried_on = True

    assert carried_on
    assert "carrying on" in capsys.readouterr().err


def test_accepting_stops_the_run(monkeypatch, at_a_keyboard):
    monkeypatch.setattr("click.confirm", lambda *a, **k: True)

    with pytest.raises(KeyboardInterrupt):
        with _confirm_before_stopping("extract", 9):
            _interrupt_now()


def test_the_question_says_what_would_be_lost(monkeypatch, at_a_keyboard):
    asked: list[str] = []
    monkeypatch.setattr("click.confirm", lambda text, **k: asked.append(text) or False)

    with _confirm_before_stopping("extract", 40):
        _interrupt_now()

    assert "Stop extract?" in asked[0]
    assert "40 patient(s)" in asked[0], "the question must say how much is in flight"


def test_the_default_answer_is_to_keep_going(monkeypatch, at_a_keyboard):
    defaults: list[bool] = []

    def _confirm(_text, default=True, **_k):
        defaults.append(default)
        return False

    monkeypatch.setattr("click.confirm", _confirm)

    with _confirm_before_stopping("embed", 9):
        _interrupt_now()

    assert defaults == [False], "Enter must not discard a run"


def test_a_second_ctrl_c_while_the_question_is_up_still_stops(monkeypatch, at_a_keyboard):
    """Answering is itself interruptible — the default handler is restored while the
    question is up, so a stuck run is never unkillable."""
    during: list[object] = []
    monkeypatch.setattr(
        "click.confirm",
        lambda *a, **k: during.append(signal.getsignal(signal.SIGINT)) or False,
    )

    with _confirm_before_stopping("embed", 9):
        _interrupt_now()

    assert during == [signal.SIG_DFL]


def test_the_handler_is_put_back_afterwards(at_a_keyboard):
    before = signal.getsignal(signal.SIGINT)

    with _confirm_before_stopping("ingest", 1):
        pass

    assert signal.getsignal(signal.SIGINT) is before


def test_a_scripted_run_dies_on_sigint_without_a_question(monkeypatch):
    """A SLURM task or a pipe has nobody to answer, and the scheduler is waiting."""
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.is_interactive", lambda: False
    )
    before = signal.getsignal(signal.SIGINT)

    with _confirm_before_stopping("ingest", 500):
        assert signal.getsignal(signal.SIGINT) is before, "installed a prompt on a cluster"
