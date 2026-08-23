"""A command that asks several questions needs a way out of each one.

Without `back`, mistyping the second of three answers means either finishing a command
you no longer want or killing the session with Ctrl-C. Without `cancel`, an operator
who realises they are in the wrong place has no way to leave without writing something.

The subtle part is the defaults: a later question suggests a path built from an earlier
answer, so going back and changing that answer has to refresh the suggestion rather
than offer one derived from the value just replaced.
"""
from __future__ import annotations

import pytest

from apps_and_interfaces.console_questions import (
    Question,
    ask_one,
    ask_sequence,
)


@pytest.fixture
def at_a_terminal(monkeypatch):
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive", lambda: True
    )


def _typing(monkeypatch, lines: list[str]) -> None:
    remaining = iter(lines)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))


def _name_then_root() -> list[Question]:
    return [
        Question("name", "Name?", lambda _a: "default_name"),
        Question("root", "Root?", lambda _a: "./data"),
        Question("folder", "Folder?", lambda a: f"{a.get('root', '?')}/in/{a.get('name', '?')}"),
    ]


def test_answers_come_back_in_order(monkeypatch, at_a_terminal):
    _typing(monkeypatch, ["study", "/data", "/charts"])
    assert ask_sequence(_name_then_root()) == {
        "name": "study", "root": "/data", "folder": "/charts"
    }


def test_an_empty_answer_takes_the_default(monkeypatch, at_a_terminal):
    _typing(monkeypatch, ["", "", ""])
    answers = ask_sequence(_name_then_root())
    assert answers == {
        "name": "default_name", "root": "./data", "folder": "./data/in/default_name"
    }


def test_back_returns_to_the_previous_question(monkeypatch, at_a_terminal):
    _typing(monkeypatch, ["study", "/wrong", "back", "/right", "" ])
    answers = ask_sequence(_name_then_root())
    assert answers["root"] == "/right"


def test_back_refreshes_a_default_derived_from_the_changed_answer(monkeypatch, at_a_terminal):
    """The failure this prevents: correct the output root, then be offered an input
    path still hanging off the root you just replaced."""
    _typing(monkeypatch, ["study", "/wrong", "back", "/right", ""])
    answers = ask_sequence(_name_then_root())
    assert answers["folder"] == "/right/in/study", answers


def test_back_at_the_first_question_says_so_and_stays(monkeypatch, at_a_terminal, capsys):
    _typing(monkeypatch, ["back", "study", "/data", "/charts"])
    answers = ask_sequence(_name_then_root())
    assert "already at the first question" in capsys.readouterr().out
    assert answers["name"] == "study"


def test_cancel_abandons_the_whole_sequence(monkeypatch, at_a_terminal):
    _typing(monkeypatch, ["study", "cancel"])
    assert ask_sequence(_name_then_root()) is None


def test_ctrl_c_abandons_the_sequence(monkeypatch, at_a_terminal):
    def _interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _interrupt)
    assert ask_sequence(_name_then_root()) is None


def test_piped_input_never_blocks_and_takes_every_default(monkeypatch):
    """CI has no terminal; a question nobody will answer would hang forever."""
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive", lambda: False
    )

    def _refuse(_prompt=""):
        raise AssertionError("asked a question with no terminal to answer it")

    monkeypatch.setattr("builtins.input", _refuse)
    answers = ask_sequence(_name_then_root())
    assert answers == {
        "name": "default_name", "root": "./data", "folder": "./data/in/default_name"
    }


def test_ask_one_also_takes_the_default_when_piped(monkeypatch):
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive", lambda: False
    )
    assert ask_one("Anything?", "the-default") == "the-default"


def test_the_control_words_are_not_answers_anyone_would_type():
    """`back` and `cancel` are read before the answer, so they cannot be entered as
    values. Every question Junior asks wants a name or a path, so this is safe — but
    it is a real constraint and worth stating."""
    from apps_and_interfaces.console_questions import BACK_WORDS, CANCEL_WORDS

    assert BACK_WORDS.isdisjoint(CANCEL_WORDS)
    assert "quit" in CANCEL_WORDS, "the word people already know for leaving must work"


def test_back_offers_what_you_typed_not_the_original_suggestion(monkeypatch, at_a_terminal):
    """`back` exists to edit an answer. Re-offering the original default would mean
    the answer was discarded, which is the opposite of what the word promises."""
    _typing(monkeypatch, ["study", "/wrong", "back", "", "/charts"])
    answers = ask_sequence(_name_then_root())
    assert answers["root"] == "/wrong", "pressing enter on revisit lost the typed answer"


def test_changing_an_answer_refreshes_the_ones_derived_from_it(monkeypatch, at_a_terminal):
    """Going back and changing the root must not leave a later path still hanging off
    the value that was just replaced."""
    _typing(monkeypatch, ["study", "/wrong", "back", "/right", ""])
    answers = ask_sequence(_name_then_root())
    assert answers["folder"] == "/right/in/study", answers


class TestPastedPaths:
    """Paths arrive by paste or by dragging a folder into the terminal. Both bring
    punctuation a shell would have eaten; at a prompt it lands inside the value.

    The case that bit: a leading `'` with no closing one turned
    `'/Users/me/Desktop/` into a real folder named `'` with the project nested inside
    it, reported as success."""

    def test_an_unbalanced_leading_quote_is_dropped(self):
        from apps_and_interfaces.console_questions import clean_answer

        assert clean_answer("'/Users/me/Desktop/") == "/Users/me/Desktop/"

    def test_matched_quotes_are_dropped(self):
        from apps_and_interfaces.console_questions import clean_answer

        assert clean_answer('"/Users/me/My Charts"') == "/Users/me/My Charts"
        assert clean_answer("'/Users/me/Desktop'") == "/Users/me/Desktop"

    def test_dragged_folders_bring_escaped_spaces(self):
        from apps_and_interfaces.console_questions import clean_answer

        assert clean_answer("/Users/me/My\\ Charts") == "/Users/me/My Charts"

    def test_an_ordinary_path_is_untouched(self):
        from apps_and_interfaces.console_questions import clean_answer

        assert clean_answer("/Users/me/ok") == "/Users/me/ok"
        assert clean_answer("  /Users/me/ok  ") == "/Users/me/ok"

    def test_cleaning_happens_before_the_answer_is_used(self, monkeypatch, at_a_terminal):
        _typing(monkeypatch, ["study", "'/tmp/desktop/", ""])
        answers = ask_sequence(_name_then_root())
        assert answers["root"] == "/tmp/desktop/"
        assert "'" not in answers["folder"]
