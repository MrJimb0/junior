"""Asking a person a short sequence of questions, with a way out of each one.

A command that asks three things in a row needs `back` — mistyping the second answer
should not mean finishing the command and starting over, or killing the session with
Ctrl-C. It also needs `cancel`, so an operator who realises they are in the wrong place
can leave without writing anything.

Kept apart from both the CLI and the interactive prompt because both ask questions and
neither should import the other.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

import click

# Typed at any question. Recognized before the answer, so a literal value of "back"
# is not something you can enter — no answer Junior asks for is one of these words.
BACK_WORDS = {"back", "b", "<"}
CANCEL_WORDS = {"cancel", "abort", "quit", "q"}


@dataclass(frozen=True)
class Question:
    """One prompt. ``default_from`` gets the answers so far, so a later question can
    suggest something derived from an earlier one."""
    key: str
    label: str
    default_from: Callable[[dict[str, str]], str]


def is_interactive() -> bool:
    """Whether there is someone at a keyboard to answer a question."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):  # a replaced or closed stream
        return False


def clean_answer(typed: str) -> str:
    """Strip the punctuation a shell would have removed, which a prompt does not.

    Paths reach these questions by being pasted, or dragged into the terminal — both
    of which bring quotes, and dragging also brings backslash-escaped spaces. At a
    shell those are consumed by the shell; here they land inside the value, so a
    stray quote became a folder literally named `'`. No answer Junior asks for
    legitimately begins or ends with a quote."""
    answer = typed.strip()
    while len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
        answer = answer[1:-1].strip()
    # An unbalanced quote is the case that actually bit: `'/Users/me/Desktop/` with
    # no closing one. Drop them wherever they sit rather than only in matched pairs.
    answer = answer.strip("\"'").strip()
    return answer.replace("\\ ", " ")


def ask_one(label: str, default: str) -> str:
    """Ask a single question. Takes the default when piped, so scripts never block."""
    if not is_interactive():
        return default
    try:
        answer = clean_answer(input(f"  {label}\n    [{default}]: "))
    except (EOFError, KeyboardInterrupt):
        click.echo()
        return default
    return answer or default


def ask_sequence(questions: list[Question]) -> dict[str, str] | None:
    """Ask each question in turn; return the answers, or None if cancelled.

    Walks an index rather than a loop over the list so `back` can step it backwards.
    Defaults are recomputed on each visit, so going back and changing an answer
    refreshes what the later questions suggest instead of offering a stale value
    derived from the answer that was just replaced."""
    answers: dict[str, str] = {}
    if not is_interactive():
        for question in questions:
            answers[question.key] = question.default_from(answers)
        return answers

    position = 0
    while position < len(questions):
        question = questions[position]
        # What you typed last time is the default when you come back to it. Offering
        # the original suggestion instead would mean `back` silently discarded the
        # answer, when its whole purpose is to let you edit it.
        previously = answers.get(question.key)
        default = previously if previously is not None else question.default_from(answers)
        try:
            typed = clean_answer(input(f"  {question.label}\n    [{default}]: "))
        except (EOFError, KeyboardInterrupt):
            click.echo()
            return None

        if typed.lower() in CANCEL_WORDS:
            return None
        if typed.lower() in BACK_WORDS:
            if position == 0:
                click.secho("  (already at the first question — `cancel` to leave)", dim=True)
                continue
            position -= 1
            continue

        answers[question.key] = typed or default
        position += 1
        # Later answers derived from this one are now stale, so drop them and let
        # their defaults be recomputed from the value that just changed.
        for later in questions[position:]:
            answers.pop(later.key, None)
    return answers


def describe_controls() -> None:
    """One dim line telling the operator the words that work at a question."""
    click.secho("  (`back` to change the previous answer, `cancel` to stop)", dim=True)
