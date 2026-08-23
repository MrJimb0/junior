"""What extraction is doing, while it does it.

Extraction is the only stage measured in minutes, and it was the only one that said
nothing until a patient finished. Twenty minutes of silence is indistinguishable from a
hang — which is how it got reported, as "something hanged".

One display, used by `junior extract` and by the extract phase of `junior run`. Two
would drift, and the one nobody was looking at would be the one that lied.

What it must not do is as load-bearing as what it does. No values: extraction output is
patient data and a terminal outlives the run in scrollback and SLURM logs. No line per
model call: a variable can make a dozen, and a display that scrolls is one nobody reads.
The unit is the variable, because that is the unit of work that can succeed or fail on
its own, and the thing somebody would act on.

``already complete`` and ``cached`` are deliberately different words. Already complete
means no work was needed -- the answer is on disk and its recorded settings match this
run. Cached means the work WAS done and a stored model response stood in for one call of
it. Reporting a resumed variable as "cached" would say the model cache is what makes
resume correct, and it is not: resume is decided by recorded extraction state, and holds
with the cache empty.
"""
from __future__ import annotations

import sys
from collections.abc import Callable

# How long a state name is padded to, so the states line up under each other and the
# eye can find "failed" without reading every line.
_STATE_COLUMN = 18

_STATES_THAT_END_A_VARIABLE = frozenset({"done", "failed", "failed again", "already complete"})


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


def report_variables(
    *, write: Callable[[str], None] | None = None, animated: bool | None = None
) -> Callable[[str, str, float | None], None]:
    """Build the ``on_variable(name, state, seconds)`` callback extraction calls.

    ``failed again`` means the retry produced the identical failure. Extraction is
    deterministic, so that is the expected outcome of retrying unchanged evidence, and
    it needs its own word — "failed" invites another retry, which will do the same.

    ``running`` and ``retrying`` are printed the moment they start, which is the whole
    point — the line appears while the work is happening, not once it is over. On a
    terminal an opening state HOLDS its line open (no newline yet), so the finishing
    state can return to column 0 and overwrite it — a variable takes one line rather
    than two. The first version printed the opening line with a newline and then
    emitted the overwrite sequence at the start of the next, blank line, so the
    "overwrite" cleared nothing and every variable rendered twice with a stale
    ``running`` above its outcome. Piped to a file both lines are kept, because a
    log wants the start time.
    """
    on_a_terminal = sys.stdout.isatty() if animated is None else animated
    open_line = {"pending": False}

    def on_variable(name: str, state: str, seconds: float | None) -> None:
        # Padded to at least the column, never truncated to it. A fixed 22 was set when
        # every recipe name fitted; the book now holds names up to 39 characters, and
        # Python pads without clipping, so `breast_clinical_stage_ajcc7` ran straight
        # into its own status and read as `breast_clinical_stage_ajcc7done`. A long name
        # loses the alignment for its own row, which is better than losing the word.
        line = (f"      {name:<{max(22, len(name) + 2)}}"
                f"{state:<{_STATE_COLUMN}}{_duration(seconds)}").rstrip()
        closes = state in _STATES_THAT_END_A_VARIABLE
        if not on_a_terminal:
            (write if write is not None else (lambda text: print(text, flush=True)))(line)
            return
        overwrite = "\r\033[K" if open_line["pending"] else ""
        if write is not None:
            # A custom sink receives whole lines, overwrite sequence included, so a
            # capture can assert the terminal behaviour without owning a terminal.
            write(overwrite + line)
        else:
            sys.stdout.write(overwrite + line + ("\n" if closes else ""))
            sys.stdout.flush()
        open_line["pending"] = not closes

    return on_variable


def report_patient(index: int, total: int, patient_id: str,
                   *, write: Callable[[str], None] | None = None) -> None:
    """The header a patient's variables appear under.

    The patient id is an identifier, not chart content, and it is already on screen in
    every other stage's per-patient line — without it a failure names no one."""
    out = write if write is not None else (lambda line: print(line, flush=True))
    out(f"\n[{index}/{total}] {patient_id}")


def report_nothing_to_do(*, write: Callable[[str], None] | None = None) -> None:
    """Said when every variable for a patient was already complete.

    Otherwise a resumed patient prints a header and nothing under it, which reads as the
    display having failed rather than as there being no work."""
    out = write if write is not None else (lambda line: print(line, flush=True))
    out("      nothing to do — every variable already complete")
