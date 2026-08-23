"""Interactive `junior` session — a prompt over the commands the CLI already exposes.

Typing `junior` with no arguments lands here. Every line typed at the prompt is
dispatched to the same click group that `junior <command>` uses, so each command has
exactly one implementation and this module is only a way of reaching it. Adding a
command to the CLI makes it available here with no change to this file.

The Shiny app is the third way in. All three — prompt, flags, app — call the same
functions under src/jr_pipeline.
"""
from __future__ import annotations

import os
import shlex
from pathlib import Path

import click

from apps_and_interfaces.console_questions import (
    BACK_WORDS,
    CANCEL_WORDS,
    clean_answer,
    is_interactive,
)

PROMPT = "junior> "

# Handled here rather than dispatched, because they act on the session itself.
_QUIT_WORDS = {"quit", "exit", "q"}
_HELP_WORDS = {"help", "?"}
# `back` is a real word inside a command's questions. Typed at the prompt itself there
# is nowhere to go, so say that instead of "No such command" — which reads as though
# the word were wrong rather than the place.
_BACK_WORDS = {"back", "b", ".."}

# The one answer to "which project?" that is not one of the projects. Spelled out rather
# than abbreviated, because the numbers already own the single keystrokes.
#
# There is deliberately no "carry on without one". A session with no project pinned falls
# back to the bundled example settings, which read the checkout's own examples/ and write
# its data/ — the state this question exists to prevent, and the one the comment above
# `resolve_run_id` in the CLI calls dangerous. The way out of the question is to leave and
# cd, which a shell can do and this prompt cannot.
_NEW_PROJECT_WORDS = {"new", "new-project", "new project"}

# Upper bound on how many values one command line may be asked for before giving up.
# No command in the CLI has close to this many required parameters; it exists so a
# command that reports the same missing value forever cannot loop.
_MAX_PROMPTS_PER_COMMAND = 8


def start(group: click.Group, version: str) -> None:
    """Read lines and run them as `junior` commands until the user quits."""
    from apps_and_interfaces.command_line_interface import (
        set_interactive_session,
    )

    # Lets commands tell where their output is being read. A shell instruction like
    # "cd somewhere" is useless advice inside a prompt that has no cd.
    set_interactive_session(True)
    click.echo()
    click.echo(
        # Not "entirely on this machine" — the same commands run on a cluster, and
        # this is the one line reassuring anyone about where patient data goes.
        f"{_GUTTER}{click.style(f'Junior v{version}', bold=True)}"
        f"{click.style('  ·  clinical extraction, auditable  ·  charts go only where your allowlist says', dim=True)}"
    )
    click.echo()
    try:
        # Which cohort before what to do with it. A command run at this prompt reads the
        # surroundings the same way one run at a shell does, and the surroundings of a
        # prompt are wherever the terminal happened to open — so the project is settled
        # once, out loud, rather than re-derived per command from a folder nobody chose.
        if not open_on_a_project(group):
            return
        _show_commands(group)
        _loop(group)
    finally:
        # Cleared on the way out so a test, or a second session in one process, does
        # not inherit a flag saying it is somewhere it is not.
        set_interactive_session(False)


def open_on_a_project(group: click.Group) -> bool:
    """Settle which project this session works on, and hold it for every command in it.

    Holding it means setting $JUNIOR_CONFIG, which config discovery already prefers over
    the walk up from the current folder while still letting an explicit --config win. It
    is the same mechanism `new-project` uses to hand the session the cohort it just made,
    so a pinned project adds no new way to be somewhere — only a way to say where.

    Standing inside a project is itself an answer, and the usual one: cd to the cohort,
    open the prompt, nothing to ask. The question is for the other case, a prompt opened
    somewhere that is not a cohort, where the folder above may hold settings belonging to
    work you are not doing, or none at all.

    False means the person asked to leave rather than answered — this is the first thing
    a session shows, so it is also the first place someone changes their mind.
    """
    found = _project_around_us()
    if not is_interactive():
        # Piped or scripted: nobody is there to choose, and discovery already answers
        # for every command that follows. Ask nothing and change nothing — pinning a
        # project nobody picked, or writing it down, would be this function inventing
        # a decision out of the absence of one.
        return True
    # A settings file sitting directly in the home folder is above nearly everything on
    # the machine, so it answers "which project" for every terminal that opens. It stays
    # usable — someone may have put it there deliberately — but it is offered rather than
    # assumed, because being above you is not the same as being yours.
    if found is not None and found.parent != Path.home():
        work_on(found)
        return True
    return _ask_which_project(group, found)


def _project_around_us() -> Path | None:
    """The project the surroundings point at, if they point at one."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
    except FileNotFoundError:
        return None
    # What discovery falls back to when there is no project anywhere. It is a config,
    # but it is not a cohort anyone chose, so it is not an answer to this question.
    if choice.reason == "bundled default":
        return None
    return choice.path


def work_on(config_path: Path) -> None:
    """Hold one project for the rest of the session, and say which one it is."""
    from apps_and_interfaces.project_registry import project_name, remember
    from jr_pipeline.runtime_infrastructure.project_context import (
        CONFIG_ENVIRONMENT_VARIABLE,
    )

    os.environ[CONFIG_ENVIRONMENT_VARIABLE] = str(config_path)
    # Remembered on use rather than only on creation, so the menu's order is the order
    # you actually work in and a project made on another machine — or by hand — joins
    # the list the first time it is opened.
    remember(config_path)
    click.echo(
        f"{_GUTTER}{click.style('Working on', dim=True)} "
        f"{click.style(project_name(config_path), fg=_COMMAND_COLOR, bold=True)}"
        f"{click.style(f'   {config_path}', dim=True)}"
    )
    for line, alarming in _how_this_project_stands(config_path):
        click.secho(f"{_GUTTER}{line}", fg="yellow" if alarming else None,
                    dim=not alarming)
    click.echo()


def _how_this_project_stands(config_path: Path) -> list[tuple[str, bool]]:
    """Where the chosen project writes and how far it has got, as lines to print.

    Choosing is a decision, and it was the one decision Junior answered with nothing but
    the name back. Everything here is already on disk and cheap to read for the single
    project just picked — which run is newest, whether the output folder is even there —
    and it is what tells you at a glance that you picked the one you meant, and where you
    left off. The alternative was finding out by running a stage."""
    import yaml

    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        SENSITIVE_LABEL,
        pipeline_run_receipts_root,
    )
    from jr_pipeline.runtime_infrastructure.project_context import RUN_ID_PATTERN

    try:
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    written = settings.get("output_root") if isinstance(settings, dict) else None
    if not written:
        return []
    output_root = Path(str(written)).expanduser()
    where = f"output {_shorten(output_root)}"
    if not output_root.is_dir():
        # Deleted, or a settings file written by hand. Not repaired — this is the
        # operator's file — but not printed like a live location either.
        return [(f"{where}   ⚠ folder missing", True)]

    receipts = output_root / SENSITIVE_LABEL / pipeline_run_receipts_root().name
    runs = [
        directory for directory in receipts.iterdir()
        if directory.is_dir() and RUN_ID_PATTERN.fullmatch(directory.name)
    ] if receipts.is_dir() else []
    if not runs:
        return [(f"{where}   no runs yet", False)]
    # The run id is not read by anyone — it is a timestamp you copy, not a fact you take
    # in at a glance. How much is in there is the part that answers "where was I".
    patients = max(runs, key=lambda directory: directory.name) / "patients"
    how_many = len([p for p in patients.iterdir() if p.is_dir()]) if patients.is_dir() else 0
    return [(f"{where}   {len(runs)} run{'s' if len(runs) != 1 else ''} · "
             f"{how_many} patient{'s' if how_many != 1 else ''}", False)]


def _shorten(path: Path) -> str:
    """A path with the home directory written as `~`, for a line meant to be read."""
    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def _ask_which_project(group: click.Group, found: Path | None) -> bool:
    """Offer the projects on this machine and take one, or leave discovery alone.

    False when the answer was `quit` — asking which project is not a gate to get past
    on the way to the session, so the words that leave a session leave from here too."""
    from apps_and_interfaces.project_registry import match_project, remembered_projects

    choices = remembered_projects()
    # The one the surroundings found is on the menu whether or not it was ever written
    # down — it is the answer the old behaviour would have given silently, so it has to
    # be reachable, and seeing it beside the others is what makes the difference visible.
    if found is not None and found not in choices:
        choices.append(found)
    # The first entry is what pressing enter takes, so a settings file in the home
    # folder never gets to be it: offering the one thing on the menu that claims every
    # other folder as the path of least resistance would undo the point of asking.
    # Sunk rather than hidden, and the order among the rest is left alone.
    choices.sort(key=lambda path: path.parent == Path.home())

    # Enter takes the first entry, but only when that entry is an ordinary project. With
    # nothing but a home-folder config to offer, sinking it changes nothing, so the
    # default becomes setting one up: of the answers available it is the only one that
    # ends somewhere defined, and it is loud and undoable where running against the
    # wrong cohort is neither.
    default = "1" if choices and choices[0].parent != Path.home() else "new"

    _show_projects(choices)
    while True:
        try:
            typed = clean_answer(input(f"{_GUTTER}project [{default}]: ")) or default
        except (EOFError, KeyboardInterrupt):
            click.echo()
            return False
        answer = typed.lower()
        # Cancelling this question and quitting the session are the same act. Inside a
        # command `cancel` returns you to the prompt, but this question IS the front of
        # the prompt, and there is nothing behind it to return to.
        if answer in _QUIT_WORDS or answer in CANCEL_WORDS:
            return False
        if answer in _NEW_PROJECT_WORDS:
            # Dispatched rather than reimplemented: `new-project` asks its own questions
            # and already points the session at what it creates.
            _dispatch(group, ["new-project"])
            from jr_pipeline.runtime_infrastructure.project_context import (
                CONFIG_ENVIRONMENT_VARIABLE,
            )

            if os.environ.get(CONFIG_ENVIRONMENT_VARIABLE):
                return True
            # It did not create one — cancelled, refused a duplicate name, or failed.
            # `_dispatch` swallows all three, so taking its return as success opened the
            # session on no project at all, which is what falling back to the bundled
            # example settings means: reading the checkout's examples and writing its
            # data. Ask again instead.
            click.secho(f"{_GUTTER}  No project was created.", dim=True)
            _show_projects(choices)
            continue
        chosen = match_project(typed, choices)
        if chosen is not None:
            work_on(chosen)
            return True
        click.secho(
            f"{_GUTTER}  No project called {typed!r}. Type its number, its name, or the "
            f"path to its folder — or `new` to set one up, `quit` to leave.",
            dim=True,
        )


def _show_projects(choices: list[Path]) -> None:
    """The menu: the projects on this machine, and the one action that is not one of them.

    Numbers address projects and the word addresses the action, so the two never compete
    for the same keystroke — a list where `1` and `n` mean different kinds of thing makes
    the reader work out which scheme they are in before they can answer.

    Every answer here ends somewhere defined: a project, a new project, or out. Nothing
    on this menu leads to running against no project at all.
    """
    from apps_and_interfaces.project_registry import project_name, project_where_it_lives

    click.secho(
        f"{_GUTTER}{'Which project?' if choices else 'No projects on this machine.'}",
        bold=True,
    )
    click.echo()
    if not choices:
        click.secho(
            f"{_GUTTER}Junior works on one cohort at a time — a folder that says where "
            f"your charts are\n{_GUTTER}and where the output goes. There is not one here "
            f"yet.",
            dim=True,
        )
        click.echo()
    name_width = max(
        max((len(project_name(path)) for path in choices), default=0), len("new")
    ) + 3
    # Each row says what the cohort IS, in the terms someone choosing between cohorts
    # thinks in. Nothing here describes how Junior finds settings files: a picker is not
    # where anyone wants to learn that, and once you are choosing from a list explicitly,
    # none of it can catch you out anyway. The layout hazards that used to be annotated
    # here are handled silently instead — see how `choices` is ordered and how `default`
    # is picked in the caller.
    for position, path in enumerate(choices, start=1):
        note = project_where_it_lives(path)
        click.echo(_project_line(str(position), project_name(path), note, name_width))
    if choices:
        click.echo()
    # No marker column: the word on the left IS what you type, so a letter beside it
    # would be a second name for the same answer.
    click.echo(_project_line(
        "", "new", "set one up — where your charts are, where output goes", name_width,
    ))
    click.echo()
    click.secho(
        f"{_GUTTER}   {'a number, a name, or a path' if choices else 'type new'}"
        f"  ·  quit to leave and cd somewhere else",
        dim=True,
    )
    click.echo()


def _project_line(marker: str, name: str, note: str, name_width: int) -> str:
    """One aligned row of the project menu, laid out like the command listing below."""
    return (
        f"{_GUTTER}  {click.style(marker.ljust(_MARKER_WIDTH), dim=True)}"
        f"{click.style(name.ljust(name_width), fg=_COMMAND_COLOR, bold=True)}"
        f"{click.style(note, dim=True)}"
    )


def _forget_what_was_typed_while_it_ran() -> None:
    """Throw away anything typed at the terminal while a command was running.

    A stage can take minutes, and it draws its progress line in place, so a keypress
    during the wait is echoed into the middle of that line — which is where a stray `^R`
    comes from. The display is cosmetic. What is not cosmetic is that those keystrokes
    are still sitting in the terminal's input buffer when the stage ends, and the next
    thing to read stdin is this prompt, which takes them as a command.

    Someone pressing Enter a few times out of impatience during a twelve-minute extract
    is the common case and is harmless. The one that is not is a half-typed word, or a
    control character, arriving as the answer to a question the prompt has not asked yet.
    Type-ahead is a reasonable thing to want from a shell; it is not a reasonable thing
    to want from something that runs a clinical pipeline, so it is discarded."""
    if not is_interactive():
        return
    try:
        import sys
        import termios
    except ImportError:  # not a POSIX terminal
        return
    # Whether anything is waiting, asked before it is thrown away. Discarding silently
    # is what makes this look broken: the terminal echoed every keystroke, so it was on
    # screen, and then it vanished with nothing said. Being ignored on purpose and being
    # ignored by accident look identical unless one of them speaks.
    try:
        import select

        something_was_typed = bool(select.select([sys.stdin], [], [], 0)[0])
    except (OSError, ValueError):
        something_was_typed = False
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (termios.error, OSError, ValueError):
        # A replaced or closed stream, or a terminal that will not flush. Losing the
        # discard is not worth losing the session over.
        return
    if something_was_typed:
        click.secho(
            "  (ignored what was typed while that ran — one command at a time)",
            dim=True,
        )


def _loop(group: click.Group) -> None:
    while True:
        try:
            line = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            return
        if not line:
            continue
        try:
            arguments = shlex.split(line)
        except ValueError:  # shlex: unbalanced quotes
            click.echo("  Unmatched quote in that line — check the quotes and try again.")
            continue

        normalized = _normalize(arguments, group)
        if normalized is None:
            continue  # normalize already said what was wrong
        arguments = normalized
        if not arguments:
            _show_commands(group)
            continue

        # Checked AFTER normalizing, not before: `junior help` strips to `help`, and
        # testing the raw line first would send that to click as a command.
        first = arguments[0].lower()
        if first in _QUIT_WORDS:
            return
        if first in _HELP_WORDS:
            _show_commands(group)
            continue
        if first in _BACK_WORDS:
            click.secho(
                "  You are at the top level — nothing to go back to. "
                "`back` works inside a command's questions. `quit` to leave.",
                dim=True,
            )
            continue
        try:
            _dispatch(group, arguments)
        except KeyboardInterrupt:
            # Ctrl-C cancels the command, not the session. Without this it escapes
            # the loop entirely — KeyboardInterrupt is a BaseException, so none of
            # _dispatch's handlers see it — and kills the prompt mid-run.
            click.echo()
            click.secho("  cancelled", dim=True)
        finally:
            # In the finally, not after the call: a cancelled command is exactly when
            # the most has been typed at the terminal, and leaving that buffered would
            # hand it to the next prompt as a command.
            _forget_what_was_typed_while_it_ran()


def _normalize(arguments: list[str], group: click.Group) -> list[str] | None:
    """Turn what a person plausibly types into what click can parse.

    Every rule here comes from a real mistyping, and each is unambiguous — none of
    them can turn one valid command into a different valid command."""
    # `<command> --help` is printed as a placeholder in this session's own footer, so
    # people copy it brackets and all. Nothing Junior takes is bracketed.
    arguments = [
        argument[1:-1] if len(argument) > 2 and argument.startswith("<")
        and argument.endswith(">") else argument
        for argument in arguments
    ]
    # The program name is implied at this prompt; `junior status` is what the docs
    # and every error message say to run, and what habit produces.
    while arguments and arguments[0] in {"junior", "jr-pipeline", "jr"}:
        arguments = arguments[1:]
    if not arguments:
        return arguments

    # `-h` is reflexive for anyone who uses a terminal, and click only accepts
    # --help here. Rewritten rather than added as an alias so it works for every
    # command at once.
    arguments = ["--help" if argument == "-h" else argument for argument in arguments]

    # `Ingest` for `ingest`. The session already matches `HELP` and `QUIT` regardless
    # of case, so punishing a capitalized command name teaches a rule that only holds
    # half the time. Lowercased only when that spelling is a real command.
    if arguments and arguments[0] not in group.commands:
        if arguments[0].lower() in group.commands:
            arguments = [arguments[0].lower(), *arguments[1:]]

    # `new project` for `new-project`. Only joins when the hyphenated form is a real
    # command, so a command followed by an argument is never welded together.
    if len(arguments) > 1 and arguments[0] not in group.commands:
        joined = f"{arguments[0]}-{arguments[1]}".lower()
        if joined in group.commands:
            arguments = [joined, *arguments[2:]]

    # The listing numbers the stages 1-4; typing the number is the obvious next move
    # once you have read it. Resolved against the same table the listing renders from,
    # so the numbers can never drift apart.
    if arguments[0].rstrip(".").isdigit():
        from apps_and_interfaces.command_line_interface import (
            PIPELINE_STAGE_DESCRIPTIONS,
        )

        stages = list(PIPELINE_STAGE_DESCRIPTIONS)
        position = int(arguments[0].rstrip("."))
        if 1 <= position <= len(stages):
            arguments = [stages[position - 1], *arguments[1:]]
        else:
            click.secho(
                f"  There is no step {position} — the steps are 1-{len(stages)}.", dim=True
            )
            return None  # already answered; do not also dump the command list
    return arguments


def _dispatch(group: click.Group, arguments: list[str]) -> None:
    """Run one command line through the click group, asking for anything it needs.

    Typing a bare command name at a prompt is a reasonable thing to do, so a missing
    required value is a question to ask rather than a usage error to print: click
    raises MissingParameter, we ask for that one value, and retry. The loop is bounded
    by the command's own parameter count, so a command that keeps reporting the same
    missing value cannot spin.

    ``standalone_mode=False`` stops click from calling sys.exit on --help or on a
    usage error, which would end the whole session instead of the one command."""
    for _ in range(_MAX_PROMPTS_PER_COMMAND):
        try:
            group.main(args=arguments, prog_name="junior", standalone_mode=False)
            return
        # MissingParameter subclasses UsageError subclasses ClickException, so it has
        # to be caught first or the generic handler below would just print it.
        except click.MissingParameter as e:
            supplied = _ask_for(e.param, arguments)
            if supplied is None:
                click.echo("  cancelled")
                return
            arguments = supplied
        except click.NoSuchOption as e:
            # Gets the same closing hint as every other usage error. Without it, an
            # unknown flag was the one failure that told you nothing about what to do.
            e.show()
            click.secho("  Type help for the command list.", dim=True)
            return
        except click.UsageError as e:
            # Click's own advice is "Try 'junior --help'", which is a command line —
            # useless to someone already inside the session. Say what works here.
            e.show()
            click.secho("  Type help for the command list.", dim=True)
            return
        except click.ClickException as e:
            e.show()
            return
        except click.exceptions.Abort:
            click.echo("  cancelled")
            return
        except SystemExit:
            return
        except Exception as e:
            # A failing command must not take the session down with it. The exception
            # class is a Python detail — the person at this prompt is a researcher, so
            # show what happened and what to try, and leave the type to the log.
            click.secho(f"  Something went wrong: {e}", fg="red")
            click.secho(
                "  `junior status` shows which config and run you are on; "
                "`<command> --help` shows its options.",
                dim=True,
            )
            return
    click.echo("  giving up — too many missing values; pass them as flags instead")


# What to call a missing value, for the ones whose internal name would show through.
# `human_readable_name` gives PATIENT_FOLDER or run_id — a Python identifier or a
# shouted placeholder, neither of which is a question.
_VALUE_LABELS = {
    "run_id": "Which run? (`status` shows the current one)",
    "run_root": "Which run folder?",
    "patient_id": "Which patient?",
    "patient_folder": "Which folder holds the chart files?",
    "config_path": "Which config file?",
    "output_path": "Where should it be written?",
    "output_dir": "Which folder should it be written to?",
    "name": "Name for this cohort",
}


def _ask_for(parameter: click.Parameter | None, arguments: list[str]) -> list[str] | None:
    """Ask for one missing value and return the fuller command line, or None to stop."""
    if parameter is None:
        return None
    # Falls back to the parameter's own name with the underscores taken out, rather
    # than click's human_readable_name — which for an Option is the raw Python
    # destination, so the prompt read `gold_path:` or `run_a:`. Every option cannot
    # be listed above, but none of them should ask in snake_case.
    name = parameter.name or ""
    label = _VALUE_LABELS.get(name) or (
        name.replace("_", " ").capitalize() if name else parameter.human_readable_name
    )
    # click.Argument has no `help`; only Option does, hence the lookup rather than
    # an attribute access.
    help_text = getattr(parameter, "help", None)
    hint = f" ({help_text})" if help_text else ""
    try:
        answer = input(f"  {label}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        click.echo()
        return None
    # Same words as a command's own questions, so `cancel` means one thing everywhere.
    # An empty line cancels too — there is no default to fall back on here.
    if not answer or answer.lower() in CANCEL_WORDS or answer.lower() in BACK_WORDS:
        return None
    # An argument is positional and goes on the end; an option needs its flag. Both
    # land after the command name, which is all click needs to parse them.
    if isinstance(parameter, click.Argument):
        return [*arguments, answer]
    return [*arguments, parameter.opts[0], answer]


_GUTTER = "  "
# Width of the step marker column ("1  "), so unnumbered commands line up under
# numbered ones instead of starting a second ragged edge.
_MARKER_WIDTH = 3


# Command names. Green reads as "something you can run" without the terminal-default
# look of cyan, and stays legible on both light and dark backgrounds.
_COMMAND_COLOR = "green"


def _command_line(name: str, description: str, name_width: int, step: int | None = None) -> str:
    """One aligned row: optional step number, command name, description."""
    marker = click.style(f"{step}".ljust(_MARKER_WIDTH), dim=True) if step else " " * _MARKER_WIDTH
    return (
        f"{_GUTTER}  {marker}"
        f"{click.style(name.ljust(name_width), fg=_COMMAND_COLOR, bold=True)}"
        f"{click.style(description, dim=True)}"
    )


def _heading(title: str, note: str, note_column: int) -> str:
    """A section title, with its note starting on the descriptions' column."""
    padding = " " * max(2, note_column - len(_GUTTER) - len(title))
    styled = f"{_GUTTER}{click.style(title.upper(), bold=True)}"
    return f"{styled}{padding}{click.style(note, dim=True)}" if note else styled


def _show_commands(group: click.Group) -> None:
    # Rendered from the CLI's own tables, so the grouping, wording, and numbering
    # here cannot drift from what `junior --help` and each command's help report.
    from apps_and_interfaces.command_line_interface import (
        APP_COMMANDS,
        META_COMMANDS,
        PIPELINE_PHASES,
        PIPELINE_STAGE_DESCRIPTIONS,
        SHARE_COMMANDS,
        START_COMMANDS,
    )

    # Three groups, because there are three moments: setting a cohort up, running it,
    # and looking at what came out. The numbered steps are the spine — a new reader
    # should be able to find "what do I type first" without reading a description.
    sections: list[tuple[str, dict[str, str], bool]] = [
        ("Start", START_COMMANDS, False),
        *((title, stages, True) for title, stages in PIPELINE_PHASES.items()),
        ("Then", {**APP_COMMANDS, **SHARE_COMMANDS}, False),
        # Listed like everything else rather than crammed into the footer as bare words.
        # These three were the only commands with no description anywhere on this screen,
        # sitting next to `help` and `quit`, which are not commands at all — a second and
        # worse listing, below the good one.
        ("Any time", META_COMMANDS, False),
    ]

    widest_name = max(
        len(name)
        for table in (
            PIPELINE_STAGE_DESCRIPTIONS, START_COMMANDS, APP_COMMANDS, SHARE_COMMANDS,
            META_COMMANDS,
        )
        for name in table
    )
    heading_width = max(len(title) for title, *_ in sections) + 2
    name_width = widest_name + 3

    step = 0
    for title, commands, numbered in sections:
        for position, (name, description) in enumerate(commands.items()):
            if numbered:
                step += 1
            # The heading sits on the first row of its group rather than above it, so a
            # group costs one line instead of three and the eye runs down one column.
            shown = title.upper() if position == 0 else ""
            marker = f"{step} " if numbered else "  "
            click.echo(
                f"{_GUTTER}{click.style(shown.ljust(heading_width), bold=True)}"
                f"{click.style(marker, dim=True)}"
                f"{click.style(name.ljust(name_width), fg=_COMMAND_COLOR, bold=True)}"
                f"{click.style(description, dim=True)}"
            )
        click.echo()

    # Just the two words that act on the session itself. What used to be here — an
    # instruction on how to read a numbered list, and three commands with no
    # descriptions — is either obvious or now listed above with the rest.
    click.secho(f"{_GUTTER}{'':{heading_width}}  help  ·  quit", dim=True)
    click.echo()
