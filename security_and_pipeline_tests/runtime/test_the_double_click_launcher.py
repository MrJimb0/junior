"""Double-clicking should open the workbench, without going through anybody's shell.

Reported twice. Opening the `jr` launcher from Finder, and then a .command file,
both printed

    zsh: no such file or directory: Users/<you>/Desktop/junior/...

with the leading slash gone. Nothing about Junior was broken. Terminal runs both by
TYPING the path into a fresh interactive shell, and oh-my-zsh's "would you like to
update? [Y/n]" was waiting on stdin: the "/" answered the prompt and the rest of the
path became the command. The .command extension changes nothing — that was the first
guess and it was wrong.

An app is the only launcher macOS runs without a shell in the way: AppleScript's
`do shell script` goes through /bin/sh -c, which reads no startup files.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "Junior Workbench.app"
SOURCE = REPO / "apps_and_interfaces" / "workbench_launcher.applescript"


def test_the_app_is_built_and_launchable():
    assert APP.is_dir(), "there is no double-click route to the workbench"
    assert (APP / "Contents" / "MacOS").is_dir(), f"{APP.name} is not a runnable bundle"


def test_the_source_is_tracked_beside_it():
    """A compiled bundle does not diff and cannot be reviewed. Whatever the app does,
    it has to be readable as text next to it — and rebuildable from it."""
    assert SOURCE.is_file(), "the app ships with no readable source"
    assert "osacompile" in SOURCE.read_text(encoding="utf-8"), (
        "the source does not say how to rebuild the app from itself"
    )


def test_it_never_starts_the_server_itself():
    """Straight through to the CLI. A launcher that reimplements the startup is a
    second place the workbench can be started differently — a different port, a
    different data root, a different project."""
    body = SOURCE.read_text(encoding="utf-8")
    assert "workbench" in body, "it never asks the CLI for the workbench"
    assert "shiny run" not in body, "it starts the server itself instead of asking the CLI"
    assert "uvicorn" not in body and "--port" not in body, (
        "it chooses the port itself, so the CLI and the app can disagree about it"
    )


def test_it_runs_from_wherever_finder_opens_it():
    """Finder's working directory is not the repository, so the app has to resolve its
    own location before running a relative ./jr."""
    body = SOURCE.read_text(encoding="utf-8")
    assert "path to me" in body and "cd " in body


def test_it_can_be_stopped_without_activity_monitor():
    """The server has no window. Something has to own its lifetime, or the only way to
    stop it is to go looking for the process."""
    body = SOURCE.read_text(encoding="utf-8")
    assert "stopServer" in body
    assert "kill" in body and "pidFile" in body


def test_a_second_double_click_does_not_start_a_second_server():
    """Two servers on two ports is how somebody ends up reviewing a run in a stale tab
    and cannot work out why their change did nothing."""
    assert "isRunning" in SOURCE.read_text(encoding="utf-8")


def test_it_runs_the_junior_in_this_checkout_and_no_other():
    """Inherited from the `jr` script this replaced, and the reason that script existed.

    A stray `junior` earlier on somebody's PATH would seal and run different code
    against this checkout's data, and the failure would read as Junior misbehaving
    rather than as the wrong Junior. The app resolves from its own bundle location and
    has no PATH candidate at all — not finding one is a dialog saying how to install."""
    # Comments stripped: the file TALKS about $PATH to say it does not use one, and an
    # assertion that cannot tell the explanation from the behaviour is not an assertion.
    code = "\n".join(line for line in SOURCE.read_text(encoding="utf-8").splitlines()
                     if not line.strip().startswith("--"))

    assert ".venv-arm64/bin/junior" in code and ".venv/bin/junior" in code
    assert "path to me" in code, "it does not resolve from its own location"
    # The `jr` script ended with `command -v junior` as a last resort. Deliberately gone.
    assert "command -v" not in code
    assert "$PATH" not in code


def test_a_missing_install_says_what_to_do():
    """A novice double-clicking a checkout with no virtualenv gets an instruction, not
    a silent nothing and not a stack trace in a log file they do not know about."""
    body = SOURCE.read_text(encoding="utf-8")
    assert "not installed yet" in body
    assert "pip install -e" in body


def test_only_one_thing_in_the_root_looks_clickable():
    """Finder hides extensions. With the app's source sitting beside it, BOTH showed as
    "Junior Workbench" — one an application, one a text file — and the obvious thing to
    click opened AppleScript in an editor. Reported by the first person to try it.

    The source lives under apps_and_interfaces/ now. Anything else in the root sharing
    the app's displayed name puts that choice back."""
    displayed = [p.name[:-4] if p.name.endswith(".app") else p.name
                 for p in REPO.iterdir() if not p.name.startswith(".")]
    clashes = [n for n in displayed if n == APP.name[:-4] and displayed.count(n) > 1]
    assert not clashes, (
        f"more than one thing in the root shows as {APP.name[:-4]!r}: "
        f"{sorted(n for n in displayed if n == APP.name[:-4])}"
    )
    assert SOURCE.parent != REPO, "the app's source is back in the root beside it"


def test_no_user_facing_message_is_mangled():
    """`display alert "Junior started but never said serverURL"` shipped, because a
    variable rename ran over the message text as well as the identifiers. A user reads
    these; a test that only checks the code does not."""
    body = SOURCE.read_text(encoding="utf-8")
    for identifier in ("serverURL", "logFile", "pidFile", "theURL"):
        assert f'"{identifier}"' not in body, (
            f"the identifier {identifier!r} appears as a quoted string — a rename most "
            "likely ran through a message somebody is meant to read"
        )


def source_without_comments() -> str:
    """The file EXPLAINS both of the bugs below in its header. An assertion that cannot
    tell the explanation from the behaviour is not an assertion."""
    return "\n".join(line for line in SOURCE.read_text(encoding="utf-8").splitlines()
                     if not line.strip().startswith("--"))


def test_the_launch_is_detached_so_the_app_is_not_left_hanging():
    """Double-clicking did nothing at all — no browser, no dialog, not even an error.

    `do shell script` does not return when the shell exits. It returns when the pipe it
    handed that shell is closed, and a backgrounded server holds that pipe open for as
    long as it runs. The app sat in the launch line forever while Junior ran invisibly
    behind it, so the only symptom was an icon that bounced and then nothing. Backgrounding
    with `&` is not enough and neither is nohup: the whole launch has to run in a subshell
    with its own output thrown away."""
    launch = [line for line in source_without_comments().splitlines()
              if "workbench > " in line]
    assert launch, "nothing in the app launches the workbench"
    for line in launch:
        assert "( nohup" in line, (
            "the launch is not wrapped in a subshell, so do shell script will hang"
        )
        assert ") > /dev/null 2>&1" in line, (
            "the launch leaves a pipe open for the server to hold, so the app hangs "
            "at the moment it starts Junior and never reaches the browser or the dialog"
        )


def test_it_waits_for_the_server_to_answer_before_opening_the_browser():
    """The browser opened on a refused connection.

    Junior prints its address about two and a half seconds before it finishes binding
    the port. The launcher opened the browser 1.5 seconds after reading that line, so
    the first thing a new user saw was the browser's cannot-connect page and a dialog
    insisting the workbench was running. The log says where the server WILL be, not
    that it is there yet — only the port can say that."""
    code = source_without_comments()
    assert "waitUntilAnswering" in code, "nothing checks that the server is up yet"
    assert "curl" in code, "the readiness check never asks the port anything"
    assert "delay 1.5" not in code, "the browser still opens on a timer"

    starter = code.split("on startOrShow()", 1)[1].split("end startOrShow", 1)[0]
    assert starter.rindex("my openInBrowser") > starter.index("waitUntilAnswering"), (
        "a fresh start opens the browser before waiting for the server to answer"
    )


def test_the_built_app_is_a_stay_open_applet():
    """Reported twice as "I clicked it and it did nothing", and the bundle looks fine.

    A stay-open applet is the only kind that keeps running after its run handler
    returns — which is what lets this one own the server, notice through `on idle`
    that the server died, and answer a second click through `on reopen`. Without the
    flag the applet quits the instant the run handler returns, and its quit handler
    stops the server: the workbench died about a second after the browser opened on
    it. Then the dead applet lingered, and while ANY instance is running macOS answers
    a double-click by activating it rather than running the script — so every click
    after that did nothing at all.

    The flag is one key in Info.plist, and nothing else about the bundle looks wrong.
    osacompile sets it from -s, but SILENTLY IGNORES -s when the .app already exists,
    so rebuilding in place is exactly how it goes missing."""
    import plistlib

    info = plistlib.loads((APP / "Contents" / "Info.plist").read_bytes())

    assert info.get("OSAAppletStayOpen") is True, (
        "the built app is not a stay-open applet, so it will quit the moment it has "
        "started the workbench and stop the server on its way out — rebuild it with "
        '`rm -rf "Junior Workbench.app"` first, or osacompile will ignore -s'
    )


def test_the_handlers_a_stay_open_applet_needs_are_all_there():
    """Each answers a way the old build failed: reopen answers the second click, idle
    notices a server that died on its own, quit is what stops the server now that no
    dialog is holding it."""
    code = source_without_comments()

    for handler in ("on run", "on reopen", "on idle", "on quit"):
        assert handler in code, f"the applet has no {handler!r} handler"
    assert "continue quit" in code, "the quit handler never lets the app actually quit"
    assert "display dialog" not in code, (
        "a modal dialog is back in the applet: it blocks every later launch, which is "
        "what made the app unclickable"
    )


def test_the_rebuild_line_says_to_delete_the_bundle_first():
    """The one instruction that cannot be left out, because leaving it out produces a
    bundle that looks correct, decompiles correctly, and does not work."""
    body = SOURCE.read_text(encoding="utf-8")

    assert "rm -rf" in body and "osacompile -s" in body, (
        "the rebuild instructions do not say to delete the bundle before compiling"
    )
