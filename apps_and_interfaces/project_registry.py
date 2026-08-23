"""The projects Junior offers you when you open the prompt outside a project folder.

Config discovery answers "which project is this directory inside", which is the right
question at a shell and on a cluster: you cd into the cohort, or you pass --config.
It is the wrong question at the prompt, which opens from wherever you happen to be —
answering it from the surroundings there means adopting whatever settings file sits
above you, and a home folder full of unrelated work sits above everything.

So the prompt asks instead, and this is what it asks from. Two halves, because a list of
paths alone rots: a path is a record of where something was, and folders get reorganised.

  look_in   folders that hold projects, searched a couple of levels down for settings
            files. Moving or renaming a project inside one of these keeps it findable,
            and a project restored from a backup or made by a colleague turns up without
            anyone registering it. This is the durable half.
  projects  individual settings files, newest use first. This half is a recency order for
            the menu, and a way to reach a project that lives outside every look_in
            folder. Entries whose file is gone are dropped on the next read.

Dropping dead entries is right for a deleted project and wrong on its own for a moved
one, which still exists and would simply vanish from the menu — which is what look_in is
for. A list that can only lose entries decays to empty as you tidy up.

Nothing here decides anything. A project reached by cd, by --config, or by $JUNIOR_CONFIG
works whether or not it was ever written down, so a lost or hand-mangled file costs you
the menu, never the run.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

from jr_pipeline.runtime_infrastructure.project_context import PROJECT_CONFIG_GLOB

# Outside the checkout, so re-cloning or moving the repo does not lose the list and one
# list serves every checkout. The environment variable moves it: a shared machine may
# want it somewhere per-user, and a test needs it somewhere disposable.
REGISTRY_ENVIRONMENT_VARIABLE = "JUNIOR_PROJECTS"
DEFAULT_REGISTRY_RELATIVE_PATH = Path(".junior") / "projects.yaml"

_FILE_HEADER = """\
# Where Junior looks for your projects. Edit this freely.
#
# look_in   folders that hold projects. Each is searched a couple of levels down for a
#           junior*.yaml, so moving or renaming a project inside one of these keeps it
#           on the menu. Add a folder here and everything in it turns up.
# projects  individual settings files, most recently used first. This sets the order of
#           the menu, and reaches projects that live outside every look_in folder. A
#           line whose file no longer exists is dropped next time Junior reads this.
#
# Being listed is a convenience, not permission: any project still works by cd-ing into
# its folder or passing --config.
"""

# Enough to offer at a prompt. A machine with more projects than this has older ones
# that are reached by cd rather than chosen from a menu, and a menu long enough to
# scroll is one you read past rather than read.
_MOST_TO_REMEMBER = 25

# How far below a look_in folder a project may sit. `new-project` puts one at
# <into>/<name>/junior_<name>.yaml, so one level is the standard layout; two covers the
# person who keeps cohorts grouped by study or by year. Deeper than that is a search of
# somebody's home directory, which is slow and turns up things they did not mean to
# offer — those are what the `projects` list is for.
_HOW_DEEP_TO_LOOK = 2


def registry_path() -> Path:
    """Where the list lives — the default, or wherever $JUNIOR_PROJECTS points.

    Both halves are read when they are asked for rather than when this module loads, so
    the answer follows the environment the call is made in. A home directory resolved at
    import time would be the one the process started with, which is the wrong answer for
    anything that sets HOME around a call, and untestable without reloading the module."""
    override = os.environ.get(REGISTRY_ENVIRONMENT_VARIABLE)
    if override:
        return Path(override).expanduser()
    return Path.home() / DEFAULT_REGISTRY_RELATIVE_PATH


class _Unreadable(Exception):
    """The file is there but could not be parsed — which is not the same as empty."""


def _read() -> tuple[list[str], list[str]]:
    """(look_in folders, project paths) as written, before any of it is checked.

    Raises _Unreadable when the file exists but will not parse. Readers treat that as an
    empty menu; WRITERS must not, because writing an empty list back over a file that was
    merely mid-write, or mid-hand-edit, turns a transient into a permanent loss of the
    durable half of the registry. A missing file is genuinely empty and returns cleanly.

    A bare list is the older one-key format, still read so an existing file keeps
    working."""
    path = registry_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], []
    except OSError as e:
        raise _Unreadable(str(e)) from e
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise _Unreadable(str(e)) from e
    if parsed is None:
        return [], []
    if isinstance(parsed, list):
        return [], [entry for entry in parsed if isinstance(entry, str)]
    if not isinstance(parsed, dict):
        raise _Unreadable("not a mapping")

    def _strings(key: str) -> list[str]:
        listed = parsed.get(key)
        return [entry for entry in listed if isinstance(entry, str)] if isinstance(listed, list) else []

    return _strings("look_in"), _strings("projects")


def _resolved(entry: str) -> Path | None:
    """One written path, expanded and resolved, or None if it cannot be."""
    try:
        return Path(entry).expanduser().resolve()
    except OSError:
        return None


def look_in_folders() -> list[Path]:
    """The folders searched for projects, as written and still existing."""
    kept: list[Path] = []
    try:
        written = _read()[0]
    except _Unreadable:
        return []
    for entry in written:
        folder = _resolved(entry)
        if folder is not None and folder.is_dir() and folder not in kept:
            kept.append(folder)
    return kept


def remembered_projects() -> list[Path]:
    """Every project Junior can offer, most recently used first.

    The written list first, in its order — that is the recency the menu wants — then
    whatever the look_in folders turn up that the list has not already named. Searching
    is what survives a project being moved or renamed: its written path dies with the
    move, but the folder it moved WITHIN is still being searched."""
    kept: list[Path] = []
    seen: set[Path] = set()

    def _keep(path: Path) -> None:
        # Resolved before the duplicate check so two spellings of one path — a symlink,
        # a trailing slash, `~` against an absolute — cannot both sit in the menu, and
        # so a project reached by both routes appears once.
        if path in seen or not path.is_file():
            return
        seen.add(path)
        kept.append(path)

    try:
        listed = _read()[1]
    except _Unreadable:
        listed = []
    for entry in listed:
        resolved = _resolved(entry)
        if resolved is not None:
            _keep(resolved)
    for folder in look_in_folders():
        for found in _search(folder):
            _keep(found)
    return kept


def _search(folder: Path) -> list[Path]:
    """Settings files under one look_in folder, nearest first.

    A folder holding two settings files is not a project — config discovery refuses that
    case as ambiguous — so it is skipped here too rather than offered as two menu rows
    that run the same cohort."""
    found: list[Path] = []
    for depth in range(1, _HOW_DEEP_TO_LOOK + 1):
        pattern = "/".join(["*"] * depth + [PROJECT_CONFIG_GLOB])
        try:
            matches = sorted(folder.glob(pattern))
        except OSError:
            continue
        by_folder: dict[Path, list[Path]] = {}
        for match in matches:
            if match.is_file() and not _is_a_template(match):
                by_folder.setdefault(match.parent, []).append(match)
        found.extend(
            alone[0] for _parent, alone in sorted(by_folder.items()) if len(alone) == 1
        )
    return found


# A settings file shipped as a worked example rather than written for a cohort. The
# repo's own is junior.example.yaml, whose first line says it "is intentionally suitable
# only for the bundled synthetic chart" and whose paths are unset env placeholders.
# It matches junior*.yaml like anything else, so a look_in folder containing a checkout
# offered it as a project named after its parent directory — "config" — and choosing it
# pointed a run at nothing.
_TEMPLATE_MARKERS = ("example", "template", "sample")


def _is_a_template(config_path: Path) -> bool:
    """Whether this settings file is a worked example rather than somebody's project.

    Read from the filename, which is the only thing true before the file is parsed and
    the only thing an author controls when shipping one. Discovery only: a path given
    with --config, or found by standing in its folder, is a deliberate choice and is
    honoured."""
    return any(marker in config_path.name.lower() for marker in _TEMPLATE_MARKERS)


def remember(config_path: Path) -> None:
    """Put a project at the top of the list, creating the file if this is the first.

    Called when a project is created and again whenever one is chosen, so the order is
    recency of use rather than of creation: the cohort you are working on this month is
    the one at the top of the menu."""
    chosen = _resolved(str(config_path))
    if chosen is None:
        return
    try:
        look_in, _ = _read()
    except _Unreadable:
        # Refuse to rewrite a file we could not read. It may be another process's
        # half-written bytes, or a hand-edit in progress; either way, writing our empty
        # view back would make a transient into a permanent loss of `look_in`.
        return
    ordered = [chosen, *(path for path in remembered_projects() if path != chosen)]
    _write(ordered[:_MOST_TO_REMEMBER], look_in)


def watch_folder(folder: Path) -> None:
    """Search this folder for projects from now on.

    The parent a project was created in is worth watching: it is where the next one will
    go, and where this one will still be after it is renamed."""
    watched = _resolved(str(folder))
    if watched is None or not watched.is_dir() or watched == Path.home():
        # Home itself is excluded: two levels below it is most of the machine, and a
        # settings file loose in the home folder is the case the prompt already treats
        # as suspect rather than as a place to go looking for more.
        return
    try:
        look_in, projects = _read()
    except _Unreadable:
        return
    if any(_resolved(entry) == watched for entry in look_in):
        return
    _write([path for path in (_resolved(p) for p in projects) if path], [*look_in, str(watched)])


def forget(config_path: Path) -> None:
    """Take a project off the written list.

    Its folder and its runs are left alone — and if it sits inside a watched folder it
    stays on the menu, because that is where the menu found it. Removing it for good
    means taking that folder out of look_in, which is a different thing to want and is
    done by editing the file."""
    dropped = _resolved(str(config_path))
    if dropped is None:
        return
    try:
        look_in, projects = _read()
    except _Unreadable:
        return
    kept = [
        path for path in (_resolved(entry) for entry in projects)
        if path and path != dropped
    ]
    _write(kept, look_in)


def _write(projects: list[Path], look_in: list[str] | None = None) -> None:
    """Replace the file. Failure is silent: the menu is a convenience, and a read-only
    home directory or a full disk must not stop a project from being created."""
    path = registry_path()
    body = yaml.safe_dump(
        {
            "look_in": [str(folder) for folder in (look_in or [])],
            "projects": [str(project) for project in projects],
        },
        sort_keys=False, default_flow_style=False,
    )
    # Written whole, then moved into place: os.replace is atomic within a filesystem, so
    # a second `junior` reading this file sees either the old contents or the new ones,
    # never the middle of a write. It cannot stop two processes overwriting each other's
    # last entry, but it does stop a torn read — which mattered more, because a torn read
    # looks like an empty registry.
    import os
    import tempfile

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as writing:
            writing.write(f"{_FILE_HEADER}{body}")
        os.replace(temporary, path)
    except OSError:
        return


def match_project(typed: str, choices: list[Path]) -> Path | None:
    """Resolve what someone typed against a menu of projects: a number, a name, a path.

    A path is accepted because the list cannot offer a project it has never been told
    about — one on a mounted drive, one a colleague set up, one restored from a backup —
    and refusing the path to a settings file that plainly exists would be a menu standing
    in the way of the thing it exists to reach."""
    typed = typed.strip()
    if typed.isdigit():
        position = int(typed)
        return choices[position - 1] if 1 <= position <= len(choices) else None
    for path in choices:
        if project_name(path).lower() == typed.lower():
            return path

    named = Path(typed).expanduser()
    if named.is_file():
        return named.resolve()
    # A project's folder rather than its settings file — what dragging the folder into
    # the terminal gives you, which is how a path gets typed here at all. Only when the
    # folder holds exactly one settings file: two would make this a guess.
    if named.is_dir():
        inside = sorted(path for path in named.glob(PROJECT_CONFIG_GLOB) if path.is_file())
        if len(inside) == 1:
            return inside[0].resolve()
    return None


def project_where_it_lives(config_path: Path) -> str:
    """Where this project IS, shortened for display — its folder, or its settings file.

    The folder, because that is a project's identity: it is what you cd to, what holds
    the runs, and what you go looking for when a name on a menu means nothing to you.
    Showing what a cohort READS instead was worse in both directions — several cohorts
    over one export all read the same folder and so all looked identical, and a project
    whose settings sit loose in the home folder had no row anywhere saying so, which
    made it a name you could not find on disk.

    The settings file rather than the folder when the file is not in a folder of its
    own, since "~" is not an answer to where a project lives."""
    path = Path(config_path)
    shown = path if path.parent == Path.home() else path.parent
    try:
        return f"~/{shown.relative_to(Path.home())}"
    except ValueError:
        return str(shown)


def project_name(config_path: Path) -> str:
    """What to call a project in a menu.

    The `project` key in its settings file is the name it was created with, which is
    what the operator typed and what `status` prints. The filename and then the folder
    stand in for it when that key is missing, as it is in files written by earlier
    versions — every project has a name in a listing, even the ones that never set one.
    """
    try:
        parsed = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        named = parsed.get("project") if isinstance(parsed, dict) else None
        if isinstance(named, str) and named.strip():
            return named.strip()
    except (OSError, yaml.YAMLError):
        pass
    stem = Path(config_path).stem
    if stem.startswith("junior_") and len(stem) > len("junior_"):
        return stem[len("junior_"):]
    return Path(config_path).parent.name
