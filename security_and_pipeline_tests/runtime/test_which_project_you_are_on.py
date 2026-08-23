"""Which cohort a session works on is chosen, not inherited from the terminal's folder.

Config discovery walks up from the current directory, which is the right answer at a
shell and on a cluster: you cd into the cohort, or you pass --config. It is the wrong
answer at the prompt, which opens wherever the terminal happened to be — and a settings
file in the home folder is above every directory on the machine, so it would answer for
all of them. These tests pin that the prompt asks instead, that the answer holds for the
commands after it, and that nothing about a scripted or piped run changed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from apps_and_interfaces import project_registry
from apps_and_interfaces.command_line_interface import main
from apps_and_interfaces.interactive_session import start
from jr_pipeline.runtime_infrastructure.project_context import CONFIG_ENVIRONMENT_VARIABLE


def _project(folder: Path, name: str) -> Path:
    """A folder with a settings file in it — the least that counts as a project.

    Carries a real `source_root`, because that is what the menu shows for each row and
    a helper that omitted it meant every row fell back to the project folder, leaving
    `project_where_it_lives` unexercised by the whole file."""
    folder.mkdir(parents=True, exist_ok=True)
    charts = folder / "charts"
    charts.mkdir(exist_ok=True)
    config = folder / f"junior_{name}.yaml"
    config.write_text(
        yaml.safe_dump({"project": name, "run_id": None, "source_root": str(charts)}),
        encoding="utf-8",
    )
    return config


@pytest.fixture
def registry(tmp_path, monkeypatch) -> Path:
    """Point the projects list at a disposable file, never the operator's own."""
    path = tmp_path / "registry" / "projects.yaml"
    monkeypatch.setenv(project_registry.REGISTRY_ENVIRONMENT_VARIABLE, str(path))
    return path


@pytest.fixture
def somewhere_that_is_not_a_project(tmp_path, monkeypatch) -> Path:
    """Stand in an empty folder whose parents hold no settings file either."""
    elsewhere = tmp_path / "not_a_project"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # Home is inside tmp_path too, so the walk up from here cannot escape into the
    # real home directory and find whatever the person running the tests has there.
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
    return elsewhere


def _session(monkeypatch, capsys, answers: list[str]) -> str:
    """Open a session with someone at the keyboard, and return everything it printed."""
    remaining = iter([*answers, "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))
    monkeypatch.setattr("apps_and_interfaces.interactive_session.is_interactive",
                        lambda: True)
    start(main, "0.0.1")
    return capsys.readouterr().out


# --- being asked ------------------------------------------------------------------

def test_a_prompt_opened_outside_a_project_asks_which_one(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    first = _project(tmp_path / "breast2026", "breast2026")
    second = _project(tmp_path / "lung2026", "lung2026")
    project_registry.remember(first)
    project_registry.remember(second)

    out = _session(monkeypatch, capsys, ["breast2026"])

    assert "Which project?" in out
    assert "breast2026" in out and "lung2026" in out
    # The choice is held for the commands that follow, which is the whole point of
    # asking: without this the next command walks up from the folder again.
    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(first)


def test_the_menu_takes_a_number_as_readily_as_a_name(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    only = _project(tmp_path / "breast2026", "breast2026")
    project_registry.remember(only)

    _session(monkeypatch, capsys, ["1"])

    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(only)


def test_choosing_a_project_moves_it_to_the_top_of_the_list(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    older = _project(tmp_path / "breast2026", "breast2026")
    newer = _project(tmp_path / "lung2026", "lung2026")
    project_registry.remember(older)
    project_registry.remember(newer)
    assert project_registry.remembered_projects()[0] == newer

    _session(monkeypatch, capsys, ["breast2026"])

    # Recency of use, not of creation: the cohort you are working on this month is the
    # one the menu offers first next time.
    assert project_registry.remembered_projects()[0] == older


def test_a_project_the_list_never_heard_of_is_reachable_by_path(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    unlisted = _project(tmp_path / "from_a_colleague", "from_a_colleague")

    _session(monkeypatch, capsys, [str(unlisted.parent)])

    # Its folder, not its settings file — a path gets here by being dragged into the
    # terminal, and what you drag is the folder.
    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(unlisted)


# --- not being asked --------------------------------------------------------------

def test_standing_inside_a_project_is_itself_the_answer(
    tmp_path, monkeypatch, capsys, registry
):
    config = _project(tmp_path / "breast2026", "breast2026")
    monkeypatch.chdir(config.parent)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)

    out = _session(monkeypatch, capsys, [])

    assert "Which project?" not in out, "asked a question the surroundings answered"
    assert "Working on" in out and "breast2026" in out
    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(config)


def test_a_piped_session_is_asked_nothing_and_pinned_to_nothing(
    tmp_path, monkeypatch, capsys, registry
):
    """A script has nobody at the keyboard, and discovery already answers for it."""
    config = _project(tmp_path / "breast2026", "breast2026")
    monkeypatch.chdir(config.parent)
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    remaining = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))
    monkeypatch.setattr("apps_and_interfaces.interactive_session.is_interactive",
                        lambda: False)

    start(main, "0.0.1")
    out = capsys.readouterr().out

    assert "Which project?" not in out
    assert CONFIG_ENVIRONMENT_VARIABLE not in os.environ
    # Nor written down: a run nobody chose anything in must not reorder the menu.
    assert not registry.exists()


# --- the home folder --------------------------------------------------------------

def test_a_settings_file_in_the_home_folder_is_offered_rather_than_assumed(
    tmp_path, monkeypatch, capsys, registry
):
    """The failure this whole question exists for.

    A junior.yaml directly in the home folder is above nearly every directory on the
    machine, so the walk up finds it from anywhere — and a prompt opened in an unrelated
    folder would silently adopt a cohort three directories up, mint a fresh run under
    it, and report that nothing had been ingested."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    stray = _project(home, "an_old_cohort")
    # Standing below it, the way every terminal on the machine is.
    working = home / "somewhere" / "else"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)

    out = _session(monkeypatch, capsys, ["quit"])

    assert "Which project?" in out
    assert "an_old_cohort" in out, "the config found above was not on the menu"
    # Offered like any other cohort. Choosing it explicitly is safe — what was unsafe
    # was it being adopted without anyone choosing, which asking prevents. So the menu
    # handles it by ordering and by the default, not by explaining a hazard that no
    # longer applies to the reader standing in front of the list.
    assert "an_old_cohort" in out
    # A project whose settings sit loose in the home folder has no folder of its own, so
    # the row names the file. Without it, "an_old_cohort" was a name with nothing on disk
    # to match it — the reader goes looking for a folder that was never there.
    assert stray.name in out, out


def test_pressing_enter_never_adopts_a_home_folder_config(
    tmp_path, monkeypatch, capsys, registry
):
    """Enter is the path of least resistance, so it must not lead to the flagged one.

    Sinking a home-folder config below the others is enough when there are others. When
    it is the whole menu, the default becomes setting one up — the only answer left that
    ends somewhere defined."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    _project(home, "an_old_cohort")
    working = home / "somewhere" / "else"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    dispatched: list[list[str]] = []
    monkeypatch.setattr("apps_and_interfaces.interactive_session._dispatch",
                        lambda _group, arguments: dispatched.append(arguments))

    _session(monkeypatch, capsys, [""])

    assert CONFIG_ENVIRONMENT_VARIABLE not in os.environ
    assert dispatched == [["new-project"]]


def test_a_home_folder_config_sinks_below_the_ordinary_ones(
    tmp_path, monkeypatch, capsys, registry
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    _project(home, "an_old_cohort")
    ordinary = _project(home / "Desktop" / "breast2026", "breast2026")
    project_registry.remember(ordinary)
    working = home / "somewhere" / "else"
    working.mkdir(parents=True)
    monkeypatch.chdir(working)

    # Enter, so what the menu put first is what gets chosen.
    _session(monkeypatch, capsys, [""])

    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(ordinary)


def test_there_is_no_way_to_carry_on_without_a_project(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    """The menu must not offer the state the question exists to prevent.

    A session with nothing pinned falls back to the bundled example settings, which read
    the checkout's own examples/ and write its data/ — and `resolve_run_id` would then be
    continuing whatever run happened to be newest in there. An earlier version of this
    menu offered exactly that, described as carrying on without one."""
    monkeypatch.delenv(CONFIG_ENVIRONMENT_VARIABLE, raising=False)
    project_registry.remember(_project(tmp_path / "breast2026", "breast2026"))

    for refusal in ("none", "skip", "stay put", "no project"):
        out = _session(monkeypatch, capsys, [refusal])
        assert CONFIG_ENVIRONMENT_VARIABLE not in os.environ, (
            f"{refusal!r} pinned something"
        )
        assert "No project called" in out, f"{refusal!r} was treated as an answer"


def test_quit_at_the_project_question_leaves_the_session(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    """`quit` is what leaves a session, and this question is the front of one."""
    project_registry.remember(_project(tmp_path / "breast2026", "breast2026"))
    remaining = iter(["quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))
    monkeypatch.setattr("apps_and_interfaces.interactive_session.is_interactive",
                        lambda: True)

    start(main, "0.0.1")
    out = capsys.readouterr().out

    from apps_and_interfaces.command_line_interface import PIPELINE_PHASES

    # The command listing is printed on the way into the loop, so its absence is how
    # a session that ended at the question is told from one that carried on.
    assert next(iter(PIPELINE_PHASES)).upper() not in out


# --- the list itself --------------------------------------------------------------

def test_a_project_whose_folder_is_gone_drops_off_the_list(tmp_path, registry):
    config = _project(tmp_path / "deleted_later", "deleted_later")
    project_registry.remember(config)
    assert project_registry.remembered_projects() == [config]

    config.unlink()

    # Dropped rather than reported. A project you deleted is not a problem to be told
    # about every time you open the prompt.
    assert project_registry.remembered_projects() == []


def test_a_hand_mangled_list_costs_the_menu_and_nothing_else(registry):
    """The file invites editing, so a stray character must not stand between an
    operator and their cohort."""
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("projects: [unclosed\n  - nonsense: {{{", encoding="utf-8")

    assert project_registry.remembered_projects() == []


def test_the_default_list_follows_the_home_directory(tmp_path, monkeypatch):
    """Resolved when asked for, not when the module loaded.

    A home directory read at import time is the one the process started with, which is
    the wrong answer anywhere HOME is set around a call — and untestable without
    reloading the module."""
    monkeypatch.delenv(project_registry.REGISTRY_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "elsewhere"))

    assert project_registry.registry_path() == (
        tmp_path / "elsewhere" / ".junior" / "projects.yaml"
    )


def test_with_no_projects_at_all_enter_offers_to_make_one(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    """Nothing to choose from, so the default is the only thing that moves you forward."""
    dispatched: list[list[str]] = []
    monkeypatch.setattr("apps_and_interfaces.interactive_session._dispatch",
                        lambda _group, arguments: dispatched.append(arguments))

    out = _session(monkeypatch, capsys, [""])

    assert "No projects on this machine." in out
    assert dispatched == [["new-project"]]


def test_the_menu_numbers_projects_and_spells_out_the_rest(
    tmp_path, monkeypatch, capsys, registry, somewhere_that_is_not_a_project
):
    """Numbers address projects, words address actions, so the two never collide.

    The single letters an earlier version used (`n`, `s`) competed with the numbering
    for the same keystroke, and `n` could have been either `new` or `none`."""
    project_registry.remember(_project(tmp_path / "breast2026", "breast2026"))

    out = _session(monkeypatch, capsys, ["quit"])

    assert "1  breast2026" in out, "projects must be numbered"
    # The action row carries no number, so a number can only ever mean a project.
    action = [line for line in out.splitlines() if line.strip().startswith("new")]
    assert action, out
    assert not any(char.isdigit() for char in action[0]), action[0]


def test_a_project_renamed_in_place_stays_on_the_menu(tmp_path, registry):
    """The failure a list of paths always has: a path records where something was.

    Renaming or moving a project kills its written path, and dropping dead entries is
    right for a deletion and wrong here — the project still exists. A list that can only
    lose entries decays to empty as you tidy up."""
    config = _project(tmp_path / "desk" / "breast2026", "breast2026")
    project_registry.remember(config)
    project_registry.watch_folder(tmp_path / "desk")

    (tmp_path / "desk" / "breast2026").rename(tmp_path / "desk" / "breast_2027")

    remembered = project_registry.remembered_projects()
    assert [project_registry.project_name(path) for path in remembered] == ["breast2026"]
    assert remembered[0] == tmp_path / "desk" / "breast_2027" / "junior_breast2026.yaml"


def test_a_deleted_project_still_drops_off(tmp_path, registry):
    config = _project(tmp_path / "desk" / "breast2026", "breast2026")
    project_registry.remember(config)
    project_registry.watch_folder(tmp_path / "desk")

    import shutil

    shutil.rmtree(tmp_path / "desk" / "breast2026")

    assert project_registry.remembered_projects() == []


def test_a_project_grouped_under_a_subfolder_is_still_found(tmp_path, registry):
    """Cohorts kept in groups — by study, by year — sit two levels below the watched
    folder, not one. Searching only one level found nothing for that layout, and every
    other test here happened to use the flat one."""
    grouped = _project(tmp_path / "desk" / "study_2026" / "breast2026", "breast2026")
    project_registry.watch_folder(tmp_path / "desk")

    assert project_registry.remembered_projects() == [grouped]


def test_a_project_nobody_registered_is_found_in_a_watched_folder(tmp_path, registry):
    """One made by a colleague, restored from a backup, or copied off a drive."""
    unregistered = _project(tmp_path / "desk" / "from_a_colleague", "from_a_colleague")
    project_registry.watch_folder(tmp_path / "desk")

    assert project_registry.remembered_projects() == [unregistered]


def test_watching_never_swallows_the_home_folder(tmp_path, monkeypatch, registry):
    """Two levels below home is most of the machine, and a settings file loose in the
    home folder is the case the prompt already treats as suspect."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    project_registry.watch_folder(home)

    assert project_registry.look_in_folders() == []


def test_a_folder_holding_two_settings_files_is_not_offered_twice(tmp_path, registry):
    """Config discovery calls that folder ambiguous and refuses it, so the menu must not
    render it as two rows that run the same cohort."""
    muddle = tmp_path / "desk" / "muddle"
    _project(muddle, "one")
    _project(muddle, "two")
    project_registry.watch_folder(tmp_path / "desk")

    assert project_registry.remembered_projects() == []


def test_the_older_one_key_file_still_reads(tmp_path, registry):
    """A file written before look_in existed is a bare mapping with only `projects`."""
    config = _project(tmp_path / "breast2026", "breast2026")
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(yaml.safe_dump({"projects": [str(config)]}), encoding="utf-8")

    assert project_registry.remembered_projects() == [config]


def test_the_list_lives_where_the_environment_says(tmp_path, monkeypatch):
    elsewhere = tmp_path / "somewhere" / "of_my_choosing.yaml"
    monkeypatch.setenv(project_registry.REGISTRY_ENVIRONMENT_VARIABLE, str(elsewhere))

    project_registry.remember(_project(tmp_path / "breast2026", "breast2026"))

    assert elsewhere.is_file()
    assert project_registry.registry_path() == elsewhere


def test_the_list_is_plain_paths_a_person_can_edit(tmp_path, registry):
    config = _project(tmp_path / "breast2026", "breast2026")
    project_registry.remember(config)

    written = registry.read_text(encoding="utf-8")

    assert written.lstrip().startswith("#"), "no explanation for whoever opens this"
    assert str(config) in written
    assert yaml.safe_load(written)["projects"] == [str(config)]


def test_a_project_is_written_down_when_it_is_created(tmp_path, monkeypatch, registry):
    from click.testing import CliRunner

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        main, ["new-project", "breast2026", "--into", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    remembered = project_registry.remembered_projects()
    assert remembered == [tmp_path / "breast2026" / "junior_breast2026.yaml"]


def test_a_name_is_what_the_project_was_created_with(tmp_path):
    config = _project(tmp_path / "some_folder", "breast2026")

    # The folder and the filename both say something else; the settings file is what
    # `status` prints and what the operator typed.
    assert project_registry.project_name(config) == "breast2026"


def test_a_settings_file_with_no_name_still_gets_one(tmp_path):
    """Files written by earlier versions have no `project` key."""
    folder = tmp_path / "an_old_cohort"
    folder.mkdir()
    config = folder / "junior.yaml"
    config.write_text(yaml.safe_dump({"run_id": None}), encoding="utf-8")

    assert project_registry.project_name(config) == "an_old_cohort"


def test_a_worked_example_is_not_offered_as_a_project(tmp_path, monkeypatch):
    """`junior-public/config/junior.example.yaml` turned up on the menu as a project
    called "config".

    It matches junior*.yaml like anything else, so a look_in folder containing a
    checkout finds it two levels down. It has no `project:` key, so the menu named it
    after its parent directory. Its own first line says it is "intentionally suitable
    only for the bundled synthetic chart", its source and output roots are unset env
    placeholders, and its models may download from Hugging Face on first use — so
    choosing it pointed a run at nothing.
    """
    from apps_and_interfaces.project_registry import remembered_projects

    desk = tmp_path / "desk"
    checkout = desk / "junior-public" / "config"
    checkout.mkdir(parents=True)
    (checkout / "junior.example.yaml").write_text("source_root: null\n", encoding="utf-8")
    real = desk / "my_cohort"
    real.mkdir()
    (real / "junior_my_cohort.yaml").write_text("project: my_cohort\n", encoding="utf-8")

    registry = tmp_path / "projects.yaml"
    registry.write_text(f"look_in:\n- {desk}\nprojects: []\n", encoding="utf-8")
    monkeypatch.setenv("JUNIOR_PROJECTS", str(registry))

    # By filename: the temp directory pytest builds for this test has "example" in its
    # own path, which a substring check on the whole path would match.
    listed = [Path(path).name for path in remembered_projects()]

    assert "junior_my_cohort.yaml" in listed
    assert "junior.example.yaml" not in listed, listed


def test_a_template_named_outright_is_still_honoured(tmp_path):
    """Discovery only. Passing one to --config, or standing in its folder, is a
    deliberate choice — the demo is meant to be runnable."""
    from apps_and_interfaces.project_registry import _is_a_template

    assert _is_a_template(Path("junior.example.yaml"))
    assert _is_a_template(Path("junior_TEMPLATE.yaml"))
    assert not _is_a_template(Path("junior_my_cohort.yaml"))
    assert not _is_a_template(Path("junior.yaml"))


def test_the_recipe_list_is_reachable_by_the_word_used_everywhere_else():
    """It was `junior variables`, and nothing else calls them variables at the point
    somebody goes looking: the folder is var_extraction_recipes/, the config key is
    `recipes:`, the run close-out asks "Editing a recipe?". Searching help for "recipe"
    found no way to list them, which is how you conclude there is not one."""
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    runner = CliRunner()
    top_level = runner.invoke(main, ["--help"]).output

    assert "recipes" in top_level

    listed = runner.invoke(main, ["recipes"]).output
    assert "date_of_birth" in listed and "stage" in listed
    # Grouped by collection, and marked for what this project actually extracts.
    assert "oncology" in listed and "✓" in listed


def test_the_old_name_still_works():
    """`variables` is what earlier notes, messages and habits say."""
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    by_old_name = CliRunner().invoke(main, ["variables"])

    assert by_old_name.exit_code == 0
    assert "date_of_birth" in by_old_name.output


def test_nothing_still_sends_people_to_the_old_name():
    """A message naming a command teaches the word for it. Two words for one list is
    how this got lost."""
    from pathlib import Path as _Path

    cli = _Path(__file__).resolve().parents[2] / "apps_and_interfaces/command_line_interface.py"
    body = cli.read_text(encoding="utf-8")

    assert "junior variables" not in body, "a message still points at the older name"
