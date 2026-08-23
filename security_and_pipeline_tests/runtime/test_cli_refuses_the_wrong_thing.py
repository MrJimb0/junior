"""The CLI must fail rather than quietly do something the operator did not ask for.

Every case here was found by deliberately mis-using the CLI. The worst was not a
crash: run outside a project and it would try to CONTINUE whichever run happened to be
newest in the checkout's own data/ — appending to a real clinical audit trail because
a command was typed in the wrong directory. Silence is the failure mode that matters
in a tool whose output is a research dataset.
"""
from __future__ import annotations

from pathlib import Path

import click
import pytest
import yaml

from apps_and_interfaces.command_line_interface import (
    cohort_patients,
    load_cohort_config,
)
from jr_pipeline.runtime_infrastructure.project_context import (
    PROJECT_CONFIG_NAME,
    resolve_run_id,
)


@pytest.fixture(autouse=True)
def _no_ambient_environment(monkeypatch):
    for name in ("JUNIOR_CONFIG", "JR_RUN_ID", "JR_DATA_ROOT", "JR_SOURCE_ROOT"):
        monkeypatch.delenv(name, raising=False)


def _existing_run(data_root: Path, run_id: str) -> None:
    (data_root / "CONTAINS_PHI" / "pipeline_run_receipts" / run_id).mkdir(parents=True)


def test_outside_a_project_it_never_joins_an_existing_run(tmp_path, monkeypatch):
    """The dangerous one. Inside a project, continuing the newest run is what makes
    `ingest` then `embed` land together. Outside one, the newest run belongs to
    somebody else's data and must not be appended to."""
    _existing_run(tmp_path, "20260101_000000_aa")
    monkeypatch.chdir(tmp_path)

    joined = resolve_run_id({}, None, data_root=tmp_path, continue_newest=False)
    assert joined != "20260101_000000_aa", "continued a run it was not asked to"


def test_inside_a_project_it_does_continue_the_newest_run(tmp_path):
    """The counterpart — the guard must not have cost the useful behaviour."""
    _existing_run(tmp_path, "20260101_000000_aa")
    assert resolve_run_id({}, None, data_root=tmp_path, continue_newest=True) == (
        "20260101_000000_aa"
    )


def test_passing_no_data_root_does_not_silently_re_enable_it(tmp_path, monkeypatch):
    """The first attempt at this fix passed data_root=None, which looked like it
    disabled the lookup but fell through to the ambient root instead."""
    _existing_run(tmp_path, "20260101_000000_aa")
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))

    assert resolve_run_id({}, None, data_root=None, continue_newest=False) != (
        "20260101_000000_aa"
    )


def test_running_with_no_project_says_where_the_work_is_going(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    load_cohort_config(None, None)
    reported = capsys.readouterr().err
    assert "no project here" in reported
    assert "reading" in reported and "writing" in reported


# --- telling three mistakes apart ------------------------------------------------


def test_a_source_folder_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(click.ClickException, match="does not exist"):
        cohort_patients({"source_root": str(tmp_path / "nope")})


def test_an_empty_source_folder_says_it_is_empty(tmp_path):
    with pytest.raises(click.ClickException, match="empty"):
        cohort_patients({"source_root": str(tmp_path)})


def test_one_patients_folder_is_recognised_as_such(tmp_path):
    """The most common mis-aim: pointing at a patient instead of the cohort. All
    three of these used to produce the same 'none found' message."""
    (tmp_path / "clinical_note.csv").write_text("patient_id,text\nP1,hi\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as raised:
        cohort_patients({"source_root": str(tmp_path)})
    message = str(raised.value)
    assert "ONE patient's folder" in message
    assert str(tmp_path.parent) in message, "does not name the folder to use instead"


# --- broken settings files --------------------------------------------------------


def test_bad_yaml_is_a_sentence_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / PROJECT_CONFIG_NAME).write_text("run_id: [oops: bad\n", encoding="utf-8")

    with pytest.raises(click.ClickException) as raised:
        load_cohort_config(None, None)
    message = str(raised.value)
    assert "not valid YAML" in message
    assert "line" in message, "does not say where the problem is"
    assert "Traceback" not in message


def test_a_valid_config_still_loads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / PROJECT_CONFIG_NAME).write_text(
        yaml.safe_dump({"project": "p", "run_id": None,
                        "source_root": str(tmp_path), "output_root": str(tmp_path)}),
        encoding="utf-8",
    )
    cfg, origin = load_cohort_config(None, None)
    assert cfg["project"] == "p"
    assert PROJECT_CONFIG_NAME in origin


# --- picking somebody else's cohort ------------------------------------------------


def _project(directory: Path, name: str, **extra) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / f"junior_{name}.yaml"
    config.write_text(
        yaml.safe_dump({"project": name, "run_id": None,
                        "source_root": str(directory / "in"),
                        "output_root": str(directory), **extra}),
        encoding="utf-8",
    )
    return config


def test_a_settings_file_in_the_home_folder_says_it_has_claimed_everything(
    tmp_path, monkeypatch, capsys
):
    """Discovery walks upward, so a project settings file left in a home directory
    quietly owns every command run anywhere below it — an unrelated checkout, a scratch
    directory, another cohort with no settings file of its own. Honoured, but said."""
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    _project(tmp_path, "stray")
    working_directory = tmp_path / "somewhere" / "else"
    working_directory.mkdir(parents=True)
    monkeypatch.chdir(working_directory)

    load_cohort_config(None, None)
    reported = capsys.readouterr().err
    assert "home folder" in reported
    assert "junior_stray.yaml" in reported


def test_a_project_whose_output_folder_has_moved_says_so(tmp_path, monkeypatch, capsys):
    """A project pins its own folder as output_root. Rename the folder and the settings
    travel with it while the pinned path does not exist — so the layout is recreated
    empty there, `status` reports a cohort with no runs, and the next stage starts the
    whole thing again somewhere nobody is looking."""
    moved_to = tmp_path / "renamed"
    (moved_to / "CONTAINS_PHI" / "pipeline_run_receipts").mkdir(parents=True)
    _project(moved_to, "cohort", output_root=str(tmp_path / "its_old_name"))
    monkeypatch.chdir(moved_to)

    load_cohort_config(None, None)
    reported = capsys.readouterr().err
    assert "not there any more" in reported
    assert str(moved_to) in reported, "does not say where the runs actually are"


def test_export_metadata_packages_the_project_you_are_standing_in(tmp_path, monkeypatch):
    """This is the artifact you hand a collaborator, and it was the one advertised
    command that never resolved a project: it used whichever data root was last bound
    in the process, so running it in one cohort after touching another zipped up the
    other one and reported success."""
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    theirs = tmp_path / "theirs"
    (theirs / "NO_PHI__shareable" / "20260101_000000_aaaa").mkdir(parents=True)
    mine = tmp_path / "mine"
    _project(mine, "mine")
    monkeypatch.setenv("JR_DATA_ROOT", str(theirs))  # left over from the other cohort
    monkeypatch.chdir(mine)

    result = CliRunner().invoke(main, ["export-metadata", "--run-id", "20260101_000000_aaaa"])
    assert result.exit_code != 0, "it exported another cohort's run"
    assert str(mine) in result.output


def test_run_uses_the_project_you_are_standing_in(tmp_path, monkeypatch):
    """`run` loaded the bundled config outright, so standing in a cohort folder and
    typing it read the checkout's own examples and wrote to the checkout's own data —
    the settings of the project the operator was standing in were never consulted."""
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    cohort = tmp_path / "cohort"
    (cohort / "in" / "P1").mkdir(parents=True)
    _project(cohort, "cohort", recipes=["date_of_birth"])
    monkeypatch.chdir(cohort)

    started: list = []
    for name, stub in (
        ("run_cohort", lambda settings, run_id=None: started.append(settings) or {}),
        ("view_results", lambda *_a, **_k: None),
        ("check_phi_containment", lambda *_a, **_k: None),
    ):
        monkeypatch.setattr(
            f"jr_pipeline.runtime_infrastructure.cohort_runner.{name}", stub
        )

    result = CliRunner().invoke(main, ["run"])
    assert result.exit_code == 0, result.output
    assert started, "run never started"
    assert started[0].data_root == cohort, "it ran against a different cohort"
    assert started[0].input_folder == cohort / "in"


def test_run_extracts_the_variables_the_project_names(tmp_path, monkeypatch):
    """Every config writer and every other reader uses `recipes`. Reading only
    `variables` meant no config ever matched, so a cohort created with a chosen list
    silently extracted the built-in default instead — while the sealed bundle, built
    from the file, recorded the list that had been asked for."""
    from jr_pipeline.runtime_infrastructure.cohort_runner import (
        cohort_settings_from_config,
    )

    settings = cohort_settings_from_config(
        {"recipes": ["date_of_death"], "output_root": str(tmp_path)}, variables=()
    )
    assert settings.variables == ["date_of_death"]
