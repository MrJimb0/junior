"""Setting up a cohort gives it a folder, a settings file, and its data directories.

Two properties matter most. Continuity: what this writes must be what `junior ingest`
then picks up — it previously wrote a Python file for a different entry point, so the
cohort you had just described was silently not the one that ran. And separation: each
cohort owns a folder, so a second `new-project` cannot overwrite the first one's
settings, which is what happened when every project wrote ./junior.yaml.

The location question asks for the PARENT. Answering "/Desktop" means "/Desktop/<name>",
not "put my PHI tree loose on the Desktop".

It must also stay scriptable: a piped or CI invocation cannot block on a question.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main
from jr_pipeline.runtime_infrastructure.project_context import project_config_name


def _answers(monkeypatch, replies: list[str]) -> None:
    """Pretend to be a terminal, and answer each prompt in turn."""
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive",
        lambda: True,
    )
    remaining = iter(replies)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(remaining))


def _piped(monkeypatch) -> None:
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive",
        lambda: False,
    )
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface._is_interactive",
        lambda: False,
    )


def _settings(project_dir: Path) -> dict:
    """The project's settings file, found by name rather than assumed."""
    return yaml.safe_load(
        (project_dir / project_config_name(project_dir.name)).read_text(encoding="utf-8")
    )


def test_the_project_gets_its_own_folder(monkeypatch, tmp_path):
    """Answering "/Desktop" must not scatter CONTAINS_PHI across the Desktop."""
    _answers(monkeypatch, ["study_a", str(tmp_path), ""])
    result = CliRunner().invoke(main, ["new-project"])
    assert result.exit_code == 0, result.output

    project = tmp_path / "study_a"
    assert (project / project_config_name(project.name)).is_file()
    assert (project / "CONTAINS_PHI").is_dir()
    assert (project / "NO_PHI__shareable").is_dir()
    assert not (tmp_path / "CONTAINS_PHI").exists(), "PHI tree landed in the parent"


def test_two_cohorts_live_side_by_side(monkeypatch, tmp_path):
    """The failure this prevents: the second new-project offering to overwrite the
    first one's settings, because both wrote ./junior.yaml."""
    for cohort in ("study_a", "study_b"):
        _answers(monkeypatch, [cohort, str(tmp_path), ""])
        result = CliRunner().invoke(main, ["new-project"])
        assert result.exit_code == 0, result.output
        assert "already lives" not in result.output

    assert _settings(tmp_path / "study_a")["project"] == "study_a"
    assert _settings(tmp_path / "study_b")["project"] == "study_b"


def test_reusing_a_name_in_the_same_place_is_a_real_collision(monkeypatch, tmp_path):
    _answers(monkeypatch, ["study_a", str(tmp_path), ""])
    CliRunner().invoke(main, ["new-project"])

    _piped(monkeypatch)
    result = CliRunner().invoke(main, ["new-project", "study_a", "--into", str(tmp_path)])
    assert "already lives" in result.output
    assert "left alone" in result.output


def test_overwrite_replaces_it(monkeypatch, tmp_path):
    _piped(monkeypatch)
    CliRunner().invoke(main, ["new-project", "c", "--into", str(tmp_path)])
    (tmp_path / "c" / project_config_name("c")).write_text("project: stale\n", encoding="utf-8")

    CliRunner().invoke(main, ["new-project", "c", "--into", str(tmp_path), "--overwrite"])
    assert _settings(tmp_path / "c")["project"] == "c"


def test_the_config_is_what_the_other_commands_read(monkeypatch, tmp_path):
    """The whole point: `junior ingest` in that folder uses this cohort."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    monkeypatch.delenv("JUNIOR_CONFIG", raising=False)
    _answers(monkeypatch, ["c", str(tmp_path), str(tmp_path / "charts")])
    CliRunner().invoke(main, ["new-project"])

    project = tmp_path / "c"
    assert find_config(start=project).path == (project / project_config_name(project.name)).resolve()
    settings = _settings(project)
    assert settings["source_root"] == str(tmp_path / "charts")
    assert settings["output_root"] == str(project)


def test_the_input_folder_defaults_inside_the_project(monkeypatch, tmp_path):
    _answers(monkeypatch, ["c", str(tmp_path), ""])
    CliRunner().invoke(main, ["new-project"])

    source_root = Path(_settings(tmp_path / "c")["source_root"])
    assert source_root.is_relative_to(tmp_path / "c"), source_root
    assert source_root.is_dir(), "nowhere to put the charts"


def test_no_run_id_is_pinned(monkeypatch, tmp_path):
    """A run id is per run, not per cohort. Pinning one would make every run in this
    cohort write into the same directory."""
    _piped(monkeypatch)
    CliRunner().invoke(main, ["new-project", "c", "--into", str(tmp_path)])
    assert _settings(tmp_path / "c")["run_id"] is None


def test_repo_paths_are_pinned_absolute(monkeypatch, tmp_path):
    """The bundled config names repo locations relatively. This file gets read from
    the cohort's folder, where "./models/..." would resolve to nothing."""
    _piped(monkeypatch)
    CliRunner().invoke(main, ["new-project", "c", "--into", str(tmp_path)])

    settings = _settings(tmp_path / "c")
    assert Path(settings["recipes_root"]).is_absolute()
    assert Path(settings["encoder"]["model_id"]).is_absolute()


def test_next_steps_are_cli_commands_including_cd(monkeypatch, tmp_path):
    """Commands find their config by looking here and upward, so the operator has to
    be told to stand in the folder that was just created."""
    _answers(monkeypatch, ["c", str(tmp_path), ""])
    result = CliRunner().invoke(main, ["new-project"])

    assert f"cd {tmp_path / 'c'}" in result.output
    assert "junior ingest" in result.output
    assert ".py" not in result.output, "still pointing at the Python entry point"


def test_it_says_where_each_kind_of_output_goes(monkeypatch, tmp_path):
    """The PHI/shareable split is the whole point of the layout; naming only the root
    would leave an operator to discover which tree they may send onward."""
    _answers(monkeypatch, ["c", str(tmp_path), ""])
    result = CliRunner().invoke(main, ["new-project"])
    for expected in ("Reads from", "structured/", "extract/", "NO_PHI__shareable/",
                     "the tree you may share"):
        assert expected in result.output, f"layout did not mention {expected!r}"


def test_it_says_to_put_patients_in_an_empty_input_folder(monkeypatch, tmp_path):
    _answers(monkeypatch, ["c", str(tmp_path), ""])
    assert "put one folder per patient" in CliRunner().invoke(main, ["new-project"]).output


def test_it_does_not_say_that_when_patients_are_already_there(monkeypatch, tmp_path):
    charts = tmp_path / "charts"
    (charts / "P1").mkdir(parents=True)
    _answers(monkeypatch, ["c", str(tmp_path), str(charts)])
    result = CliRunner().invoke(main, ["new-project"])
    assert "put one folder per patient" not in result.output


def test_a_piped_invocation_never_blocks(monkeypatch, tmp_path):
    """CI has no terminal. Reading a question nobody will answer would hang forever."""
    _piped(monkeypatch)

    def _refuse(_prompt=""):
        raise AssertionError("prompted despite having no terminal")

    monkeypatch.setattr("builtins.input", _refuse)

    result = CliRunner().invoke(main, ["new-project", "piped", "--into", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "piped" / project_config_name("piped")).is_file()


def test_flags_win_over_asking(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive",
        lambda: True,
    )
    monkeypatch.setattr(
        "builtins.input", lambda _prompt="": (_ for _ in ()).throw(
            AssertionError("asked for something already given")
        )
    )

    result = CliRunner().invoke(main, [
        "new-project", "given", "--into", str(tmp_path),
        "--input", str(tmp_path / "i"), "--variables", "stage",
    ])
    assert result.exit_code == 0, result.output
    assert _settings(tmp_path / "given")["recipes"] == ["stage"]


def test_it_does_not_ask_about_extraction(monkeypatch, tmp_path):
    """Setting a cohort up means building its corpus — ingest, embed, index — and none
    of those care which variables get extracted. That belongs to extraction, later."""
    asked: list[str] = []
    monkeypatch.setattr(
        "apps_and_interfaces.console_questions.is_interactive",
        lambda: True,
    )
    replies = iter(["c", str(tmp_path), ""])

    def _record(prompt=""):
        asked.append(prompt)
        return next(replies)

    monkeypatch.setattr("builtins.input", _record)

    result = CliRunner().invoke(main, ["new-project"])
    assert result.exit_code == 0, result.output
    assert not any("variable" in p.lower() or "extract" in p.lower() for p in asked), asked


def test_it_can_be_reached_from_the_prompt_without_arguments():
    """`new-project` at the prompt used to be a usage error demanding NAME."""
    from apps_and_interfaces.command_line_interface import new_project_cmd

    name_argument = next(p for p in new_project_cmd.params if p.name == "name")
    assert not name_argument.required, "bare `new-project` still errors instead of asking"


def test_it_refuses_to_invent_the_folder_you_named(tmp_path, monkeypatch):
    """Creating the parent too would honour a typo. A stray quote once produced a
    folder literally named `'` with the project nested inside, reported as success."""
    _piped(monkeypatch)
    result = CliRunner().invoke(main, ["new-project", "p", "--into", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "does not exist" in result.output
    assert not (tmp_path / "nope").exists(), "invented the folder anyway"


def test_a_name_cannot_contain_quotes_or_slashes(tmp_path, monkeypatch):
    _piped(monkeypatch)
    for bad in ("bad'name", "sub/proj", 'say"what'):
        result = CliRunner().invoke(main, ["new-project", bad, "--into", str(tmp_path)])
        assert result.exit_code != 0, f"accepted {bad!r}"
        assert "quotes or slashes" in result.output
