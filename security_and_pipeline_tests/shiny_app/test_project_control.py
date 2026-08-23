"""The app's project panel drives the CLI's own project machinery.

Switching pins the same two variables `junior workbench` pins at launch, and the New
project button IS `junior new-project` in a subprocess — every question answered by
the form, so a scripted creation never blocks on a prompt and the app never grows a
project-creation dialect of its own.
"""
from __future__ import annotations

import os
from pathlib import Path

import project_control

from jr_pipeline.runtime_infrastructure.project_context import CONFIG_ENVIRONMENT_VARIABLE


def _a_project_config(tmp_path: Path, name: str = "study") -> Path:
    folder = tmp_path / name
    folder.mkdir()
    config = folder / "junior.yaml"
    config.write_text(
        f"project: {name}\n"
        f"source_root: {tmp_path / 'charts'}\n"
        "output_root: data\n",
        encoding="utf-8",
    )
    return config


def test_project_info_anchors_a_relative_output_root_beside_the_settings(tmp_path):
    config = _a_project_config(tmp_path)

    info = project_control.project_info(config)

    assert info.name == "study"
    assert info.output_root == (tmp_path / "study" / "data").resolve()


def test_switch_pins_what_the_workbench_command_pins(tmp_path, monkeypatch):
    config = _a_project_config(tmp_path)

    info = project_control.switch_to(config)

    assert os.environ[CONFIG_ENVIRONMENT_VARIABLE] == str(config.resolve())
    assert os.environ["JR_DATA_ROOT"] == str(info.output_root)
    # The registry remembers the switch — most-recent-first is what makes the
    # project menu's ordering true.
    assert config.resolve() in project_control.known_projects()


def test_the_create_button_is_a_junior_new_project_invocation():
    command = project_control.command_for_new_project(
        "my_study", "/charts/here", "/cohorts/parent",
    )

    module_flag = command.index("-m")
    assert command[module_flag + 1: module_flag + 3] == [
        "apps_and_interfaces.command_line_interface", "new-project",
    ]
    assert command[module_flag + 3] == "my_study"
    assert command[command.index("--input") + 1] == "/charts/here"
    assert command[command.index("--into") + 1] == "/cohorts/parent"


def test_create_project_runs_the_real_cli_and_lands_a_switchable_config(tmp_path):
    charts = tmp_path / "charts"
    (charts / "P1").mkdir(parents=True)
    (charts / "P1" / "clinical_note.csv").write_text("text\nhello\n", encoding="utf-8")

    output, config_path = project_control.create_project(
        "fresh_study", str(charts), str(tmp_path),
    )

    assert config_path is not None, output
    assert config_path.parent == (tmp_path / "fresh_study").resolve()
    assert config_path.name.startswith("junior") and config_path.suffix == ".yaml"
    info = project_control.switch_to(config_path)
    assert info.source_root == str(charts)


def test_a_failed_creation_reports_the_clis_own_words(tmp_path):
    """The reliable refusal: a project that already exists, without --overwrite."""
    charts = tmp_path / "charts"
    charts.mkdir()
    first, config_path = project_control.create_project("study", str(charts), str(tmp_path))
    assert config_path is not None, first

    output, second_path = project_control.create_project("study", str(charts), str(tmp_path))

    assert second_path is None
    assert output, "a failure with no words to act on"
