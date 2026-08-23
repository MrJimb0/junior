"""`junior workbench` — the CLI door into the app.

The server itself is not started here (a stub records the launch instead). What is
pinned is everything the command decides before launching: which project and run it
resolves, the environment it pins for the app process and its children, and the
refusals — a named run with nothing in it, and a missing Shiny install.
"""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import apps_and_interfaces.command_line_interface as cli
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.project_context import CONFIG_ENVIRONMENT_VARIABLE

RUN = "20260101_010101_aa"


def _write_extract(dr, run_id: str, patient_id: str = "Patient_A") -> None:
    var_dir = extract_output_dir(phi_patient_run_dir(run_id, patient_id, dr)) / "date_of_birth"
    var_dir.mkdir(parents=True, exist_ok=True)
    (var_dir / "result.json").write_text(json.dumps({"payload": {"ok": True}}), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    """A minimal project: a config naming this tmp tree as its output root."""
    config = tmp_path / "junior.yaml"
    config.write_text(
        f"output_root: {tmp_path / 'data'}\n"
        f"source_root: {tmp_path / 'charts'}\n",
        encoding="utf-8",
    )
    (tmp_path / "charts").mkdir()
    return config


@pytest.fixture
def launched(monkeypatch):
    """Record the server launch instead of performing it; report the env as pinned."""
    record = {}

    def fake_run(command, check):
        record["command"] = command
        import os

        record["env"] = {k: v for k, v in os.environ.items()
                         if k.startswith(("JR_", "JUNIOR_"))}

    monkeypatch.setattr("subprocess.run", fake_run)
    # The command chooses subprocess (not exec) inside the interactive prompt; force
    # that branch so the test regains control after the "launch".
    monkeypatch.setattr(cli, "_in_interactive_session", lambda: True)
    return record


def test_workbench_pins_the_project_and_the_newest_run_for_the_app(project, launched):
    dr = project.parent / "data"
    _write_extract(dr, "20260101_010101_aa")
    _write_extract(dr, "20260102_020202_bb")

    result = CliRunner().invoke(cli.main, ["workbench", "--config", str(project)])

    assert result.exit_code == 0, result.output
    # The app process reads these three: the pinned project config (so the Run
    # button's child resolves the same project), the cohort's output tree, and the
    # run to open on. Newest run wins when none was named.
    assert launched["env"][CONFIG_ENVIRONMENT_VARIABLE] == str(project)
    assert launched["env"]["JR_DATA_ROOT"] == str(dr)
    assert launched["env"]["JR_REVIEW_RUN_ID"] == "20260102_020202_bb"
    assert "JR_REVIEW_PATIENT_ID" not in launched["env"]
    # And the launch is `python -m shiny run` on the shipped app file.
    assert launched["command"][1:4] == ["-m", "shiny", "run"]
    assert launched["command"][4].endswith("apps_and_interfaces/shiny_review_app/app.py")


def test_a_named_run_with_no_extracted_values_is_refused_with_the_ones_that_have_them(
    project, launched,
):
    dr = project.parent / "data"
    _write_extract(dr, RUN)

    result = CliRunner().invoke(
        cli.main, ["workbench", "--config", str(project), "--run-id", "20269999_999999_ff"],
    )

    assert result.exit_code != 0
    assert "has no extracted values" in result.output
    assert RUN in result.output, "does not name the runs that do have values"
    assert "command" not in launched, "launched the app on a run it just refused"


def test_a_project_with_no_runs_still_opens_with_a_warning(project, launched):
    result = CliRunner().invoke(cli.main, ["workbench", "--config", str(project)])

    assert result.exit_code == 0, result.output
    assert "no extracted values yet" in result.output
    assert "JR_REVIEW_RUN_ID" not in launched["env"]


def test_workbench_without_shiny_names_the_install_it_needs(project, monkeypatch):
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **k: None if name == "shiny" else real_find_spec(name, *a, **k),
    )

    result = CliRunner().invoke(cli.main, ["workbench", "--config", str(project)])

    assert result.exit_code != 0
    assert "pip install -e '.[app]'" in result.output


def test_workbench_is_advertised_beside_the_other_commands():
    result = CliRunner().invoke(cli.main, ["--help"])

    assert result.exit_code == 0
    assert "workbench" in result.output
    assert cli.APP_COMMANDS["workbench"].split(" — ")[0] in result.output


def test_the_address_is_printed_whether_or_not_a_browser_opens(project, launched, monkeypatch):
    """There were two ways to reach the app and both fail quietly. --launch-browser
    goes through Python's webbrowser module, which returns without opening anything
    when it finds no handler; and the URL otherwise appeared only in uvicorn's
    "Uvicorn running on ..." line, which is INFO and suppressed by the --log-level the
    command passes to keep its own banner on screen.

    So a workbench that started perfectly well printed a banner and nothing else, with
    no address to fall back on. Reported by an operator whose browser stayed shut."""
    monkeypatch.setenv(CONFIG_ENVIRONMENT_VARIABLE, str(project))
    dr = project.parent / "data"
    _write_extract(dr, RUN)

    result = CliRunner().invoke(cli.main, ["workbench"])

    assert result.exit_code == 0, result.output
    port = launched["command"][launched["command"].index("--port") + 1]
    assert f"http://127.0.0.1:{port}" in result.output, "no address to fall back on"
    # And say that falling back is a thing to do, since the browser failing is silent.
    assert "browser does not open" in result.output
