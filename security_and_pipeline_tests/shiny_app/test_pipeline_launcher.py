"""Start-tab launcher: what it offers to run, and how it reads a run's stdout.

The full-spine subprocess itself needs the encoder + LLM weights, so these tests
drive the tail/parse path with a stand-in child process instead — the part that
decides what the operator sees and which run the review picker jumps to.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pipeline_launcher
import pytest


def _finished_run(script: str) -> pipeline_launcher.PipelineRun:
    """A PipelineRun over a throwaway child, tailed to completion synchronously."""
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    run = pipeline_launcher.PipelineRun(patients=["P1"], variables=["v"], process=process)
    run._tail()
    return run


def test_the_start_tab_offers_the_release_set_and_nothing_else():
    """A tick on the Start tab RUNS the recipe it names, so what the tab offers is
    what this repository publishes: the release set and nothing else. Non-clinical
    scaffolding (the smoke recipe) and anything not in the release set stay
    unoffered even when a recipe folder is on disk."""
    offered = pipeline_launcher.available_variables()
    on_disk = pipeline_launcher.recipes_on_disk()

    assert len(pipeline_launcher.RELEASE_VARIABLES) == 8
    assert offered == [v for v in pipeline_launcher.RELEASE_VARIABLES if v in on_disk]
    for dropped in ("treatment_lines", "reranker_smoke", "race_ethnicity",
                    "menopausal_status", "local_treatment_original",
                    "invasive_cancer_confirmation", "pathology_table"):
        assert dropped not in offered, (
            f"{dropped} is dropped by the release plan but the Start tab offers it"
        )


def test_the_recipe_scan_still_sees_everything_on_disk():
    """It is the OFFER that narrowed, not the ability to see the tree. date_of_birth
    sits under basic/ and reranker_smoke under smoke/, so this also pins that the scan
    still finds a recipe at any depth."""
    on_disk = pipeline_launcher.recipes_on_disk()

    assert "date_of_birth" in on_disk
    assert "reranker_smoke" in on_disk
    assert on_disk == sorted(on_disk)


def test_readiness_problems_is_a_list_of_plain_english_blockers():
    problems = pipeline_launcher.readiness_problems()

    assert isinstance(problems, list)
    assert all(isinstance(p, str) for p in problems)


def test_run_id_is_captured_from_the_child_output():
    run = _finished_run(
        "print('Run ID:  20260101_010101_abcd'); print('DONE run_id=20260101_010101_abcd')"
    )

    assert run.run_id == "20260101_010101_abcd"
    assert run.succeeded
    assert not run.is_running
    assert "Finished" in run.status_line()


def test_progress_bar_repaints_are_dropped_from_the_log():
    # tqdm redraws with \r, which universal newlines turns into one line per repaint.
    run = _finished_run(
        r"print('=== Step 2: Embed ==='); print('Loading weights:  50%|#####     | 1/2')"
    )

    assert "=== Step 2: Embed ===" in run.log()
    assert not any("%|" in line for line in run.log())


def test_a_failing_child_is_reported_as_stopped():
    run = _finished_run("import sys; print('boom'); sys.exit(3)")

    assert not run.succeeded
    assert "exit code 3" in run.status_line()


def test_the_run_button_is_a_junior_run_invocation():
    """The app must not grow a pipeline dialect of its own: the Run button's
    subprocess IS the CLI's `run` command — same config resolution, same engine —
    with the operator's picks as ordinary flags."""
    command = pipeline_launcher.command_for(
        "/cohort/folder", ["patient_a", "patient_b"], ["date_of_birth"], "/data/root"
    )
    module_flag = command.index("-m")
    assert command[module_flag + 1 : module_flag + 4] == [
        "apps_and_interfaces.command_line_interface", "run", "/cohort/folder",
    ]
    output_flag = command.index("--output")
    assert command[output_flag + 1] == "/data/root"
    assert command.count("--patient") == 2
    assert command[command.index("--patient") + 1] == "patient_a"
    assert command[command.index("--variable") + 1] == "date_of_birth"


def _finished_stage(stage: str, scripts: list[str]) -> pipeline_launcher.StageRun:
    """A StageRun over throwaway children, walked to completion synchronously."""
    run = pipeline_launcher.StageRun(
        stage=stage, patients=["P1"], environment=dict(os.environ),
        commands=[[sys.executable, "-u", "-c", s] for s in scripts],
    )
    run._walk()
    return run


def test_the_stage_buttons_are_the_cli_stages():
    """The buttons exist to BE the CLI, not to resemble it. A stage the CLI renames or
    adds has to reach the Start tab, and the only thing that makes that happen is this
    test failing on the day it changes."""
    from apps_and_interfaces.command_line_interface import PIPELINE_PHASES, main

    assert pipeline_launcher.STAGES == tuple(PIPELINE_PHASES["Run"])
    for stage in pipeline_launcher.STAGES:
        assert stage in main.commands, (
            f"the Start tab offers a {stage!r} button and the CLI has no such command"
        )


def test_a_stage_button_runs_the_command_a_shell_would():
    command = pipeline_launcher.stage_command_for("embed")

    assert command[1:] == ["-u", "-m", "apps_and_interfaces.command_line_interface", "embed"]
    # No folder, no --output, no --variable. The stage command has none of them to be
    # given, and inventing them here is precisely the app growing a dialect: two ways
    # to run a stage that can disagree about which cohort it read.
    assert not any(argument.startswith("--") for argument in command)
    assert pipeline_launcher.stage_command_for("index", "P1")[-2:] == ["--patient", "P1"]


def test_a_stage_that_is_not_a_stage_is_refused():
    """`junior summarize` is a real command and not a pipeline stage. A button wired to
    one would spawn it happily and report it as a finished stage."""
    with pytest.raises(ValueError):
        pipeline_launcher.stage_command_for("summarize")


def test_ticking_every_patient_runs_the_cohort_once_not_once_each():
    whole = pipeline_launcher.stage_commands_for("ingest", ["A", "B"], ["A", "B"])
    subset = pipeline_launcher.stage_commands_for("ingest", ["B"], ["A", "B"])

    assert len(whole) == 1 and "--patient" not in whole[0]
    assert [c[-1] for c in subset] == ["B"]


def test_only_extract_sends_the_reviewer_to_the_run():
    """Adopting a finished ingest into the review picker would jump the reviewer to a
    run that has no values in it yet."""
    assert pipeline_launcher.StageRun("extract", [], [], {}).produces_values
    assert not pipeline_launcher.StageRun("ingest", [], [], {}).produces_values


def test_a_stage_reports_its_run_and_finishes():
    run = _finished_stage("ingest", ["print('  run     20260101_010101_abcd')"])

    assert run.run_id == "20260101_010101_abcd"
    assert run.succeeded and not run.is_running
    assert "ingest finished" in run.status_line()


def test_a_failing_stage_stops_the_walk_instead_of_running_the_rest():
    """The stages are ordered. Embedding a cohort whose ingest just failed reads the
    half-written corpus the failure left behind."""
    run = _finished_stage("embed", ["import sys; sys.exit(3)", "print('SHOULD NOT RUN')"])

    assert not run.succeeded
    assert "exit code 3" in run.status_line()
    assert not any("SHOULD NOT RUN" in line for line in run.log())


def test_the_stages_after_ingest_continue_the_run_ingest_opened():
    """Reported as "I ticked date of death and it didn't seem to run".

    On the bundled settings — no project — every stage command deliberately starts a
    FRESH run: continuing whatever is newest on disk from an unknown directory could
    append to a cohort nobody asked about. So Ingest, then Extract, extracted from an
    empty run it had just minted, and said it had finished. Naming the run is how the
    CLI already lets a caller who knows better say so, and it is what a SLURM array
    task passes."""
    assert "--run-id" not in pipeline_launcher.stage_command_for("ingest")

    carried = pipeline_launcher.stage_command_for("embed", None, "20260101_010101_aa")
    assert carried[-2:] == ["--run-id", "20260101_010101_aa"]


def test_only_extract_is_handed_the_ticked_variables():
    """`junior embed --variable x` is refused by the CLI rather than ignored, so a
    button that passed one would fail the stage outright."""
    extract = pipeline_launcher.stage_command_for("extract", None, "R", ["date_of_death"])
    embed = pipeline_launcher.stage_command_for("embed", None, "R", ["date_of_death"])

    assert extract[-2:] == ["--variable", "date_of_death"]
    assert "--variable" not in embed


def test_extract_can_be_asked_for_a_variable_the_project_does_not_list():
    """The tick has to outrank the project's `recipes:` list, or ticking anything the
    project did not already name is a control that does nothing."""
    from apps_and_interfaces.command_line_interface import main

    variable_option = next(
        p for p in main.commands["extract"].params if p.name == "variables"
    )
    assert variable_option.multiple, "--variable cannot name more than one variable"
    assert "extract" in main.commands and "variables" not in [
        p.name for p in main.commands["embed"].params if not p.hidden
    ], "--variable is offered on a stage that extracts nothing"


def test_starting_over_asks_the_cli_to_start_over():
    """Clearing the carried run id is NOT how a run is started over.

    The app is always inside a project — `junior workbench` pins one when it launches —
    and inside a project a bare `junior ingest` continues the newest run. So "no run id"
    means continue, which is the opposite of start over. --new-run is the CLI's own
    answer, and the one it prints when a run needs redoing."""
    fresh = pipeline_launcher.stage_command_for("ingest", None, "", None, new_run=True)

    assert fresh[-1] == "--new-run"
    assert "--run-id" not in fresh


def test_a_new_run_is_one_run_not_one_per_patient():
    """--new-run on every invocation would mint a run per patient, leaving each one
    holding a single patient and the cohort split across four of them."""
    commands = pipeline_launcher.stage_commands_for(
        "ingest", ["A", "B", "C"], ["A", "B", "C", "D"], new_run=True,
    )

    assert [c.count("--new-run") for c in commands] == [1, 0, 0]


def test_a_run_cannot_be_both_new_and_named():
    """The CLI refuses the pair outright, so building it would only fail at the child."""
    with pytest.raises(ValueError):
        pipeline_launcher.stage_command_for("ingest", None, "20260101_x", None, new_run=True)
