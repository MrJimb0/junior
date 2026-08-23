"""A run that closes out has to say what to do next.

Junior's design premise is iteration: change a recipe, run it again, see whether it got
better, and learn from the difference. That loop ended on a tick and a shell prompt.
Which variable to look at, where its recipe lives, and that a change requires a fresh
run were all things you had to know already or discover by hitting an error.
"""
from __future__ import annotations

import json

from apps_and_interfaces.command_line_interface import (
    _RUN_INVARIANT_CFG_KEYS,
    _what_to_do_next,
)


def _result(run_root, patient, variable, ok):
    d = run_root / "patients" / patient / "extract" / variable
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps({"payload": {"ok": ok, "variable": variable}}))


def _sealed_config(run_root, recipes_root):
    code = run_root / "code"
    code.mkdir(parents=True, exist_ok=True)
    (code / "config_resolved.yaml").write_text(f"recipes_root: {recipes_root}\n")


def test_it_names_the_failing_variable_and_where_its_recipe_lives(tmp_path, capsys):
    run_root = tmp_path / "run"
    _result(run_root, "P1", "stage", ok=False)
    _result(run_root, "P2", "stage", ok=True)
    _result(run_root, "P1", "date_of_birth", ok=True)
    recipes = tmp_path / "var_extraction_recipes"
    (recipes / "oncology" / "general_oncology" / "stage" / "v1").mkdir(parents=True)
    _sealed_config(run_root, recipes)

    _what_to_do_next(run_root)

    out = capsys.readouterr().out
    # A pointer, not a second warning: the stage already said what failed and how much.
    assert "stage recipe:" in out
    assert "failed for 1 of 2" not in out, "the failure is announced twice"
    # The nesting by collection is not something a reader has any reason to know, so
    # the path is the point of the line, not decoration.
    assert "stage/v1" in out
    assert "date_of_birth" not in out, "a variable that worked is not a thing to go fix"


def test_it_explains_why_a_change_needs_a_new_run(tmp_path, capsys):
    """Hitting the run-lock as an error is how this used to be learned."""
    run_root = tmp_path / "run"
    _result(run_root, "P1", "stage", ok=True)
    _sealed_config(run_root, tmp_path / "var_extraction_recipes")

    _what_to_do_next(run_root)

    out = capsys.readouterr().out
    assert "junior extract --new-run" in out
    assert "embeddings will be re-used" in out, "the reason, not just the instruction"


def test_a_clean_run_still_says_where_to_go(tmp_path, capsys):
    run_root = tmp_path / "run"
    _result(run_root, "P1", "stage", ok=True)
    _sealed_config(run_root, tmp_path / "var_extraction_recipes")

    _what_to_do_next(run_root)

    out = capsys.readouterr().out
    assert "--new-run" in out
    assert "⚠" not in out, "a clean run has nothing to warn about"
    assert "recipe:" not in out, "a variable that worked is not a thing to go fix"


def test_it_says_nothing_when_nothing_was_extracted(tmp_path, capsys):
    """Ending embed should not lecture about recipes."""
    run_root = tmp_path / "run"
    run_root.mkdir()

    _what_to_do_next(run_root)

    assert capsys.readouterr().out == ""


def test_a_missing_sealed_config_does_not_crash_the_close_out(tmp_path, capsys):
    """This runs at the very end of `finish`. It must never be the thing that fails."""
    run_root = tmp_path / "run"
    _result(run_root, "P1", "stage", ok=False)

    _what_to_do_next(run_root)

    out = capsys.readouterr().out
    # Without the sealed config there is no recipes_root, so the recipe pointer is
    # dropped and the guidance still prints. It must not take the close-out down.
    assert "junior extract --new-run" in out


def test_it_covers_each_way_a_run_gets_changed(tmp_path, capsys):
    """The three things someone does after a run: touch a recipe, change which
    variables run, or change what it runs over (more patients, another cohort, another
    machine). Each has a different answer and only one of them is "carry on"."""
    run_root = tmp_path / "run"
    _result(run_root, "P1", "stage", ok=True)
    _sealed_config(run_root, tmp_path / "var_extraction_recipes")

    _what_to_do_next(run_root)

    out = capsys.readouterr().out
    for question in ("Editing a recipe?",
                     "Adding or removing a recipe/variable?",
                     "Changing cohort or corpus settings, or rerunning everything from scratch?"):
        assert question in out, f"no guidance for: {question}"
    # Starting over must never send someone to invent a run id.
    assert "--run-id" not in out, "the routine loop should not mention --run-id"
    assert out.count("--new-run") >= 4
    # The distinction the whole command model rests on: a recipe change reuses the
    # corpus, a cohort or corpus change rebuilds it.
    assert "junior extract --new-run" in out
    # Rebuilding is driven a stage at a time, not by one command that hides where it stopped.
    assert "junior ingest --new-run" in out
    assert "junior run --new-run" not in out
    # Removing a variable requires running nothing; the line must not read as a command.
    assert "removed   keep this run" in out


def test_the_execution_settings_advice_matches_what_is_actually_locked():
    """The note says changing EXECUTION SETTINGS needs a fresh run, not changing
    machines — moving machines needs nothing on its own. That wording is only honest
    while there are settings a run is actually fixed to."""
    assert "max_chunks_per_prompt" in _RUN_INVARIANT_CFG_KEYS
    assert _RUN_INVARIANT_CFG_KEYS, "nothing is locked, so the advice is telling people to start over for no reason"


def test_the_removed_a_variable_advice_matches_what_is_actually_locked():
    """"removed a variable -> nothing" holds because the recipe LIST is not locked to
    the run. Lock it and removing one starts being refused, and the note is wrong."""
    assert "recipes" not in _RUN_INVARIANT_CFG_KEYS


def test_the_more_patients_advice_matches_what_is_actually_locked():
    """"more patients -> same run" holds because where charts are read from is not
    locked. It is also why "a different cohort" has to be advice rather than a refusal:
    nothing stops you, so the note is the only thing that says not to."""
    assert "source_root" not in _RUN_INVARIANT_CFG_KEYS
