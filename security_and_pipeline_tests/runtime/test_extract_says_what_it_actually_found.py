"""`junior extract` must not report a value it did not find, or read a list nobody chose.

The missing-model guard fixed "we never looked" at the point where a variable is
extracted. The same confusion came back one layer up, in how the result is reported.
A recipe names its answer after itself; the terminal report, when that field was null,
fell back to the first field whose name had no underscore in it — and every prompting
recipe requires a free-text ``rationale``, which is always filled in. So a chart with
no date of birth in it printed ``date_of_birth: found`` in green, and with values shown
printed the model's sentence about the chart as though it were the date.

The other half is the choice itself. A new project inherits its variable list from the
bundled settings, so the first extract reads a list nobody has seen, and a name with no
recipe on disk is dropped from the plan and never mentioned — the run reports success
with the column simply absent, which reads exactly like a cohort in which nothing was
found.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import click
import pytest
import yaml

from apps_and_interfaces.command_line_interface import (
    _planned_variables,
    _report_extracted_variables,
    _resolve_recipes_root,
    _resolve_repo_root,
)


def _reported(capsys, variable: str, data: dict, ok: bool = True) -> str:
    _report_extracted_variables({"variables": {variable: {"ok": ok, "data": data}}})
    return capsys.readouterr().out


# --- reporting what came back ------------------------------------------------------


def test_a_null_answer_is_not_reported_as_found(capsys):
    """The one that mattered: a chart with no date of birth in it said `found`."""
    reported = _reported(capsys, "date_of_birth", {
        "date_of_birth": None,
        "dob_certainty": 0,
        "rationale": "no date of birth appears in the retrieved passages",
    })
    assert ": found" not in reported, "a null answer was reported as found"
    assert "nothing found in this chart" in reported


def test_the_models_rationale_is_never_mistaken_for_the_answer(capsys, monkeypatch):
    """With values on screen the fallback printed the rationale sentence in the place
    the value goes — a larger amount of chart text than the value it stood in for."""
    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    reported = _reported(capsys, "date_of_birth", {
        "date_of_birth": None,
        "rationale": "the consult note describes a patient in her sixties",
    })
    assert "sixties" not in reported, "the rationale was printed as the value"


def test_an_empty_list_is_not_a_value(capsys):
    """Several recipes spell absence as an empty list, so treating only null as absent
    reported them as found."""
    assert "nothing found" in _reported(capsys, "treatment_lines", {
        "treatment_lines": [], "rationale": "no systemic therapy documented",
    })


def test_zero_is_still_an_answer(capsys):
    """Absence is checked by type rather than by falsiness, or a legitimate 0 or False
    would be reported as nothing found."""
    assert ": found" in _reported(capsys, "prior_lines", {"prior_lines": 0})


def test_a_real_answer_is_still_reported_as_found(capsys):
    """The guard must not have cost the useful signal."""
    assert ": found" in _reported(capsys, "date_of_birth", {
        "date_of_birth": "1973-01-02", "rationale": "from the demographics table",
    })


def test_a_recipe_shaped_differently_is_still_answered(capsys):
    """A recipe whose result is a table of rows has no field named after itself. It
    used to decline to say anything at all — "read: open it to see what it holds" —
    which was fifteen of the eighteen recipes in the book.

    Declining was the safe answer to the wrong question. The risk being avoided was
    guessing the answer field and picking up `rationale`, which every prompting recipe
    fills in; the provenance rule reads any shape and treats a rationale as structural,
    so it answers without that hole."""
    empty = _reported(capsys, "pathology_table", {"rows": [], "rationale": "none"})
    assert "nothing found" in empty, "an empty table is a chart with nothing in it"
    assert "open it to see what it holds" not in empty

    filled = _reported(capsys, "pathology_table",
                       {"rows": [{"specimen": "left breast"}], "rationale": "one report"})
    assert ": found" in filled, "the records ARE the finding for a row-mapped recipe"


def test_a_rationale_on_its_own_is_not_a_finding(capsys):
    """The exact trap the old guess fell into: every prompting recipe requires a
    rationale and always fills one in, so a recipe whose only content is its own
    reasoning reported a value for a chart that had none."""
    reported = _reported(capsys, "menopausal_status",
                         {"status_at_diagnosis": None, "rationale": "the notes do not say"})
    assert "nothing found" in reported


def test_a_failed_variable_is_still_reported_as_failed(capsys):
    assert "failed" in _reported(capsys, "date_of_birth", {}, ok=False)


# --- which variables get read ------------------------------------------------------


def _config(recipes) -> dict:
    """A project as the CLI hands one to these helpers.

    The allow-list is here because leaving it out is what let a defect through: without
    it the confirmation returns before it ever looks at a model, so the branch that
    reads the list — the one an operator with a real project always takes — was never
    run by a test that claimed to cover the confirmation."""
    repo_root = _resolve_repo_root()
    return {
        "recipes": recipes,
        "recipes_root": str(_resolve_recipes_root({}, repo_root)),
        "allowlist_path": str(
            repo_root / "deployment" / "local" / "llm_allowlist_local3b.yaml"
        ),
    }


def test_a_variable_with_no_recipe_refuses_before_anything_is_written():
    """The planner returns a name it could not find in `missing` and carries on, which
    is right for a dependency and wrong for a name the operator asked for: the run
    reported success with the column absent, and a reader of that dataset cannot tell
    that from a cohort in which nothing was found."""
    with pytest.raises(click.ClickException) as raised:
        _planned_variables(_config(["date_of_brith"]))
    message = str(raised.value)
    assert "date_of_brith" in message
    assert "junior recipes" in message, "does not say how to find the real names"


def test_a_project_that_names_no_variables_refuses():
    with pytest.raises(click.ClickException, match="does not say which variables"):
        _planned_variables(_config([]))


def test_a_variable_that_exists_still_plans():
    """The counterpart — the guard must not refuse a real cohort."""
    planned = _planned_variables(_config(["date_of_birth"]))
    assert [recipe.name for recipe in planned.recipes] == ["date_of_birth"]


def test_the_confirmation_names_the_variables_and_the_patients(capsys, monkeypatch):
    """Embedding confirms its model because that choice is locked for the run.
    Extraction's list is never locked, which is why nothing ever showed it — and a new
    project inherits it, so the first run reads a list nobody chose."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    assert cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    reported = capsys.readouterr().out
    assert "date_of_birth" in reported
    assert "9 patient(s)" in reported


def test_the_confirmation_names_the_model_that_will_read_the_chart(capsys, monkeypatch):
    """Naming only the file that lists the endpoints leaves the choice that most
    decides the values unstated — and the two shipped lists differ in a way that
    matters, one loading a model from this machine and one naming a public hub id it
    will download."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    assert "local_qwen" in capsys.readouterr().out


def test_a_model_that_downloads_its_weights_says_so(capsys, monkeypatch):
    """The sentence that matters most on this screen. The other shipped allow-list names
    a model id on a public hub and opts into fetching it, so confirming a run under it
    means a machine holding patient data reaches the network — the one thing an operator
    would want to stop before saying yes."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    cfg["allowlist_path"] = str(
        _resolve_repo_root() / "deployment" / "local" / "llm_allowlist.yaml"
    )
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    reported = capsys.readouterr().out
    assert "downloads its weights" in reported
    assert "Qwen/Qwen2.5-0.5B-Instruct" in reported, "does not name the model it would fetch"


def test_a_model_folder_that_is_not_there_says_so(capsys, monkeypatch):
    """A local endpoint whose weights are missing fails at the first patient. Saying so
    here costs nothing; finding out after a cohort has been ingested and embedded does."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    allowlist = Path(_resolve_repo_root()) / "deployment" / "local" / "llm_allowlist_local3b.yaml"
    moved = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
    moved["allowed_endpoints"][0]["url"] = "./models/extraction/not_there"
    moved["allowed_endpoints"][0]["default_model"] = "./models/extraction/not_there"
    written = Path(tempfile.mkdtemp()) / "moved.yaml"
    written.write_text(yaml.safe_dump(moved), encoding="utf-8")

    cfg = _config(["date_of_birth"])
    cfg["allowlist_path"] = str(written)
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    assert "is not there, so these will fail" in capsys.readouterr().out


def test_an_endpoint_the_allow_list_does_not_have_is_named(capsys, monkeypatch):
    """`model_override` picks one endpoint by name. Asking for one that is not on the
    list is a typo worth catching before the run, not a silent fallback to another."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    cfg["model_override"] = "a_model_nobody_configured"
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    reported = capsys.readouterr().out
    assert "a_model_nobody_configured" in reported
    assert "lists no endpoint named" in reported


def test_a_project_with_no_allow_list_says_the_run_will_fail(capsys, monkeypatch):
    """A variable that needs a model and a project that names none: extraction refuses
    on its own, and this is where that becomes cheap to fix."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    cfg.pop("allowlist_path")
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    assert "none set — the variables above will fail" in capsys.readouterr().out


def test_an_unreadable_allow_list_is_reported_as_a_sentence(capsys, monkeypatch):
    """A path that is not there, or a file that will not parse, comes back as the
    allow-list loader's own sentence rather than a Python error or silence."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)

    cfg = _config(["date_of_birth"])
    cfg["allowlist_path"] = str(_resolve_repo_root() / "deployment" / "local" / "not_a_file.yaml")
    cli._confirm_variables(cfg, _planned_variables(cfg), 9)
    reported = capsys.readouterr().out
    assert "not_a_file.yaml" in reported
    assert "Traceback" not in reported


def test_a_model_that_cannot_be_described_does_not_stop_the_run(capsys, monkeypatch):
    """This screen describes; extraction is what decides, and refuses on its own when a
    variable needs a model and none can be built. A sentence that cannot be drawn must
    not be what takes a cohort down."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: True)
    monkeypatch.setattr(click, "confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "jr_pipeline.pipeline_steps.step_7_extract_variables.providers."
        "llm_endpoint_allowlist.load_allowlist",
        lambda _path: (_ for _ in ()).throw(AttributeError("something moved")),
    )

    cfg = _config(["date_of_birth"])
    assert cli._confirm_variables(cfg, _planned_variables(cfg), 9)


def test_nothing_is_asked_when_no_one_is_there_to_answer(monkeypatch):
    """A cluster task and a piped command must never block on a question."""
    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "is_interactive", lambda: False)
    cfg = _config(["date_of_birth"])
    assert cli._confirm_variables(cfg, _planned_variables(cfg), 9)
