"""Mapping a site's column names asks which site, not which file path.

Most exports are already mapped — a worked example ships with the repo — so the first
question is which of the known mappings this export is, and drafting a new one is the
last option rather than the only one.

A drafted map is written into the PROJECT, beside its settings file, never into the
checkout. Writing it to deployment/<site>/ made every throwaway a permanent part of the
codebase: it stayed on this menu long after its author deleted what they thought was the
only copy, showed up as an untracked file in the repo, and would have been published with
it, carrying one site's internal column names into everybody else's copy.

Which maps get offered follows from that. A map ships with Junior, or some cohort on this
machine is actually using it. Anything else that happens to be a yaml in the checkout is
not an answer to "which mapping matches your export".

The step that used to be left to the operator, and so got skipped, is recording the
choice in the cohort's config. Skipping it fails quietly — ingest falls back to
generic column guesses and only mentions it as a preflight advisory.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import (
    _known_column_maps,
    _point_config_at_column_map,
    main,
)
from jr_pipeline.runtime_infrastructure.project_context import PROJECT_CONFIG_NAME

REPO = Path(__file__).resolve().parents[2]


def test_the_shipped_example_mapping_is_found():
    found = _known_column_maps()
    assert "example_site" in found, found
    assert found["example_site"].name.endswith(".yaml")


def test_a_config_that_only_mentions_the_key_in_a_comment_is_not_a_mapping():
    """The bundled laptop.yaml documents `chunk_metadata_columns:` in a comment. A
    substring match offered it as a site mapping, which it is not."""
    found = _known_column_maps()
    assert "local" not in found, f"laptop.yaml was mistaken for a column map: {found}"


def test_every_found_mapping_actually_parses_as_one():
    for site, path in _known_column_maps().items():
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(parsed.get("chunk_metadata_columns"), dict), site


def test_a_drafted_map_lands_in_the_project_not_the_checkout(tmp_path, monkeypatch):
    """The whole reason a throwaway stuck around.

    Written into deployment/<site>/, a map made while trying the tool out becomes part
    of the codebase: offered on this menu for good, untracked in the repo, and published
    with it."""
    from apps_and_interfaces.command_line_interface import _where_a_drafted_map_goes

    project = tmp_path / "breast2026"
    project.mkdir()
    (project / "junior_breast2026.yaml").write_text(
        yaml.safe_dump({"project": "breast2026", "output_root": str(project)}),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    destination = _where_a_drafted_map_goes("shc")

    assert destination == project / "shc_column_map.yaml"
    assert REPO not in destination.parents, "a draft was written into the checkout"


def test_with_no_project_a_draft_lands_where_you_are_standing(tmp_path, monkeypatch):
    from apps_and_interfaces.command_line_interface import _where_a_drafted_map_goes

    monkeypatch.chdir(tmp_path)

    destination = _where_a_drafted_map_goes("shc")

    assert destination == tmp_path / "shc_column_map.yaml"
    assert REPO not in destination.parents


def test_a_map_a_cohort_uses_is_offered_even_though_it_is_not_in_the_checkout(
    tmp_path, monkeypatch
):
    """The second source: a map some project here actually reads its charts with."""
    from apps_and_interfaces import project_registry

    monkeypatch.setenv(
        project_registry.REGISTRY_ENVIRONMENT_VARIABLE, str(tmp_path / "projects.yaml")
    )
    project = tmp_path / "breast2026"
    project.mkdir()
    in_use = project / "shc_column_map.yaml"
    in_use.write_text(
        yaml.safe_dump({"chunk_metadata_columns": {"clinical_note": {"text_columns": ["t"]}}}),
        encoding="utf-8",
    )
    config = project / "junior_breast2026.yaml"
    config.write_text(
        yaml.safe_dump({"project": "breast2026", "chart_columns_file": str(in_use)}),
        encoding="utf-8",
    )
    project_registry.remember(config)

    found = _known_column_maps()

    assert found.get("breast2026") == in_use
    assert "example_site" in found, "the shipped maps must still be offered"


def test_a_stray_yaml_a_cohort_stopped_using_is_not_offered(tmp_path, monkeypatch):
    """A project that no longer points at a map stops proposing it."""
    from apps_and_interfaces import project_registry

    monkeypatch.setenv(
        project_registry.REGISTRY_ENVIRONMENT_VARIABLE, str(tmp_path / "projects.yaml")
    )
    project = tmp_path / "breast2026"
    project.mkdir()
    (project / "abandoned_column_map.yaml").write_text(
        yaml.safe_dump({"chunk_metadata_columns": {"clinical_note": {"text_columns": ["t"]}}}),
        encoding="utf-8",
    )
    config = project / "junior_breast2026.yaml"
    config.write_text(yaml.safe_dump({"project": "breast2026"}), encoding="utf-8")
    project_registry.remember(config)

    assert "breast2026" not in _known_column_maps()


def _cohort(tmp_path: Path) -> Path:
    """A project folder with a junior.yaml, as new-project would leave it."""
    project = tmp_path / "cohort"
    project.mkdir()
    (project / PROJECT_CONFIG_NAME).write_text(
        "# a comment that must survive\nproject: c\nrun_id: null\n", encoding="utf-8"
    )
    return project


def test_choosing_a_mapping_records_it_in_the_cohort_config(monkeypatch, tmp_path):
    """Telling someone to add a line to a YAML file is the step that gets skipped."""
    project = _cohort(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.delenv("JUNIOR_CONFIG", raising=False)

    chosen = _known_column_maps()["example_site"]
    _point_config_at_column_map(chosen)

    settings = yaml.safe_load((project / PROJECT_CONFIG_NAME).read_text(encoding="utf-8"))
    assert settings["chart_columns_file"] == str(chosen)
    assert settings["project"] == "c", "rewriting the config dropped its other settings"


def test_rewriting_the_config_keeps_its_comments(monkeypatch, tmp_path):
    project = _cohort(tmp_path)
    monkeypatch.chdir(project)
    monkeypatch.delenv("JUNIOR_CONFIG", raising=False)

    _point_config_at_column_map(_known_column_maps()["example_site"])
    assert "a comment that must survive" in (project / PROJECT_CONFIG_NAME).read_text()


def test_with_no_cohort_config_it_prints_the_line_instead(monkeypatch, tmp_path, capsys):
    """Outside a cohort there is nothing to record it in, so say what to add rather
    than editing the bundled config that every other cohort shares."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JUNIOR_CONFIG", raising=False)

    _point_config_at_column_map(Path("/some/map.yaml"))
    out = capsys.readouterr().out
    assert "chart_columns_file: /some/map.yaml" in out
    assert "Add this to" in out


def test_piped_output_still_prints_the_draft(tmp_path):
    """`junior columns <folder> > map.yaml` has to keep working for scripts."""
    charts = tmp_path / "P1"
    charts.mkdir()
    (charts / "clinical_note.csv").write_text("patient_id,text,date\nP1,hi,2020-01-01\n",
                                              encoding="utf-8")

    result = CliRunner().invoke(main, ["columns", str(charts)])
    assert result.exit_code == 0, result.output
    assert "chunk_metadata_columns:" in result.output
    parsed = yaml.safe_load(
        "\n".join(line for line in result.output.splitlines() if not line.startswith("#"))
    )
    assert isinstance(parsed.get("chunk_metadata_columns"), dict)


def test_it_names_the_columns_being_embedded_not_the_tables(capsys):
    """The unit is a column, not a table: `text_columns:` names the column inside a
    table that holds free text, and one table can offer several. Saying "3 of 19
    tables will be searched" mis-stated what the mapping decides."""
    from apps_and_interfaces.command_line_interface import (
        _describe_what_gets_embedded,
    )

    _describe_what_gets_embedded(_known_column_maps()["example_site"])
    out = capsys.readouterr().out
    assert "column(s) will be embedded" in out
    # table.column, so it is unambiguous which column of which file.
    assert "clinical_note.text" in out
    # And what contributing no column actually costs — nothing quotable as evidence.
    assert "have no text column" in out
    assert "evidence" in out
    assert "labs" in out


def test_a_table_with_two_text_columns_counts_as_two(tmp_path, capsys):
    """Counting tables would report one; the thing being embedded is the column."""
    from apps_and_interfaces.command_line_interface import (
        _describe_what_gets_embedded,
    )

    map_file = tmp_path / "map.yaml"
    map_file.write_text(
        yaml.safe_dump({"chunk_metadata_columns": {
            "note": {"text_columns": ["body", "addendum"]},
            "labs": {"document_date": "date"},
        }}),
        encoding="utf-8",
    )

    _describe_what_gets_embedded(map_file)
    out = capsys.readouterr().out
    assert "2 column(s)" in out
    assert "note.body" in out and "note.addendum" in out


def test_it_says_what_to_do_next(capsys):
    """A command that finishes into a bare prompt leaves the operator to remember
    where they were in a sequence they have run once."""
    from apps_and_interfaces.command_line_interface import (
        _what_next_after_columns,
    )

    _what_next_after_columns()
    out = capsys.readouterr().out
    assert "Next" in out
    assert "junior ingest" in out


def test_it_does_not_hand_a_file_to_the_system_default_application(monkeypatch, tmp_path):
    """Opening a .yaml with the system handler launched a spreadsheet suite, which
    crashed. A CLI should not start a GUI the operator did not ask for."""
    from apps_and_interfaces.command_line_interface import _offer_to_open

    monkeypatch.delenv("EDITOR", raising=False)
    monkeypatch.delenv("VISUAL", raising=False)
    opened: list[str] = []
    monkeypatch.setattr("click.edit", lambda **kwargs: opened.append("edit"))
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface._is_interactive",
        lambda: True,
    )

    target = tmp_path / "map.yaml"
    target.write_text("chunk_metadata_columns: {}\n", encoding="utf-8")
    _offer_to_open(target)
    assert not opened, "launched an editor with no $EDITOR set"


def test_it_uses_the_editor_you_chose_when_you_chose_one(monkeypatch, tmp_path):
    from apps_and_interfaces.command_line_interface import _offer_to_open

    monkeypatch.setenv("EDITOR", "vim")
    opened: list[str] = []
    monkeypatch.setattr("click.edit", lambda **kwargs: opened.append("edit"))
    monkeypatch.setattr("click.confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface._is_interactive",
        lambda: True,
    )

    _offer_to_open(tmp_path / "map.yaml")
    assert opened == ["edit"]


class TestEmbedConfirmsBeforeCommitting:
    """Embedding pins the model for the whole run and is the first slow step, so it
    is the last moment the choice is cheap to change."""

    def _shown(self, monkeypatch, capsys, run_root, cfg) -> str:
        import click as click_module

        from apps_and_interfaces import command_line_interface as cli

        monkeypatch.setattr(cli, "is_interactive", lambda: True)
        monkeypatch.setattr(click_module, "confirm", lambda *a, **k: True)
        cli._confirm_embedding_model(cfg, run_root, 9)
        return capsys.readouterr().out

    def test_it_says_what_model_which_columns_and_where(self, monkeypatch, capsys, tmp_path):
        cfg = {
            "encoder": {"model_id": "/models/BioClinical-ModernBERT"},
            "chart_columns_file": str(_known_column_maps()["example_site"]),
        }
        out = self._shown(monkeypatch, capsys, tmp_path / "run", cfg)
        assert "BioClinical-ModernBERT" in out, "did not say which model"
        assert "clinical_note.text" in out, "did not say which columns"
        assert "patients/<patient>/" in out, "did not say where output goes"

    def test_it_warns_when_the_model_folder_is_missing(self, monkeypatch, capsys, tmp_path):
        cfg = {"encoder": {"model_id": str(tmp_path / "no_such_model")}}
        out = self._shown(monkeypatch, capsys, tmp_path / "run", cfg)
        assert "does not exist" in out

    def test_it_does_not_ask_again_once_the_run_has_embeddings(self, monkeypatch, tmp_path):
        """Re-running embed for a late-arriving patient must not re-ask: the choice
        was made and cannot differ within a run."""
        import click as click_module

        from apps_and_interfaces import command_line_interface as cli

        run_root = tmp_path / "run"
        fragments = run_root / "state_fragments"
        fragments.mkdir(parents=True)
        (fragments / "state.pid_1.jsonl").write_text(
            '{"payload": {"entity": {"step": "embed", "patient_id": "P1"},'
            ' "to_state": "completed", "ts": "2026-01-01T00:00:00+00:00"}}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(cli, "is_interactive", lambda: True)
        monkeypatch.setattr(click_module, "confirm", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("asked again after the run already chose")))

        assert cli._confirm_embedding_model(
            {"encoder": {"model_id": "/m"}}, run_root, 1) is True

    def test_a_piped_run_never_blocks_on_the_question(self, monkeypatch, tmp_path):
        import click as click_module

        from apps_and_interfaces import command_line_interface as cli

        monkeypatch.setattr(cli, "is_interactive", lambda: False)
        monkeypatch.setattr(click_module, "confirm", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("asked with no terminal to answer")))

        assert cli._confirm_embedding_model(
            {"encoder": {"model_id": "/m"}}, tmp_path / "run", 1) is True
