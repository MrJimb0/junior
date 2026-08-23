"""The CLI finds its config and its run so you do not have to name them.

Typing `--config` and `--run-id` on every command is the repetition a CLI should
absorb, but guessing wrong is worse than asking, so the order of preference is fixed
and an explicit flag always wins. These tests pin that order, and pin that a run
continues rather than silently starting a new one — the failure that would otherwise
scatter one cohort's outputs across several run directories.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jr_pipeline.runtime_infrastructure.project_context import (
    PROJECT_CONFIG_NAME,
    find_config,
    newest_run_id,
    resolve_run_id,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_ambient_environment(monkeypatch):
    for name in ("JUNIOR_CONFIG", "JR_RUN_ID", "JR_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_an_explicit_config_wins_over_everything(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("run_id: r\n", encoding="utf-8")
    (tmp_path / PROJECT_CONFIG_NAME).write_text("run_id: nearby\n", encoding="utf-8")
    monkeypatch.setenv("JUNIOR_CONFIG", str(tmp_path / "from_env.yaml"))

    assert find_config(explicit, start=tmp_path).path == explicit.resolve()


def test_the_environment_wins_over_a_nearby_file(tmp_path, monkeypatch):
    (tmp_path / PROJECT_CONFIG_NAME).write_text("run_id: nearby\n", encoding="utf-8")
    from_env = tmp_path / "from_env.yaml"
    from_env.write_text("run_id: env\n", encoding="utf-8")
    monkeypatch.setenv("JUNIOR_CONFIG", str(from_env))

    assert find_config(start=tmp_path).path == from_env.resolve()


def test_a_project_config_is_found_from_a_subdirectory(tmp_path):
    """A config at the top of a project applies inside it, the way a git repo does."""
    (tmp_path / PROJECT_CONFIG_NAME).write_text("run_id: top\n", encoding="utf-8")
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)

    choice = find_config(start=deep)
    assert choice.path == (tmp_path / PROJECT_CONFIG_NAME).resolve()
    assert str(tmp_path) in choice.reason


def test_falls_back_to_the_bundled_config(tmp_path):
    choice = find_config(start=tmp_path)
    assert choice.path == (REPO / "deployment" / "local" / "laptop.yaml").resolve()
    assert choice.reason == "bundled default"


def test_the_choice_says_where_it_came_from():
    """A command that picks a config for you has to name it, or a wrong cohort is
    silent — the one failure this whole mechanism could introduce."""
    for choice in (find_config(), find_config(Path(__file__))):
        assert choice.reason


# --- which run ------------------------------------------------------------------


def _make_run(data_root: Path, run_id: str) -> None:
    (data_root / "CONTAINS_PHI" / "pipeline_run_receipts" / run_id).mkdir(parents=True)


def test_an_explicit_run_id_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_RUN_ID", "from_env")
    assert resolve_run_id({"run_id": "from_config"}, "explicit", data_root=tmp_path) == "explicit"


def test_the_environment_wins_over_the_config(tmp_path, monkeypatch):
    """The SLURM scripts export JR_RUN_ID; every array task must land in one run."""
    monkeypatch.setenv("JR_RUN_ID", "from_env")
    assert resolve_run_id({"run_id": "from_config"}, None, data_root=tmp_path) == "from_env"


def test_the_newest_run_continues_when_nothing_names_one(tmp_path):
    """`junior ingest` then `junior embed` must land in the same run without being
    told, or the second command starts an empty one and finds nothing to work on."""
    _make_run(tmp_path, "20260101_000000_aa")
    _make_run(tmp_path, "20260202_000000_bb")
    assert resolve_run_id({}, None, data_root=tmp_path) == "20260202_000000_bb"


def test_a_hand_made_directory_cannot_capture_the_next_command(tmp_path):
    """Only canonically-named runs count, so a scratch directory sorting last does
    not quietly become the run everything continues into."""
    _make_run(tmp_path, "20260101_000000_aa")
    _make_run(tmp_path, "zzz_scratch")
    assert newest_run_id(tmp_path) == "20260101_000000_aa"


def test_a_fresh_id_when_there_is_no_run_yet(tmp_path):
    minted = resolve_run_id({}, None, data_root=tmp_path)
    from jr_pipeline.runtime_infrastructure.project_context import RUN_ID_PATTERN

    assert RUN_ID_PATTERN.fullmatch(minted), minted


class TestTheSettingsFileIsNamedAfterItsProject:
    """A dozen files all called junior.yaml are indistinguishable in an editor tab,
    a search result, or a folder listing. The name carries the project instead."""

    def test_the_name_carries_the_project(self):
        from jr_pipeline.runtime_infrastructure.project_context import project_config_name

        assert project_config_name("test6") == "junior_test6.yaml"

    def test_awkward_characters_do_not_reach_the_filename(self):
        from jr_pipeline.runtime_infrastructure.project_context import project_config_name

        assert project_config_name("breast 2026") == "junior_breast_2026.yaml"
        assert project_config_name("") == "junior.yaml"

    def test_a_named_config_is_discovered(self, tmp_path):
        (tmp_path / "junior_study7.yaml").write_text("project: study7\n", encoding="utf-8")
        choice = find_config(start=tmp_path)
        assert choice.path.name == "junior_study7.yaml"

    def test_the_old_plain_name_still_works(self, tmp_path):
        """Projects created before the rename must keep running."""
        (tmp_path / "junior.yaml").write_text("project: old\n", encoding="utf-8")
        assert find_config(start=tmp_path).path.name == "junior.yaml"

    def test_two_projects_in_one_folder_is_refused_not_guessed(self, tmp_path):
        """A project owns its folder, so two here means two projects were mixed.
        Picking one silently would run the wrong cohort."""
        (tmp_path / "junior_a.yaml").write_text("project: a\n", encoding="utf-8")
        (tmp_path / "junior_b.yaml").write_text("project: b\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError) as raised:
            find_config(start=tmp_path)
        assert "More than one" in str(raised.value)
        assert "junior_a.yaml" in str(raised.value)
