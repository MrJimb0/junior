"""Shared fixtures for the security_and_pipeline_tests suite.

The pipeline reads configuration from ``JR_*`` environment variables, plus
``JUNIOR_CONFIG`` (the pinned project config file the workbench exports for its
child processes). A test that sets one must not leak that state into sibling
tests (the env-leak flake class). The autouse fixture below snapshots every such
variable before each test and restores the exact prior state afterward — values
changed, variables added, and variables deleted by the test are all undone.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _is_junior_setting(name: str) -> bool:
    return name.startswith("JR_") or name.startswith("JUNIOR_")


@pytest.fixture(autouse=True)
def _projects_list_stays_out_of_the_operators_home(tmp_path_factory, monkeypatch):
    """Send the remembered-projects list somewhere disposable, for every test.

    `new-project` writes what it created to that list, so without this every test that
    creates a cohort appends a pytest temp folder to the list belonging to whoever ran
    the suite — a test run editing a real person's files outside the checkout."""
    from apps_and_interfaces.project_registry import REGISTRY_ENVIRONMENT_VARIABLE

    monkeypatch.setenv(
        REGISTRY_ENVIRONMENT_VARIABLE,
        str(tmp_path_factory.mktemp("projects_list") / "projects.yaml"),
    )


@pytest.fixture(autouse=True)
def _restore_jr_env():
    """Snapshot/restore all Junior environment variables around each test."""
    saved = {k: v for k, v in os.environ.items() if _is_junior_setting(k)}
    try:
        yield
    finally:
        for key in [k for k in os.environ if _is_junior_setting(k)]:
            if key not in saved:
                del os.environ[key]
        for key, value in saved.items():
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _runs_land_somewhere_disposable(tmp_path_factory, monkeypatch):
    """Send the data root somewhere disposable, for every test.

    Same reason as the projects list above, and the same failure: a suite that writes
    into the checkout's own ``data/``. Months of it had accumulated there — 114M, and
    seventeen run folders named ``e2e``, ``tm2``, ``prog3``, ``RUNX`` and the like that
    nobody made on purpose. They were tests, writing to the default root because
    nothing had pinned one.

    Left alone it is not only clutter: those are PHI-classified directories appearing
    in a repository on every test run, and a stale one under a run id a test reuses is
    state one run can hand to the next."""
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path_factory.mktemp("data_root")))


@pytest.fixture(scope="session", autouse=True)
def _nothing_writes_into_the_checkouts_data_folder():
    """Fail the session if the suite left anything in the repository's own data/.

    The fixture above pins the root, which is enough for a test that lets it stand. A
    test that deliberately unsets JR_DATA_ROOT — several do, to exercise how the
    default is resolved — falls back to <repo>/data, and if it then writes, it writes
    here. That is the case a pinned root cannot catch, so it is caught after the fact
    instead: a leak becomes a red session naming the paths, rather than a folder that
    quietly grows for months."""
    repo_data = Path(__file__).resolve().parents[1] / "data"

    def whats_there() -> set[str]:
        if not repo_data.is_dir():
            return set()
        return {str(p.relative_to(repo_data)) for p in repo_data.rglob("*") if p.is_file()}

    before = whats_there()
    yield
    leaked = sorted(whats_there() - before)
    assert not leaked, (
        f"the suite wrote {len(leaked)} file(s) into the checkout's own data/ — a test "
        "that unsets JR_DATA_ROOT falls back to it, so pin one or use tmp_path:\n  "
        + "\n  ".join(leaked[:20])
    )
