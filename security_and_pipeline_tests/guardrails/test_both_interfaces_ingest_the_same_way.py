"""The workbench's ingest button and `junior ingest` must read a chart the same way.

The Run button spawns `junior run` as a subprocess, so a full run cannot diverge. The
Start tab's ingest button is the one place the app calls a pipeline step directly, with
a config it builds itself — and a config built by hand is a config that drifts.

It has drifted once already: the button omitted chart_columns_file, so it resolved every
chart field by generic guess while the same project's `junior ingest` used the site's
map, and the difference was frozen into the per-file sidecars before anyone could see
it. text_column was the same shape of hole, still open: ingest falls back to a column
literally named "text", which is silently wrong for any site whose export names it
something else.

So the button carries every corpus-shaping setting the project names, and this says so.
"""
from __future__ import annotations

from apps_and_interfaces.shiny_review_app.cohort_ingest import _ingest_cfg
from jr_pipeline.runtime_infrastructure.corpus_inheritance import CORPUS_AFFECTING_CFG_KEYS

# What the button owns: it exists to ingest the folder the operator picked, under a run
# it just made. Everything else about how a chart is read belongs to the project.
_THE_BUTTONS_OWN = {"run_id", "source_root", "files"}


def test_every_corpus_shaping_setting_reaches_the_ingest_the_button_runs(tmp_path, monkeypatch):
    named = {key: f"value-for-{key}" for key in CORPUS_AFFECTING_CFG_KEYS}
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_settings",
        lambda: dict(named),
    )
    # The map is a real file elsewhere; its resolution has its own tests.
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_column_map",
        lambda: "/somewhere/columns.yaml",
    )

    cfg = _ingest_cfg(tmp_path, "20990101_000000_aa")

    missing = sorted(
        key for key in CORPUS_AFFECTING_CFG_KEYS
        if key not in _THE_BUTTONS_OWN and key != "chart_columns_file"
        and cfg.get(key) != named[key]
    )
    assert not missing, (
        "the workbench ingest button drops these settings, so it reads a chart "
        f"differently from `junior ingest` on the same project: {missing}"
    )


def test_the_button_still_ingests_the_folder_the_operator_picked(tmp_path, monkeypatch):
    """The project's own source_root must not win: the point of the button is the
    folder in front of the operator, and a run id it just minted."""
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_settings",
        lambda: {"source_root": "/somewhere/else", "files": ["only-this.csv"]},
    )
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_column_map",
        lambda: None,
    )

    cfg = _ingest_cfg(tmp_path, "20990101_000000_aa")

    assert cfg["source_root"] == str(tmp_path)
    assert cfg["files"] == "auto"
    assert cfg["run_id"] == "20990101_000000_aa"


def test_a_named_but_unreadable_column_map_is_not_passed_on(tmp_path, monkeypatch):
    """Passing the raw unresolved name would have ingest look for a file that is not
    there. Unreadable is the same as unnamed, and both mean generic column handling —
    which the operator can at least be told about."""
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_settings",
        lambda: {"chart_columns_file": "columns.yaml"},
    )
    monkeypatch.setattr(
        "apps_and_interfaces.shiny_review_app.cohort_ingest._project_column_map",
        lambda: None,
    )

    assert "chart_columns_file" not in _ingest_cfg(tmp_path, "20990101_000000_aa")
