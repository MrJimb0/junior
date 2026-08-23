"""The workbench app imports, builds its UI + server, and shows an honest empty
state when nothing was selected.

Importing ``app`` executes everything that runs at ``shiny run`` boot: the UI tree
is constructed and the picker probes the receipts root on disk. This catches an
import-time break (a bad symbol, a UI construction error) that the unit tests
would miss because they never load app.py.

What a failed pick does instead of showing something else is pinned in
test_review_never_files_against_the_wrong_patient.py.
"""
from __future__ import annotations

import app as review_app
from shiny import App

from apps_and_interfaces.command_line_interface import PIPELINE_PHASES


def test_app_module_builds_a_shiny_app():
    assert isinstance(review_app.app, App)
    assert callable(review_app.server)
    assert review_app.MAX_EVIDENCE_SLOTS == 6
    assert isinstance(review_app.REVIEWABLE_RUNS, list)


def test_nothing_selected_is_an_empty_view_not_somebody_elses_rows():
    view = review_app._load_view(None, None)

    assert view.is_real_run is False
    assert view.nothing_selected is True
    assert view.rows == []
    assert view.run_id is None


def test_a_stale_pick_keeps_its_own_identity(tmp_path, monkeypatch):
    # A run/patient with no extract output on disk (a stale or deleted pick) renders
    # empty and keeps its own identity — it is the reviewer's pick failing, not a boot
    # with nothing picked yet.
    import run_results

    monkeypatch.setattr(run_results, "DATA_ROOT", tmp_path)
    view = review_app._load_view("20260101_010101_aa", "Nobody")

    assert view.is_real_run is False
    assert view.nothing_selected is False
    assert view.rows == []
    assert view.run_id == "20260101_010101_aa"


def test_the_stage_strip_matches_the_cli_stage_order():
    """The app and `junior --help` must name the same stages in the same order —
    two surfaces telling the operator two different pipelines is the drift this
    pins shut. The CLI's PIPELINE_PHASES is the single source of that order."""
    cli_order = " → ".join(
        name for phase in PIPELINE_PHASES.values() for name in phase
    )
    assert review_app.PIPELINE_STAGES == cli_order


def test_step_one_answers_underneath_its_own_button():
    """Reported as "scan folder does nothing in the app".

    Scanning worked; saying so did not. The scan's only reply was a line at the FOOT
    of the Start tab, past steps 2 and 3 — measured at 969px below the button, off the
    bottom of a 950px window. A folder that scans to nothing also renders an EMPTY
    picker, so a scan that failed changed nothing whatsoever on screen: the button
    read as dead. Step 1 answers between its own button and its own picker now."""
    rendered = str(review_app.app_ui)
    button = rendered.index('id="cohort_scan"')
    answer = rendered.index('id="cohort_scan_status"')
    picker = rendered.index('id="cohort_picker"')

    assert button < answer < picker, (
        "step 1's reply is not between the Scan button and the patient picker"
    )
    assert answer < rendered.index('id="cohort_status"'), (
        "the scan still reports at the foot of the tab, a screen below the button"
    )


def test_the_app_opens_on_a_scanned_cohort_with_its_patient_ticked():
    """Asked for as "open with the test patient selected".

    The folder box arrives already filled in, so the Scan click in front of it was a
    step nobody ever chose differently — and until it happened, every button below
    that needs a patient answered "Scan a folder first (step 1)". The bundled default
    has to scan to something, or the app opens on the same empty picker it used to."""
    from pathlib import Path

    import cohort_ingest

    result = cohort_ingest.scan_cohort(review_app.COHORT_INPUT_DEFAULT)

    assert not result.error, result.error
    assert [p.patient_id for p in result.patients] == ["Test_Patient"]
    assert review_app._scan_message(result)[0] == "ok"

    # Seeded from that scan when the session starts, rather than left at None for a
    # click to fill in. The picker ticks everything it is given (`selected=list(...)`).
    source = Path(review_app.__file__).read_text(encoding="utf-8")
    assert "cohort_ingest.scan_cohort(COHORT_INPUT_DEFAULT)" in source
    assert "reactive.Value(opening_scan)" in source


def test_a_scan_that_failed_is_marked_a_problem_not_a_result():
    """"Not a folder: ..." in the same blue as "Found 3 patient folders" is a failure
    dressed as an answer — and it is the opening scan that will hit this first, on any
    project whose input folder is not there yet."""

    class _Failed:
        error = "Not a folder: /nowhere"
        patients: list = []
        folder = "/nowhere"

    assert review_app._scan_message(_Failed()) == ("problem", "Not a folder: /nowhere")


def test_no_export_leaves_the_project_through_the_browser():
    """Asked for as "patient HIPAA is critical" after Review offered a Download CSV.

    A browser download writes chart values wherever that browser keeps downloads:
    outside the project tree the whole design keeps PHI inside, routinely cloud-synced,
    and invisible to every containment check the pipeline runs. Three buttons did it —
    the Review CSV and both whole-run value exports — and the Workbench one warned that
    the file was PHI in the same breath as offering the download. They write into the
    run's own answers folder now, beside the tables `junior extract` puts there."""
    from pathlib import Path

    source = Path(review_app.__file__).read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines()
                     if not line.strip().startswith("#"))

    assert "download_button" not in code, "the app offers a browser download again"
    assert "render.download" not in code, "the app has a download route again"
    assert "shiny-download" not in str(review_app.app_ui), (
        "a download control is rendered into the page"
    )


def test_an_export_lands_inside_the_projects_phi_tree(tmp_path):
    """Where the exports go, asked of the same helpers that put `junior extract`'s
    tables there — so the app and the CLI cannot disagree about the folder."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
        phi_root,
        run_output_dir,
    )

    destination = run_output_dir(phi_intermediate_run_dir("20260101_010101_aa", tmp_path))

    assert destination.is_relative_to(phi_root(tmp_path)), (
        f"an export would be written to {destination}, outside the PHI tree"
    )
