"""Junior — cohort extraction workbench (Shiny for Python).

Open it with ``junior workbench``, or directly from the repo root:
``python -m shiny run apps_and_interfaces/shiny_review_app/app.py``.

The app is the browser counterpart of the CLI, and it drives the same engine —
the Run button spawns ``junior run`` itself. Four screens:

  * Start — point at a folder of patient folders, pick the variables to extract,
    and run the pipeline (ingest → embed → index → extract), with preflight and
    ingest-only for checking inputs before any model loads;
  * Review — the selected run's per-variable values with the chart text each one
    came from, CSV export, and review capture (corrections, confirmations, the
    sampling frame, and evidence-relevance judgements);
  * Workbench — read-only inspection of the selected run: the recipe that ran,
    the evidence the LLM saw, the exact prompts and responses, validation
    verdicts, the NO_PHI exhaust inventory, and the shareable metadata export;
  * How it works — what each stage does.

With no run selected there is nothing to show and nothing to judge: the review
and workbench panels say so and point at Start, and every feedback writer
refuses. A selected run that cannot be read says so instead of quietly showing
something else in its place.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

import cohort_ingest
import pipeline_launcher
import project_control
import recipe_editing
import run_inspection

# When launched via `shiny run apps_and_interfaces/shiny_review_app/app.py`, the
# script's directory is on sys.path — siblings are top-level modules, not a package.
import run_results
from feedback_capture import (
    emit_relevance_label_exhaust,
    read_sampling_frame,
    write_chunk_relevance,
    write_confirmation,
    write_correction,
    write_sampling_frame,
)
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

# ── Review data source: pick a run + patient to review ───────────────────────
# The review tabs load a real pipeline run's per-variable result.json. A picker
# (wired in the sidebar, resolved server-side) lets the reviewer switch between
# any run/patient with extract output on disk — so a whole cohort is reviewable.
# resolve_run() honors JR_REVIEW_RUN_ID / JR_REVIEW_PATIENT_ID for the INITIAL
# selection, else the newest run's first patient.
# Feedback writes target the same data root the run was read from, cwd-independent.
ACTIVE_DR = run_results.DATA_ROOT
# Fixed pool of evidence-relevance button slots; only the active variable's chunks render.
MAX_EVIDENCE_SLOTS = 6

REVIEWABLE_RUNS: list[str] = run_results.list_reviewable_runs()
_BOOT_RUN = run_results.resolve_run()
DEFAULT_RUN_ID: str | None = _BOOT_RUN.run_id if _BOOT_RUN else None
DEFAULT_PATIENT_ID: str | None = _BOOT_RUN.patient_id if _BOOT_RUN else None

# How an operator produces a first run without the app, shown wherever the answer
# to "nothing here yet" is to go make a run.
FIRST_RUN_HINT = (
    "Run the pipeline on the Start tab — or at a shell: "
    "junior run --patient Test_Patient — then press ↻ Refresh runs."
)


def _patient_label(patient_id: str) -> str:
    """How the selected patient is named to the reviewer. A run whose patient folders
    are gone leaves the patient select empty, which has to read as such rather than as
    a blank where a patient id should be."""
    return f"patient {patient_id}" if patient_id else "no patient selected"


@dataclass
class ActiveView:
    """The run/patient currently under review, resolved for the render functions.

    Three states, and keeping them apart is what keeps a reviewer's judgement attached
    to the right patient:
      * a real run — rows read from its ``result.json`` files, feedback is writable;
      * nothing selected — no run has been picked (a fresh checkout has none to pick),
        so there is nothing to show and nothing a judgement could attach to;
      * a selection that produced nothing — a run WAS chosen and had no readable extract
        output for the selected patient (or had no patient at all). It keeps its run id
        and shows no rows. Substituting anything else here is how a reviewer files a
        correction against somebody they never saw.
    """

    rows: list[run_results.VariableRow]
    rows_by_variable: dict[str, run_results.VariableRow]
    run_id: str | None
    patient_id: str
    is_real_run: bool
    nothing_selected: bool
    source_label: str

    @property
    def export_run_id(self) -> str:
        """The run id written on every exported CSV row."""
        return self.run_id or "NO_RUN_selected"

    @property
    def export_source(self) -> str:
        """The provenance marker written on every exported CSV row."""
        if self.is_real_run:
            return "real_pipeline_run"
        if self.nothing_selected:
            return "NO_RUN_selected_nothing_extracted"
        return "selected_run_had_no_readable_extract_output"


def _nothing_selected_view() -> ActiveView:
    return ActiveView(
        rows=[],
        rows_by_variable={},
        run_id=None,
        patient_id="",
        is_real_run=False,
        nothing_selected=True,
        source_label=(
            "No run selected — this project has no completed extraction to review yet."
        ),
    )


def _unreadable_selection_view(run_id: str, patient_id: str, failure: str = "") -> ActiveView:
    """A run was selected and produced no readable rows. Named as itself rather than
    folded into the empty state: the reviewer asked for this patient, and showing them
    something else under their own selection is worse than showing them nothing."""
    detail = f" ({failure})" if failure else ""
    return ActiveView(
        rows=[],
        rows_by_variable={},
        run_id=run_id,
        patient_id=patient_id,
        is_real_run=False,
        nothing_selected=False,
        source_label=(
            f"run {run_id} · {_patient_label(patient_id)} — NO READABLE EXTRACT OUTPUT{detail}"
        ),
    )


def _load_view(run_id: str | None, patient_id: str | None) -> ActiveView:
    """Load the rows for one run+patient. The empty state stands in only when no run was
    selected; a selection that yields nothing says so instead.

    A run with no patient counts as a selection that yielded nothing: a run whose
    patient folders are gone leaves the patient select empty, and that is the reviewer's
    own pick failing, not a boot with nothing picked yet.
    """
    if not run_id:
        return _nothing_selected_view()
    if not patient_id:
        return _unreadable_selection_view(run_id, "")
    failure = ""
    try:
        rows = run_results.load_run_variables(run_id, patient_id)
    except Exception as could_not_load:  # noqa: BLE001 — reported, never swallowed
        rows, failure = [], str(could_not_load).strip()
    if not rows:
        return _unreadable_selection_view(run_id, patient_id, failure)
    return ActiveView(
        rows=rows,
        rows_by_variable={r.variable: r for r in rows},
        run_id=run_id,
        patient_id=patient_id,
        is_real_run=True,
        nothing_selected=False,
        source_label=f"real run {run_id} · patient {patient_id}",
    )


def _why_feedback_cannot_be_saved(view: ActiveView) -> str:
    """Empty for a real run; otherwise why no feedback can be written from this view,
    and what to do instead.

    Every feedback writer is irreversible and lands in the expert-label corpus that
    accuracy is later measured against. A correction filed with no run behind it, or
    against a run with no output, is a fabricated gold record that nothing downstream
    can tell from a real one — so the writers refuse rather than accept it quietly.
    """
    if view.is_real_run:
        return ""
    if view.nothing_selected:
        return (
            "No run is selected, so there is no extracted value a judgement could "
            f"attach to. {FIRST_RUN_HINT}"
        )
    if not view.patient_id:
        return (
            f"Run {view.run_id} has no patient with readable extract output, so there is "
            "nothing to judge. Pick another run in the sidebar, or re-run extract for "
            "this one, then press ↻ Refresh runs."
        )
    return (
        f"Run {view.run_id} · patient {view.patient_id} has no readable extract output, "
        "so there is no extracted value to judge. Re-run extract for this patient, then "
        "press ↻ Refresh runs."
    )


def _no_row_for_variable_reason(variable: str, view: ActiveView) -> str:
    """Why a judgement about a variable this patient has no row for is refused. Writing
    it anyway records an agreement with a value that was never produced."""
    return (
        f"{variable} was not extracted for patient {view.patient_id} in run "
        f"{view.run_id}, so there is no value to judge. Pick a variable from the list — "
        "it lists only what this run extracted for this patient."
    )


def _csv_export_name(view: ActiveView) -> str:
    """The export's filename. It lands in a folder next to real ones, so a no-output
    case has to be obvious from the name alone."""
    if view.is_real_run:
        return f"junior_extraction_{view.run_id}_{view.patient_id}.csv"
    if view.nothing_selected:
        return "junior_extraction_NO_RUN_selected.csv"
    return f"junior_extraction_NO_OUTPUT_{view.run_id}_{view.patient_id or 'no_patient'}.csv"


# Start tab: the input folder to scan (defaults to the bundled examples/) and the
# production home for raw patient folders, surfaced as a hint (often empty on a
# fresh checkout). Resolved once at boot.
COHORT_INPUT_DEFAULT = str(cohort_ingest.default_input_folder())
PHI_INPUT_FOLDER = cohort_ingest.phi_input_folder()

# What this checkout can extract, and anything that would stop a full run before the
# operator clicks Run. Both read the disk once, at boot.
AVAILABLE_VARIABLES: list[str] = pipeline_launcher.available_variables()
READINESS_PROBLEMS: list[str] = pipeline_launcher.readiness_problems()
# Pre-tick one cheap, proven variable so a first run is one click; the operator
# re-picks for real work.
DEFAULT_VARIABLE: list[str] = (
    ["date_of_birth"] if "date_of_birth" in AVAILABLE_VARIABLES else AVAILABLE_VARIABLES[:1]
)
# The pipeline's stages, named for the operator in the order they run. Retrieval and
# reranking happen inside extract, per recipe — the same stage model the CLI numbers
# (a guard test holds the two surfaces to the same order).
PIPELINE_STAGES = "ingest → embed → index → extract"


# ────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Inter", sans-serif; }
.app-title { font-size: 22px; font-weight: 600; margin: 0; }
.app-sub   { font-size: 13px; color: #6b6b6b; margin: 2px 0 0 0; }
.engine-pill {
    display: inline-block; padding: 3px 8px; border-radius: 999px;
    background: #e8f1ff; color: #1d4f9e; font-size: 11px; font-weight: 500;
}
.evidence-card {
    border: 1px solid #e1e6ee; border-radius: 8px; padding: 12px 14px;
    margin-bottom: 10px; background: #fafbfd;
}
.evidence-header {
    display: flex; justify-content: space-between; font-size: 12px;
    color: #5a6273; margin-bottom: 6px;
}
.evidence-rank {
    display: inline-block; background: #1d4f9e; color: #fff;
    border-radius: 4px; padding: 1px 7px; font-size: 11px;
    font-weight: 600; margin-right: 8px;
}
.evidence-text { font-size: 13px; line-height: 1.5; white-space: pre-wrap; }
.section-title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.06em;
                 color: #6b6b6b; font-weight: 600; margin: 18px 0 8px 0; }

.extract-table {
    width: 100%; border-collapse: collapse; font-size: 12.5px;
    margin-top: 8px;
}
.extract-table th {
    background: #f4f7fc; text-align: left; padding: 7px 10px;
    border-bottom: 2px solid #1d4f9e; font-weight: 600; color: #1d4f9e;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
}
.extract-table td {
    padding: 8px 10px; border-bottom: 1px solid #e8ecf2;
    vertical-align: top; line-height: 1.4;
}
.extract-table tr:nth-child(even) td { background: #fafbfd; }
.extract-table .var-name { font-weight: 600; color: #1d4f9e; white-space: nowrap; }
.extract-table .var-value { font-weight: 500; color: #111; }
.extract-table .var-quote {
    font-style: italic; color: #444; font-size: 12px; max-width: 380px;
}
.extract-table .var-source { font-size: 11px; color: #6b6b6b; white-space: nowrap; }
.extract-table .var-conf {
    font-variant-numeric: tabular-nums; text-align: right;
}
.csv-preview {
    background: #1e2330; color: #d8dde8; padding: 12px 14px;
    border-radius: 6px; font-family: ui-monospace, Menlo, monospace;
    font-size: 11.5px; line-height: 1.5; overflow-x: auto; white-space: pre;
    max-height: 260px;
}
.extract-meta {
    display: flex; gap: 18px; font-size: 12px; color: #5a6273;
    margin: 8px 0 14px 0; flex-wrap: wrap;
}
.extract-meta b { color: #1d4f9e; }

.start-step {
    border: 1px solid #e1e6ee; border-radius: 8px; padding: 14px 16px;
    margin-bottom: 14px; background: #fff;
}
.start-step h5 {
    margin: 0 0 6px 0; font-size: 13px; text-transform: uppercase;
    letter-spacing: 0.06em; color: #1d4f9e; font-weight: 600;
}
.start-step .hint { font-size: 12px; color: #6b6b6b; margin: 0 0 10px 0; }
.stage-strip {
    font-size: 12px; color: #1d4f9e; background: #eef3fc; border-radius: 6px;
    padding: 7px 10px; display: inline-block; margin-bottom: 12px;
}
.readiness {
    border-left: 3px solid #d9822b; background: #fdf6ec; padding: 10px 12px;
    border-radius: 6px; font-size: 12px; color: #7a4b12; margin-bottom: 14px;
}
.readiness ul { margin: 6px 0 0 16px; padding: 0; }
.run-log {
    background: #1e2330; color: #d8dde8; padding: 12px 14px; border-radius: 6px;
    font-family: ui-monospace, Menlo, monospace; font-size: 11.5px;
    line-height: 1.5; overflow: auto; white-space: pre-wrap;
    max-height: 340px; margin-top: 10px;
}
.suggested-q {
    display: block; text-align: left; margin: 4px 0;
    font-size: 13px; padding: 7px 9px; border-radius: 6px;
    background: #fff; border: 1px solid #d8dde6; cursor: pointer;
}
.suggested-q:hover { background: #eef3fc; border-color: #9bb7e0; }
.inspect-block {
    border: 1px solid #e1e6ee; border-radius: 8px; padding: 12px 14px;
    margin-bottom: 12px; background: #fafbfd;
}
.inspect-block h6 {
    margin: 0 0 6px 0; font-size: 12px; color: #1d4f9e; font-weight: 600;
}
.inspect-block pre {
    background: #1e2330; color: #d8dde8; padding: 10px 12px; border-radius: 6px;
    font-family: ui-monospace, Menlo, monospace; font-size: 11.5px;
    line-height: 1.5; overflow: auto; white-space: pre-wrap;
    max-height: 320px; margin: 0;
}
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.tags.div(
            ui.tags.p("Junior", class_="app-title"),
            ui.tags.p("Cohort extraction workbench", class_="app-sub"),
        ),
        ui.output_ui("sidebar_project"),
        ui.tags.p(
            "Everything below runs on this machine. No chart data leaves it.",
            style="font-size: 11px; color: #6b6b6b; margin-top: 10px;",
        ),
        ui.tags.div("Review run", class_="section-title"),
        # Always rendered (even with no runs yet) so a run produced after boot can be
        # reached in-session via Refresh — the choices are refreshed server-side.
        ui.input_select(
            "review_run", "Run",
            choices=REVIEWABLE_RUNS, selected=DEFAULT_RUN_ID, width="100%",
        ),
        ui.input_select(
            "review_patient", "Patient",
            choices=(
                run_results.list_patients_with_extract(DEFAULT_RUN_ID)
                if DEFAULT_RUN_ID else []
            ),
            selected=DEFAULT_PATIENT_ID, width="100%",
        ),
        ui.input_action_button(
            "review_refresh", "↻ Refresh runs",
            class_="suggested-q", style="width:auto; padding:4px 10px; margin-top:2px;",
        ),
        (
            ui.tags.p(
                f"No completed runs yet. {FIRST_RUN_HINT}",
                style="font-size: 11px; color: #6b6b6b; margin-top: 6px;",
            )
            if not REVIEWABLE_RUNS else ui.tags.span()
        ),
        ui.output_ui("data_source_pill"),
        width=320,
        bg="#ffffff",
    ),
    ui.tags.style(CUSTOM_CSS),
    ui.navset_tab(
        ui.nav_panel(
            "Start",
            ui.tags.h4("Extract variables from a cohort",
                       style="margin: 4px 0 6px 0;"),
            ui.tags.p(
                "Point Junior at a folder of patient folders, choose the variables to "
                "extract, and run the pipeline. Every value it produces is reviewable, "
                "with the chart text it came from, on the Review tab.",
                style="color: #6b6b6b; font-size: 13px; max-width: 760px;",
            ),
            ui.tags.div(PIPELINE_STAGES, class_="stage-strip"),
            ui.output_ui("readiness_panel"),
            ui.tags.div(
                ui.tags.h5("0 · Project"),
                ui.tags.p(
                    "The cohort every panel below reads and writes. Switching re-pins "
                    "the app and every run it spawns, exactly as `junior workbench` "
                    "pins it at launch.",
                    class_="hint",
                ),
                ui.output_ui("pr_current"),
                ui.tags.div(
                    ui.input_select("pr_known", None, choices=[], width="420px"),
                    ui.input_action_button("pr_switch", "Switch to selected"),
                    style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;",
                ),
                ui.tags.div(
                    ui.input_text("pr_name", None, placeholder="new_project_name", width="200px"),
                    ui.input_text("pr_charts", None,
                                  placeholder="folder of patient folders", width="260px"),
                    ui.input_text("pr_into", None,
                                  placeholder="where the project folder goes", width="260px"),
                    ui.input_action_button("pr_create", "New project", class_="btn-primary"),
                    style="display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-top:8px;",
                ),
                ui.output_ui("pr_status"),
                class_="start-step",
            ),
            ui.tags.div(
                ui.tags.h5("1 · Cohort"),
                ui.tags.p(
                    f"Production cohorts live under {PHI_INPUT_FOLDER} (often empty on a "
                    "fresh checkout). The default below is the bundled examples/ folder — "
                    "one synthetic patient.",
                    class_="hint",
                ),
                ui.input_text(
                    "cohort_folder", "Input folder (its subfolders are the patients)",
                    value=COHORT_INPUT_DEFAULT, width="100%",
                ),
                ui.input_action_button("cohort_scan", "Scan folder", class_="btn-primary"),
                ui.output_ui("cohort_scan_status"),
                ui.output_ui("cohort_picker"),
                class_="start-step",
            ),
            ui.tags.div(
                ui.tags.h5("2 · Variables"),
                ui.tags.p(
                    "One recipe per variable, discovered under var_extraction_recipes/. "
                    "Each selected variable runs its recipe (retrieve → rerank → extract) "
                    "on every selected patient.",
                    class_="hint",
                ),
                (
                    ui.tags.div(
                        ui.tags.div(
                            ui.input_action_link("vars_all", "Select all"),
                            ui.tags.span(" · ", style="color: #ccc;"),
                            ui.input_action_link("vars_none", "Clear"),
                            style="font-size: 12px; margin-bottom: 2px;",
                        ),
                        ui.input_checkbox_group(
                            "run_variables", None,
                            choices=AVAILABLE_VARIABLES,
                            selected=DEFAULT_VARIABLE,
                        ),
                    )
                    if AVAILABLE_VARIABLES else
                    ui.tags.p("No recipes found — nothing to extract.",
                              style="color:#b03a2e; font-size: 12px;")
                ),
                class_="start-step",
            ),
            ui.tags.div(
                ui.tags.h5("3 · Run"),
                ui.tags.p(
                    "Run the whole pipeline, or use the text-only steps to check the inputs "
                    "first: preflight reports what would block a patient, and ingest-only "
                    "writes the structured parquets without loading any model.",
                    class_="hint",
                ),
                ui.tags.div(
                    ui.input_action_button(
                        "run_pipeline", "▶ Run pipeline", class_="btn-primary",
                    ),
                    ui.input_action_button("run_stop", "Stop run"),
                    ui.input_action_button("cohort_preflight", "Preflight selected"),
                    ui.input_action_button("cohort_ingest", "Ingest only (Step 1)"),
                    style="display: flex; gap: 12px; flex-wrap: wrap;",
                ),
                ui.tags.p(
                    "Or one stage at a time — each button runs the command of the same "
                    "name, the way `junior embed` runs it at a shell. A stage takes its "
                    "cohort from this project rather than from the folder box above, "
                    "because the command itself has no folder to be given. The stages "
                    "continue whichever run is named below; New run starts a fresh one "
                    "by reading the charts in again.",
                    class_="hint", style="margin: 14px 0 8px 0;",
                ),
                ui.tags.div(
                    ui.input_action_button("run_new", "New run"),
                    style="display: flex; gap: 12px; margin-bottom: 8px;",
                ),
                ui.tags.div(
                    *[ui.input_action_button(f"run_{stage}", f"{number} {stage.title()}")
                      for number, stage in enumerate(pipeline_launcher.STAGES, start=1)],
                    style="display: flex; gap: 12px; flex-wrap: wrap;",
                ),
                ui.output_ui("stage_run_line"),
                ui.output_ui("run_status"),
                ui.output_ui("run_log"),
                class_="start-step",
            ),
            ui.output_ui("cohort_status"),
            ui.output_ui("cohort_results"),
        ),
        ui.nav_panel(
            "Review",
            ui.tags.p(
                "Per-variable structured export. Every value traces to a verbatim chart "
                "quote — this is what the research dataset row for this patient looks like.",
                style="color: #6b6b6b; font-size: 13px;",
            ),
            ui.output_ui("extract_meta_panel"),
            ui.output_ui("extract_table_panel"),
            ui.tags.div("CSV preview", class_="section-title"),
            ui.output_ui("extract_csv_panel"),
            ui.tags.div(
                ui.input_action_button("save_csv", "Save CSV", class_="btn-primary"),
                style="margin-top: 10px;",
            ),
            ui.tags.p(
                "Written beside this run's other answers, inside the project's own "
                "CONTAINS_PHI folder — the same place `junior extract` puts them. These "
                "are patient values, so they stay in the project tree rather than going "
                "to a browser's downloads folder.",
                class_="hint", style="margin-top: 6px;",
            ),
            ui.output_ui("save_csv_status"),
            ui.tags.div("Flag a value", class_="section-title"),
            ui.tags.p(
                "Spot a wrong value? Flag it and supply the correction. Saved as "
                "expert ground truth (extraction_correction) — `junior eval-values` "
                "scores the run against it.",
                style="color: #6b6b6b; font-size: 12px;",
            ),
            # Told before the click, not after it: everything below writes an
            # irreversible expert-label record accuracy is later measured against.
            ui.output_ui("feedback_write_gate_notice"),
            ui.tags.div(
                # Choices are (re)populated server-side from the active run/patient
                # whenever the picker changes (see the review_selection effect).
                ui.input_select(
                    "fb_variable", "Variable",
                    choices=[r.variable for r in (_BOOT_RUN.rows if _BOOT_RUN else [])],
                    width="280px",
                ),
                ui.input_text("fb_correct", "Correct value", placeholder="e.g. 2024-01-22", width="280px"),
                ui.input_action_button("fb_submit", "Submit correction", class_="btn-primary"),
                # Confirm-correct supplies the agreed cases the sensitivity
                # denominator needs (without it, reviewed cases are an error-enriched sample).
                ui.input_action_button("fb_confirm", "Confirm correct", class_="btn-success"),
                # Record which (patient, variable) were drawn for review -- the
                # sampling frame that makes the denominator unbiased.
                ui.input_action_button("fb_draw", "Record review sample"),
                style="display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap;",
            ),
            ui.output_ui("fb_status"),
            # Per-chunk relevance judgments — the training signal for a future
            # reranker. Tied to the selected variable's real evidence chunk ids.
            ui.tags.div("Judge evidence relevance", class_="section-title"),
            ui.tags.p(
                "For the variable selected above, mark each retrieved evidence chunk "
                "relevant or not. Saved as chunk_relevance, and mirrored into the run's "
                "NO_PHI exhaust so the label joins the ranks it graded.",
                style="color: #6b6b6b; font-size: 12px;",
            ),
            ui.output_ui("chunk_relevance_panel"),
            ui.output_ui("cr_status"),
        ),
        ui.nav_panel(
            "Workbench",
            ui.tags.p(
                "Read-only inspection of the selected run — the recipe that ran, the "
                "evidence the model saw, the exact prompts and responses, the validation "
                "verdicts, and the run's shareable summary. Everything here is read from "
                "the run's own artifacts.",
                style="color: #6b6b6b; font-size: 13px; max-width: 760px;",
            ),
            ui.output_ui("wb_summary_panel"),
            ui.tags.div(
                ui.input_select("wb_variable", "Variable", choices=[
                    r.variable for r in (_BOOT_RUN.rows if _BOOT_RUN else [])
                ], width="280px"),
                style="margin-top: 6px;",
            ),
            ui.tags.div("Recipe", class_="section-title"),
            ui.output_ui("wb_recipe_panel"),
            ui.tags.div("Evidence the model saw", class_="section-title"),
            ui.output_ui("wb_evidence_panel"),
            ui.tags.div("Evidence selection", class_="section-title"),
            ui.output_ui("wb_selection_panel"),
            ui.tags.div("LLM exchange", class_="section-title"),
            ui.output_ui("wb_exchange_panel"),
            ui.tags.div("Validation verdicts", class_="section-title"),
            ui.output_ui("wb_invariants_panel"),
            ui.tags.div("Patient run files", class_="section-title"),
            ui.output_ui("wb_files_panel"),
            ui.tags.div("Whole-run values", class_="section-title"),
            ui.tags.p(
                "Every patient's extracted values in one table — what a cohort looks "
                "like after a run. Wide is one row per patient; long is one row per "
                "value and stays readable when a recipe returns a list. THIS FILE IS "
                "PHI, so it is written beside the chart data in this run's answers "
                "folder — never the shareable tree, and never a downloads folder.",
                style="color: #6b6b6b; font-size: 12px;",
            ),
            ui.tags.div(
                ui.input_action_button("wb_values_wide", "Save values — wide CSV"),
                ui.input_action_button("wb_values_long", "Save values — long CSV"),
                style="display:flex; gap:12px;",
            ),
            ui.output_ui("wb_values_status"),
            ui.tags.div("Share this run", class_="section-title"),
            ui.tags.p(
                "The NO_PHI exhaust is the run's de-identified inventory. Export packages "
                "it with the result tables and the exact code and settings that produced "
                "them — no patient data; every file is scanned first, and one hit stops "
                "the export. The same artifact as `junior export-metadata`.",
                style="color: #6b6b6b; font-size: 12px;",
            ),
            ui.output_ui("wb_exhaust_panel"),
            ui.input_action_button("wb_export", "Write shareable zip", class_="btn-primary"),
            ui.output_ui("wb_export_status"),
        ),
        ui.nav_panel(
            "Recipes",
            ui.tags.p(
                "Author and edit the variable recipes — the YAML, prompts, output "
                "schema and helper on disk are what every run reads. Each save is "
                "validated by the pipeline's own recipe loader before it stands; an "
                "edit the loader refuses is rolled back with its message shown here.",
                style="color: #6b6b6b; font-size: 13px; max-width: 760px;",
            ),
            ui.tags.div(
                ui.tags.h5("Recipe"),
                ui.tags.div(
                    ui.input_select("rc_recipe", None, choices=[], width="420px"),
                    ui.input_action_button("rc_refresh", "↻ Refresh"),
                    ui.input_action_button("rc_new_version", "New version"),
                    style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;",
                ),
                ui.output_ui("rc_draft_notice"),
                class_="start-step",
            ),
            ui.tags.div(
                ui.tags.h5("Edit"),
                ui.input_select("rc_file", "File", choices=[], width="420px"),
                ui.input_text_area(
                    "rc_editor", None, value="", rows=22, width="100%", spellcheck="false",
                ),
                ui.tags.div(
                    ui.input_action_button("rc_save", "Save", class_="btn-primary"),
                    ui.input_action_button("rc_finish", "Finish draft", class_="btn-success"),
                    style="display:flex; gap:12px; margin-top:8px;",
                ),
                ui.output_ui("rc_save_status"),
                class_="start-step",
            ),
            ui.tags.div(
                ui.tags.h5("New recipe"),
                ui.tags.p(
                    "Copy a working recipe into a new variable. The wiring — file "
                    "names, schema and prompt paths, the helper module — is rewritten "
                    "for you; the words are not, so the copy is a draft the pipeline "
                    "refuses to run until you finish it here.",
                    class_="hint",
                ),
                ui.tags.div(
                    ui.input_text("rc_new_name", None,
                                  placeholder="new_variable_name", width="260px"),
                    ui.input_select("rc_new_template", None, choices=[], width="260px"),
                    ui.input_action_button("rc_create", "Create draft", class_="btn-primary"),
                    style="display:flex; gap:12px; align-items:center; flex-wrap:wrap;",
                ),
                ui.output_ui("rc_create_status"),
                class_="start-step",
            ),
        ),
        ui.nav_panel(
            "How it works",
            ui.tags.div(
                ui.tags.h4("What a run does, stage by stage"),
                ui.tags.ol(
                    ui.tags.li(
                        ui.tags.b("Ingest"),
                        " — each patient's source files are preserved whole as "
                        "structured parquets. Text-only; no model.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Embed"),
                        " — the chart is chunked and encoded with a local clinical "
                        "encoder, on this machine.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Index"),
                        " — the chunk vectors go into a per-patient HNSW index.",
                    ),
                    ui.tags.li(
                        ui.tags.b("Extract"),
                        " — per variable, the recipe's queries retrieve candidate chunks, "
                        "reranking orders them into the evidence packet, and a local LLM "
                        "fills the recipe's output schema from it. Every value carries "
                        "the chunk id it came from, which is what makes it reviewable "
                        "on the Review tab.",
                    ),
                    ui.tags.li(
                        "Reviewer corrections, confirmations, and evidence-relevance "
                        "judgments are captured beside the run — `junior eval-values` "
                        "scores the run against them."
                    ),
                ),
                ui.tags.p(
                    "Nothing leaves this machine: local weights, no API calls, no telemetry. "
                    "The Workbench tab reads the run's own artifacts, so what it shows is "
                    "what actually happened."
                ),
                style="max-width: 760px; padding: 14px 6px;",
            ),
        ),
        id="main_tabs",
    ),
    title="Junior — cohort extraction",
    fillable=False,
)


# ────────────────────────────────────────────────────────────────────
# Server
# ────────────────────────────────────────────────────────────────────

def _newest_run_here() -> str:
    """The newest run under this project's data root, or "" if it has none yet."""
    from jr_pipeline.runtime_infrastructure.project_context import newest_run_id

    try:
        return newest_run_id(run_results.DATA_ROOT) or ""
    except Exception:
        return ""


def _scan_message(result) -> tuple[str, str]:
    """Step 1's message line for a scan: its level, and what it says."""
    if result.error:
        return ("problem", result.error)
    how_many = len(result.patients)
    return ("ok", f"Found {how_many} patient folder{'s' if how_many != 1 else ''} "
                  f"in {result.folder}.")


def server(input: Inputs, output: Outputs, session: Session) -> None:
    # ── Review selection: which run + patient the Review/Workbench tabs show.
    #    cur() is the single source the render + feedback functions read, so
    #    switching the picker walks a whole cohort without restarting the app. ──
    sel_run = reactive.Value(DEFAULT_RUN_ID or "")
    sel_patient = reactive.Value(DEFAULT_PATIENT_ID or "")

    @reactive.Calc
    def cur() -> ActiveView:
        return _load_view(sel_run() or None, sel_patient() or None)

    # ignore_init: the sidebar selects are already seeded to DEFAULT_RUN_ID /
    # DEFAULT_PATIENT_ID at construction, so a boot-time firing would only clobber a
    # non-first JR_REVIEW_PATIENT_ID. Only react to an actual user switch.
    @reactive.Effect
    @reactive.event(input.review_run, ignore_init=True)
    def _on_review_run():
        run_id = input.review_run() or ""
        patients = run_results.list_patients_with_extract(run_id) if run_id else []
        first = patients[0] if patients else ""
        sel_run.set(run_id)
        sel_patient.set(first)
        ui.update_select("review_patient", choices=patients, selected=first)

    @reactive.Effect
    @reactive.event(input.review_patient)
    def _on_review_patient():
        sel_patient.set(input.review_patient() or "")

    # Re-read the runs on disk so a run produced after boot (e.g. by another process
    # or a project run) becomes reviewable without restarting; keep the current
    # selection if it still exists, else jump to the newest. Shared with the project
    # switch, which changes which disk the runs are read from.
    def _refresh_runs_now(keep_selection: bool = True) -> None:
        runs = run_results.list_reviewable_runs()
        keep_run = (input.review_run() or "") if keep_selection else ""
        run_id = keep_run if keep_run in runs else (runs[0] if runs else "")
        patients = run_results.list_patients_with_extract(run_id) if run_id else []
        keep_pat = (input.review_patient() or "") if keep_selection else ""
        patient = keep_pat if keep_pat in patients else (patients[0] if patients else "")
        sel_run.set(run_id)
        sel_patient.set(patient)
        ui.update_select("review_run", choices=runs, selected=run_id or None)
        ui.update_select("review_patient", choices=patients, selected=patient or None)

    @reactive.Effect
    @reactive.event(input.review_refresh)
    def _on_refresh_runs():
        _refresh_runs_now()

    # ── Project panel: show, switch, create — the CLI's own machinery ──
    pr_current_value = reactive.Value(project_control.current_project())
    pr_status_value = reactive.Value("")

    def _pr_refresh_known(select: str | None = None) -> None:
        choices = {
            str(path): f"{path.parent.name}  ·  {path}"
            for path in project_control.known_projects()
        }
        selected = select if select in choices else next(iter(choices), None)
        ui.update_select("pr_known", choices=choices, selected=selected)

    @reactive.Effect
    def _pr_boot():
        with reactive.isolate():
            _pr_refresh_known()

    def _pr_switch_everything_to(info) -> None:
        """The one place the app changes cohorts: module state, env for children,
        the pickers, and the Start tab's folder box all move together."""
        global ACTIVE_DR
        run_results.DATA_ROOT = info.output_root
        ACTIVE_DR = info.output_root
        pr_current_value.set(info)
        _refresh_runs_now(keep_selection=False)
        if info.source_root:
            ui.update_text("cohort_folder", value=info.source_root)

    @reactive.Effect
    @reactive.event(input.pr_switch)
    def _pr_on_switch():
        chosen = input.pr_known() or ""
        if not chosen:
            pr_status_value.set("No project selected.")
            return
        try:
            info = project_control.switch_to(Path(chosen))
        except Exception as failed:  # noqa: BLE001 — reported, never swallowed
            pr_status_value.set(f"Not switched. {failed}")
            return
        _pr_switch_everything_to(info)
        pr_status_value.set(f"Switched to {info.name} — reading {info.output_root}")

    @reactive.Effect
    @reactive.event(input.pr_create)
    def _pr_on_create():
        name = (input.pr_name() or "").strip()
        charts = (input.pr_charts() or "").strip()
        into = (input.pr_into() or "").strip()
        if not name or not charts or not into:
            pr_status_value.set(
                "Name the project, point at the folder of patient folders, and say "
                "where the project's own folder should go."
            )
            return
        output, config_path = project_control.create_project(name, charts, into)
        if config_path is None:
            pr_status_value.set(f"Nothing was created. {output.splitlines()[-1] if output else ''}")
            return
        info = project_control.switch_to(config_path)
        _pr_switch_everything_to(info)
        _pr_refresh_known(select=str(config_path))
        ui.update_text("pr_name", value="")
        pr_status_value.set(
            f"Created and switched to {info.name}. Its settings live at "
            f"{info.config_path} — `junior columns` maps your export's column names "
            "before the first ingest."
        )

    @output
    @render.ui
    def pr_current():
        info = pr_current_value.get()
        if info is None:
            return ui.HTML('<div class="readiness">No project is pinned — create one '
                           'below, or open the app with `junior workbench` from a '
                           'project folder.</div>')
        return ui.HTML(
            '<div class="extract-meta">'
            f'<span><b>Project:</b> {html.escape(info.name)}</span>'
            f'<span><b>Charts:</b> {html.escape(info.source_root or "—")}</span>'
            f'<span><b>Output:</b> {html.escape(str(info.output_root))}</span>'
            '</div>'
        )

    @output
    @render.ui
    def pr_status():
        msg = pr_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        color = "#b03a2e" if ("Not switched" in msg or "Nothing was" in msg
                              or "No project" in msg) else "#1d4f9e"
        return ui.tags.p(msg, style=f"font-size: 12px; color: {color}; margin-top: 8px;")

    @output
    @render.ui
    def sidebar_project():
        info = pr_current_value.get()
        name = info.name if info else "no project pinned"
        return ui.tags.p(f"project: {name}",
                         style="font-size: 11px; color: #1d4f9e; margin-top: 6px;")

    # Keep the correction and workbench variable dropdowns in step with the active
    # run/patient, preserving the reviewer's current pick when it still exists.
    @reactive.Effect
    def _sync_fb_variable_choices():
        variables = [r.variable for r in cur().rows]
        with reactive.isolate():
            current = input.fb_variable() or ""
        selected = current if current in variables else None
        ui.update_select("fb_variable", choices=variables, selected=selected)

    @reactive.Effect
    def _sync_wb_variable_choices():
        variables = [r.variable for r in cur().rows]
        with reactive.isolate():
            current = input.wb_variable() or ""
        selected = current if current in variables else None
        ui.update_select("wb_variable", choices=variables, selected=selected)

    @output
    @render.ui
    def data_source_pill():
        # A selection with nothing behind it is a loud banner, not a grey aside: a
        # reviewer who misses it spends an hour judging values that are not there.
        view = cur()
        if view.is_real_run:
            pill, pill_style, label_style = "Real run", "", "color:#6b6b6b;"
        elif view.nothing_selected:
            pill = "NO RUN SELECTED"
            pill_style = "background:#d9822b; color:#fff; font-weight:700;"
            label_style = "color:#7a4b12; font-weight:600;"
        else:
            pill = "SELECTED RUN HAS NOTHING TO REVIEW"
            pill_style = "background:#d9822b; color:#fff; font-weight:700;"
            label_style = "color:#7a4b12; font-weight:600;"
        return ui.tags.div(
            ui.tags.span(pill, class_="engine-pill", style=pill_style),
            ui.tags.p(
                view.source_label,
                style=f"font-size: 11px; {label_style} margin-top: 6px; word-break: break-all;",
            ),
            (
                ui.tags.p(
                    "Feedback capture is off in this state — nothing here can be saved "
                    "as expert ground truth.",
                    style=f"font-size: 11px; {label_style} margin-top: 4px;",
                )
                if not view.is_real_run else ui.tags.span()
            ),
        )

    @output
    @render.ui
    def feedback_write_gate_notice():
        refusal = _why_feedback_cannot_be_saved(cur())
        if not refusal:
            return ui.tags.span()
        return ui.HTML(
            '<div class="readiness" style="border-left-color:#b03a2e; '
            'background:#fdecea; color:#b03a2e;"><b>Feedback capture is off.</b> '
            f'{html.escape(refusal)}</div>'
        )

    # ── Review tab (the selected run's result.json rows) ──
    @output
    @render.ui
    def extract_meta_panel():
        view = cur()
        rows = view.rows
        if not rows:
            # A run selected with no patient at all, and a patient whose extract output
            # cannot be read, are different situations with different next steps — and
            # telling the first "re-run extract for this patient" names a patient nobody
            # picked. The write-gate helper already separates them, so ask it rather than
            # keeping a second, shorter list of the states here.
            return ui.HTML(
                '<div class="readiness">'
                f'<b>Nothing to review.</b> {html.escape(view.source_label)}<br>'
                'No value can be corrected, confirmed or judged from here. '
                f'{html.escape(_why_feedback_cannot_be_saved(view))}'
                '</div>'
            )
        n_ok = sum(1 for r in rows if r.ok)
        n_failed = sum(1 for r in rows if r.ok is False)
        # One changed chart row invalidates every offset this run recorded, so the
        # warning belongs above the whole table, not beside the quote that tripped it.
        # What clears it is not one fixed command: a row that changed since embed and a
        # table that changed since ingest both land here, and re-running embed does
        # nothing for the second. The evidence carries which step it needs, so take it
        # from there rather than naming one and being wrong half the time.
        next_steps = sorted({
            chunk.chart_changed_next_step
            for row in rows for chunk in row.evidence if chunk.chart_changed_next_step
        })
        stale_banner = (
            '<div class="readiness" style="border-left-color:#b03a2e; '
            'background:#fdecea; color:#b03a2e;">'
            "<b>This run's evidence no longer matches the chart it came from.</b> "
            "The chart changed after the pipeline recorded it, so these quotes cannot "
            "be trusted to be the text the model read. Before exporting or reviewing "
            f"these values, {html.escape(' and '.join(next_steps))}."
            '</div>'
            if next_steps else ""
        )
        return ui.HTML(
            f'{stale_banner}'
            f'<div class="extract-meta">'
            f'<span><b>Patient:</b> {html.escape(view.patient_id)}</span>'
            f'<span><b>Variables:</b> {len(rows)}</span>'
            f'<span><b>Outcomes:</b> {n_ok} ok, {n_failed} failed</span>'
            f'<span><b>Run:</b> {html.escape(view.export_run_id)}</span>'
            f'</div>'
        )

    @output
    @render.ui
    def extract_table_panel():
        rows = cur().rows
        if not rows:
            return ui.tags.p(
                "No extracted variables to show.",
                style="color: #6b6b6b; font-size: 13px;",
            )

        def conf_cell(r: run_results.VariableRow) -> str:
            return "—" if r.confidence is None else f"{r.confidence:.2f}"

        def ok_cell(r: run_results.VariableRow) -> str:
            if r.ok is None:
                return ""
            return (
                '<span style="color:#1e7e34;">ok</span>'
                if r.ok
                else '<span style="color:#b03a2e;">failed</span>'
            )

        def quote_cell(r: run_results.VariableRow) -> str:
            # An unreadable quote renders as its reason, in red and unquoted, so it is
            # never mistaken for a verbatim phrase the chart actually contained.
            quote = html.escape(r.evidence_quote)
            if r.evidence_text_is_readable:
                return f'"{quote}"'
            return f'<span style="color:#b03a2e; font-style:normal;">{quote}</span>'

        body_rows = "\n".join(
            f"""
<tr>
  <td class="var-name">{html.escape(r.variable)}</td>
  <td class="var-value">{html.escape(r.value)}</td>
  <td class="var-conf">{conf_cell(r)}</td>
  <td class="var-conf">{ok_cell(r)}</td>
  <td class="var-source">{html.escape(r.evidence_source)}</td>
  <td class="var-quote">{quote_cell(r)}</td>
  <td class="var-source">{html.escape(r.model)}</td>
</tr>
"""
            for r in rows
        )
        return ui.HTML(
            f"""
<table class="extract-table">
  <thead>
    <tr>
      <th>Variable</th><th>Value</th><th>Conf</th><th>OK</th>
      <th>Evidence chunk</th><th>Evidence quote</th><th>Recipe</th>
    </tr>
  </thead>
  <tbody>{body_rows}</tbody>
</table>
"""
        )

    @output
    @render.ui
    def extract_csv_panel():
        view = cur()
        rows = view.rows
        if not rows:
            return ui.tags.p("—", style="color: #6b6b6b;")
        csv_text = run_results.rows_as_csv(
            rows, view.patient_id,
            run_id=view.export_run_id, source=view.export_source,
        )
        # Preview first ~12 lines so it fits on screen
        lines = csv_text.splitlines()
        preview = "\n".join(lines[:13])
        if len(lines) > 13:
            preview += f"\n… ({len(lines)-13} more rows)"
        return ui.HTML(
            f'<div class="csv-preview">{html.escape(preview)}</div>'
        )

    def _save_beside_the_run(view, filename: str, text: str) -> tuple[str, str]:
        """Write one export into the run's own answers folder, and say where it went.

        Deliberately not a browser download. These are chart values, and a download
        puts them wherever the browser keeps downloads: outside the project tree the
        whole design keeps PHI inside, routinely cloud-synced, and invisible to every
        containment check the pipeline runs. The run's answers folder is where
        `junior extract` already writes its tables, so an export lands beside them.

        ACTIVE_DR, not the boot-time data root: switching project re-points it, and an
        export written to where the app STARTED would land in another cohort's tree."""
        from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
            phi_intermediate_run_dir,
            run_output_dir,
        )

        destination = run_output_dir(phi_intermediate_run_dir(view.run_id, ACTIVE_DR))
        try:
            destination.mkdir(parents=True, exist_ok=True)
            written = destination / filename
            written.write_text(text, encoding="utf-8")
        except OSError as could_not_write:
            return ("problem", f"Nothing was saved. {could_not_write}")
        return ("ok", f"Saved to {written}")

    save_csv_status_value: reactive.Value = reactive.Value(("", ""))   # (level, message)

    @reactive.Effect
    @reactive.event(input.save_csv)
    def _on_save_csv():
        view = cur()
        if not view.is_real_run:
            save_csv_status_value.set(
                ("problem", "No run selected, so there are no values to save.")
            )
            return
        save_csv_status_value.set(_save_beside_the_run(
            view, _csv_export_name(view),
            run_results.rows_as_csv(
                view.rows, view.patient_id,
                run_id=view.export_run_id, source=view.export_source,
            ),
        ))

    @output
    @render.ui
    def save_csv_status():
        level, message = save_csv_status_value.get()
        if not message:
            return ui.tags.div()
        color = "#b03a2e" if level == "problem" else "#1d4f9e"
        return ui.tags.p(message,
                         style=f"font-size: 12px; color: {color}; margin: 6px 0 0 0; "
                               f"word-break: break-all;")

    # ── Feedback capture. All four producers write under the active run id to
    #    data/CONTAINS_PHI/expert_label_corrections/<run_id>/, and each one asks
    #    _why_feedback_cannot_be_saved first, so the only run id they ever write under is
    #    a real pipeline run's. Anything else — no run selected, or a run with no
    #    readable output — is refused with that reason in the status line. ──
    fb_status_value = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.fb_submit)
    def _on_fb_submit():
        view = cur()
        refusal = _why_feedback_cannot_be_saved(view)
        if refusal:
            fb_status_value.set(f"Nothing was saved. {refusal}")
            return
        variable = (input.fb_variable() or "").strip()
        correct = (input.fb_correct() or "").strip()
        if not variable or not correct:
            fb_status_value.set("Pick a variable and enter the correct value.")
            return
        row = view.rows_by_variable.get(variable)
        if row is None:
            # Without this, the correction records original_value=None — a claim that
            # the run produced nothing here, when in fact it produced no row at all.
            fb_status_value.set(
                f"Nothing was saved. {_no_row_for_variable_reason(variable, view)}"
            )
            return
        path = write_correction(
            patient_id=view.patient_id,
            variable=variable,
            correct_value=correct,
            original_value=row.value,
            run_id=view.run_id,
            model=row.model,
            evidence_chunk_id=row.evidence_source,
            dr=ACTIVE_DR,
        )
        ui.update_text("fb_correct", value="")
        fb_status_value.set(f"Saved correction for {variable} → {path.name}")

    @reactive.Effect
    @reactive.event(input.fb_confirm)
    def _on_fb_confirm():
        # Reviewer reviewed the value and AGREED (the confirm-correct signal).
        view = cur()
        refusal = _why_feedback_cannot_be_saved(view)
        if refusal:
            fb_status_value.set(f"Nothing was saved. {refusal}")
            return
        variable = (input.fb_variable() or "").strip()
        if not variable:
            fb_status_value.set("Pick a variable to confirm.")
            return
        row = view.rows_by_variable.get(variable)
        if row is None:
            # Without this, the confirmation records agreed=true against
            # reviewed_value=None — agreement with a value nobody ever saw.
            fb_status_value.set(
                f"Nothing was saved. {_no_row_for_variable_reason(variable, view)}"
            )
            return
        path = write_confirmation(
            patient_id=view.patient_id,
            variable=variable,
            reviewed_value=row.value,
            run_id=view.run_id,
            dr=ACTIVE_DR,
        )
        fb_status_value.set(f"Confirmed {variable} as correct → {path.name}")

    @reactive.Effect
    @reactive.event(input.fb_draw)
    def _on_fb_draw():
        # Persist the sampling frame -- which (patient, variable) were drawn for
        # review and by what rule -- the random draw the denominator requires.
        view = cur()
        refusal = _why_feedback_cannot_be_saved(view)
        if refusal:
            fb_status_value.set(f"Nothing was saved. {refusal}")
            return
        drawn = [{"patient_id": view.patient_id, "variable": r.variable} for r in view.rows]
        try:
            path = write_sampling_frame(
                drawn=drawn,
                rule="shiny_review_all_extracted_variables",
                run_id=view.run_id,
                dr=ACTIVE_DR,
            )
            # One click adds one patient; the frame accumulates the whole cohort. The
            # number that matters is the run-wide total, because that is the denominator
            # every accuracy figure from this review is computed against — reporting the
            # click's own count told a forty-patient review it had drawn two.
            frame_total = read_sampling_frame(path)["n_drawn"]
        except RuntimeError as sample_not_recorded:
            fb_status_value.set(f"Nothing was saved. {sample_not_recorded}")
            return
        fb_status_value.set(
            f"Added {len(drawn)} (patient, variable) pairs for patient {view.patient_id}. "
            f"Run {view.run_id}'s review sample now holds {frame_total} pairs in total — "
            f"that total is the denominator for this run's accuracy figures → {path.name}"
        )

    # ── Chunk-relevance toggle. A fixed pool of slot handlers maps each
    #    click to the selected variable's i-th evidence chunk, so the dynamically
    #    rendered buttons need no per-render handler registration. ──
    cr_status_value = reactive.Value("")
    # (run, patient, variable, chunk_id) -> last judgment this session. Drives the panel
    # re-render so a judged chunk shows its state and the reviewer doesn't
    # blind-double-click. Keyed on the run and patient too, because a chunk id is minted
    # from the patient and source row and is therefore byte-identical across two runs of
    # the same patient: keyed on the variable alone, a "✓ marked relevant" earned in one
    # run followed the reviewer into another where nothing had been recorded, and the
    # chunk they still needed to judge looked already done.
    cr_judgments: reactive.Value[dict[tuple[str | None, str, str, str], str]] = (
        reactive.Value({})
    )

    @output
    @render.ui
    def chunk_relevance_panel():
        variable = (input.fb_variable() or "").strip()
        judged = cr_judgments.get()
        view = cur()
        row = view.rows_by_variable.get(variable)
        if not row or not row.evidence:
            return ui.tags.p(
                "No evidence chunks recorded for this variable.",
                style="color: #6b6b6b; font-size: 13px;",
            )
        cards = []
        for i, ev in enumerate(row.evidence[:MAX_EVIDENCE_SLOTS]):
            shown = ev.text or f"({ev.text_unavailable_reason or 'evidence text could not be resolved'})"
            if len(shown) > 600:
                shown = shown[:600] + "…"
            if ev.chart_changed_next_step:
                note = (
                    ' <span style="color:#b03a2e; font-weight:600;">· the chart this points '
                    "at changed after the pipeline recorded it — this run's evidence cannot "
                    f"be trusted; {html.escape(ev.chart_changed_next_step)}</span>"
                )
            elif ev.text_unavailable_reason:
                note = (
                    f' <span style="color:#b03a2e;">· {html.escape(ev.text_unavailable_reason)}'
                    "</span>"
                )
            else:
                note = ""
            verdict = judged.get((view.run_id, view.patient_id, variable, ev.chunk_id))
            badge = ""
            if verdict:
                color = "#1e7e34" if verdict == "relevant" else "#b03a2e"
                badge = f' <span style="color:{color}; font-weight:600;">✓ marked {verdict}</span>'
            cards.append(
                ui.tags.div(
                    ui.HTML(
                        f'<div class="evidence-header"><span>'
                        f'<span class="evidence-rank">chunk {i + 1}</span>'
                        f'<code style="font-size:11px;">{html.escape(ev.chunk_id)}</code>{note}{badge}'
                        f'</span></div>'
                        f'<div class="evidence-text">{html.escape(shown)}</div>'
                    ),
                    ui.tags.div(
                        ui.input_action_button(f"cr_yes_{i}", "Relevant", class_="btn-success"),
                        ui.input_action_button(f"cr_no_{i}", "Not relevant"),
                        style="display:flex; gap:8px; margin-top:8px;",
                    ),
                    class_="evidence-card",
                )
            )
        if len(row.evidence) > MAX_EVIDENCE_SLOTS:
            cards.append(ui.tags.p(
                f"Showing {MAX_EVIDENCE_SLOTS} of {len(row.evidence)} evidence chunks; "
                f"{len(row.evidence) - MAX_EVIDENCE_SLOTS} not reviewable here.",
                style="color: #b03a2e; font-size: 12px; margin-top: 6px;",
            ))
        return ui.tags.div(*cards)

    def _make_relevance_handler(slot: int, relevant: bool):
        trigger = getattr(input, f"cr_yes_{slot}") if relevant else getattr(input, f"cr_no_{slot}")

        @reactive.Effect
        @reactive.event(trigger)
        def _handler():
            view = cur()
            refusal = _why_feedback_cannot_be_saved(view)
            if refusal:
                cr_status_value.set(f"Nothing was saved. {refusal}")
                return
            variable = (input.fb_variable() or "").strip()
            row = view.rows_by_variable.get(variable)
            if row is None:
                cr_status_value.set(
                    f"Nothing was saved. {_no_row_for_variable_reason(variable, view)}"
                )
                return
            if slot >= len(row.evidence):
                # The panel re-rendered under the click, so this slot no longer points
                # at one of the selected variable's chunks.
                cr_status_value.set(
                    f"Nothing was saved: evidence chunk {slot + 1} is not one of "
                    f"{variable}'s {len(row.evidence)} chunks. Click the card you meant."
                )
                return
            ev = row.evidence[slot]
            path = write_chunk_relevance(
                patient_id=view.patient_id,
                variable=variable,
                chunk_id=ev.chunk_id,
                relevant=relevant,
                run_id=view.run_id,
                dr=ACTIVE_DR,
            )
            verb = "relevant" if relevant else "not relevant"
            # Twin write: the PHI-side entry above carries the raw chunk id for
            # training-text hydration; the NO_PHI exhaust record carries its surrogate
            # so the label joins the run's selection_judgment ranks. A write-gate
            # failure is surfaced in the status line, never dropped.
            exhaust_note = ""
            try:
                emit_relevance_label_exhaust(
                    patient_id=view.patient_id,
                    variable=variable,
                    chunk_id=ev.chunk_id,
                    relevant=relevant,
                    run_id=view.run_id,
                    dr=ACTIVE_DR,
                )
                exhaust_note = " + NO_PHI relevance_label"
            except Exception:
                # The PHI-side judgement is already written; only its shareable twin
                # failed, so this label is absent from the run's exhaust unless it is
                # minted again. Said as a consequence and a remedy — the exception's
                # class name told the reviewer nothing they could act on, under a
                # leading clause that read as success.
                exhaust_note = (
                    " — but only on this machine's side: the shareable copy was not "
                    "written, so this judgement will not reach model training. "
                    "Re-run `junior collect-feedback` for this run to mint it."
                )
            cr_judgments.set({**cr_judgments.get(),
                              (view.run_id, view.patient_id, variable, ev.chunk_id): verb})
            cr_status_value.set(
                f"Marked chunk {slot + 1} ({ev.chunk_id}) {verb} for {variable} → {path.name}"
                f"{exhaust_note}"
            )
        return _handler

    for _slot in range(MAX_EVIDENCE_SLOTS):
        _make_relevance_handler(_slot, True)
        _make_relevance_handler(_slot, False)

    @output
    @render.ui
    def cr_status():
        msg = cr_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        return ui.tags.p(msg, style="font-size: 12px; color: #1d4f9e; margin-top: 8px;")

    @output
    @render.ui
    def fb_status():
        msg = fb_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        return ui.tags.p(msg, style="font-size: 12px; color: #1d4f9e; margin-top: 8px;")

    # ── Workbench tab: read-only inspection of the selected run ──

    def _wb_refusal(view: ActiveView) -> str:
        """Why the workbench has nothing to inspect from this view. Same three states
        as the review screen; only the verbs differ (inspect, not judge)."""
        if view.is_real_run:
            return ""
        if view.nothing_selected:
            return f"No run is selected, so there is nothing to inspect. {FIRST_RUN_HINT}"
        return (
            f"{view.source_label} — there is no extract output to inspect. "
            "Pick another run or patient in the sidebar."
        )

    def _wb_variable() -> str:
        return (input.wb_variable() or "").strip()

    @output
    @render.ui
    def wb_summary_panel():
        view = cur()
        refusal = _wb_refusal(view)
        if refusal:
            return ui.HTML(f'<div class="readiness">{html.escape(refusal)}</div>')
        summary = run_inspection.read_run_summary(view.run_id, ACTIVE_DR)
        if summary is None:
            return ui.HTML(
                '<div class="extract-meta">'
                f'<span><b>Run:</b> {html.escape(view.run_id or "")}</span>'
                '<span>Not summarized yet — a run is summarized when extract closes '
                'it out, or by `junior summarize`.</span></div>'
            )
        per_step = summary.get("per_step_completed") or {}
        steps = ", ".join(f"{name} {count}" for name, count in per_step.items()) or "—"
        return ui.HTML(
            '<div class="extract-meta">'
            f'<span><b>Run:</b> {html.escape(view.run_id or "")}</span>'
            f'<span><b>Status:</b> {html.escape(str(summary.get("status", "?")))}</span>'
            f'<span><b>Steps completed:</b> {html.escape(steps)}</span>'
            f'<span><b>Extraction failures:</b> '
            f'{html.escape(str(summary.get("n_extraction_failed", 0)))}</span>'
            '</div>'
        )

    @output
    @render.ui
    def wb_recipe_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        variable = _wb_variable()
        if not variable:
            return ui.tags.p("Pick a variable above.", style="color: #6b6b6b; font-size: 13px;")
        recipe = run_inspection.read_recipe_text(view.run_id, variable, ACTIVE_DR)
        if recipe is None:
            return ui.tags.p(
                f"No recipe found for {variable} — not in this run's sealed bundle, "
                "not in the working tree.",
                style="color: #b03a2e; font-size: 13px;",
            )
        return ui.HTML(
            '<div class="inspect-block">'
            f'<h6>{html.escape(recipe.path)} — read from {html.escape(recipe.source)}</h6>'
            f'<pre>{html.escape(recipe.text)}</pre></div>'
        )

    def _wb_step_blocks(items, title_of, body_of):
        blocks = [
            ui.HTML(
                '<div class="inspect-block">'
                f'<h6>{html.escape(title_of(item))}</h6>'
                f'<pre>{html.escape(body_of(item))}</pre></div>'
            )
            for item in items
        ]
        return ui.tags.div(*blocks)

    @output
    @render.ui
    def wb_evidence_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        variable = _wb_variable()
        bundles = run_inspection.read_prepared_evidence(
            view.run_id, view.patient_id, variable, ACTIVE_DR,
        ) if variable else []
        if not bundles:
            return ui.tags.p(
                "No prepared evidence on disk for this variable — a table-only or "
                "no-LLM recipe step assembles none.",
                style="color: #6b6b6b; font-size: 13px;",
            )
        return _wb_step_blocks(
            bundles, lambda b: f"step {b.step_id} — formatted_evidence.txt", lambda b: b.text,
        )

    @output
    @render.ui
    def wb_selection_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        variable = _wb_variable()
        selections = run_inspection.read_evidence_selection(
            view.run_id, view.patient_id, variable, ACTIVE_DR,
        ) if variable else []
        if not selections:
            return ui.tags.p(
                "No evidence-selection metadata on disk for this variable.",
                style="color: #6b6b6b; font-size: 13px;",
            )

        def body(s: run_inspection.SelectionSummary) -> str:
            by_type = ", ".join(f"{k}: {v}" for k, v in s.tokens_by_doc_type.items()) or "—"
            return (
                f"blocks: {s.block_count}\n"
                f"evidence tokens (est.): {s.evidence_tokens} / ceiling {s.max_context_tokens}\n"
                f"tokens by doc_type: {by_type}"
            )

        return _wb_step_blocks(selections, lambda s: f"step {s.step_id}", body)

    @output
    @render.ui
    def wb_exchange_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        variable = _wb_variable()
        exchanges = run_inspection.read_llm_exchanges(
            view.run_id, view.patient_id, variable, ACTIVE_DR,
        ) if variable else []
        if not exchanges:
            return ui.tags.p(
                "No step receipts on disk for this variable.",
                style="color: #6b6b6b; font-size: 13px;",
            )
        return _wb_step_blocks(
            exchanges,
            lambda e: f"step {e.step_id} — messages sent, then the raw response",
            lambda e: f"{e.messages}\n\n─── response ───\n{e.response}",
        )

    @output
    @render.ui
    def wb_invariants_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        variable = _wb_variable()
        reports = run_inspection.read_invariants(
            view.run_id, view.patient_id, variable, ACTIVE_DR,
        ) if variable else []
        if not reports:
            return ui.tags.p(
                "No validation verdicts on disk for this selection.",
                style="color: #6b6b6b; font-size: 13px;",
            )
        return _wb_step_blocks(reports, lambda r: r.name, lambda r: r.text)

    @output
    @render.ui
    def wb_files_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        files = run_inspection.list_patient_files(view.run_id, view.patient_id, ACTIVE_DR)
        if not files:
            return ui.tags.p("No files on disk for this patient.",
                             style="color: #6b6b6b; font-size: 13px;")
        listing = "\n".join(f"{f.rel_path}  ({f.size_bytes:,} bytes)" for f in files)
        return ui.HTML(f'<div class="inspect-block"><pre>{html.escape(listing)}</pre></div>')

    @output
    @render.ui
    def wb_exhaust_panel():
        view = cur()
        if _wb_refusal(view):
            return ui.tags.p("—", style="color: #6b6b6b; font-size: 13px;")
        manifest = run_inspection.read_exhaust_manifest(view.run_id, ACTIVE_DR)
        if manifest is None:
            return ui.tags.p(
                "No finalized exhaust for this run yet — it is finalized when the run "
                "closes out, or by `junior summarize`.",
                style="color: #6b6b6b; font-size: 13px;",
            )
        rows = "\n".join(
            f"{name}: {n_rows} rows ({n_failed} failed validation)"
            for name, n_rows, n_failed in manifest.record_types
        ) or "no record types"
        header = (
            f"schema {manifest.schema_version} · vocab {manifest.vocab_version} · "
            f"surrogates {manifest.surrogate_version} · secret {manifest.secret_fingerprint}"
        )
        return ui.HTML(
            '<div class="inspect-block">'
            f'<h6>{html.escape(header)}</h6>'
            f'<pre>{html.escape(rows)}</pre></div>'
        )

    def _wb_values_filename(shape: str) -> str:
        view = cur()
        return f"junior_run_values_{shape}_{view.run_id or 'NO_RUN_selected'}.csv"

    wb_values_status_value: reactive.Value = reactive.Value(("", ""))   # (level, message)

    def _save_run_values(shape: str) -> None:
        view = cur()
        if not view.is_real_run:
            wb_values_status_value.set(
                ("problem", "No run selected, so there are no values to save.")
            )
            return
        wb_values_status_value.set(_save_beside_the_run(
            view, _wb_values_filename(shape),
            run_inspection.run_values_csv(view.run_id, shape=shape, dr=ACTIVE_DR),
        ))

    @reactive.Effect
    @reactive.event(input.wb_values_wide)
    def _on_wb_values_wide():
        _save_run_values("wide")

    @reactive.Effect
    @reactive.event(input.wb_values_long)
    def _on_wb_values_long():
        _save_run_values("long")

    @output
    @render.ui
    def wb_values_status():
        level, message = wb_values_status_value.get()
        if not message:
            return ui.tags.div()
        color = "#b03a2e" if level == "problem" else "#1d4f9e"
        return ui.tags.p(message,
                         style=f"font-size: 12px; color: {color}; margin: 6px 0 0 0; "
                               f"word-break: break-all;")

    wb_export_status_value = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.wb_export)
    def _on_wb_export():
        view = cur()
        refusal = _wb_refusal(view)
        if refusal:
            wb_export_status_value.set(f"Nothing was exported. {refusal}")
            return
        try:
            bundle = run_inspection.export_shareable_zip(view.run_id, ACTIVE_DR)
        except Exception as not_exported:  # noqa: BLE001 — reported, never swallowed
            wb_export_status_value.set(f"Nothing was exported. {not_exported}")
            return
        wb_export_status_value.set(
            f"Wrote {bundle} — run-level metadata only, no patient data; every file in "
            "it was scanned before it was written."
        )

    @output
    @render.ui
    def wb_export_status():
        msg = wb_export_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        return ui.tags.p(msg, style="font-size: 12px; color: #1d4f9e; margin-top: 8px;")

    # ── Recipes tab: authoring and editing, gated by the pipeline's own loader ──
    rc_versions = reactive.Value(recipe_editing.list_recipe_versions())
    rc_save_status_value = reactive.Value("")
    rc_create_status_value = reactive.Value("")

    def _rc_choices() -> dict[str, str]:
        return {str(v.version_dir): v.label for v in rc_versions.get()}

    def _rc_selected_dir():
        chosen = input.rc_recipe() or ""
        return Path(chosen) if chosen else None

    def _rc_refresh_versions(select: str | None = None) -> None:
        rc_versions.set(recipe_editing.list_recipe_versions())
        choices = _rc_choices()
        selected = select if select in choices else (next(iter(choices), None))
        ui.update_select("rc_recipe", choices=choices, selected=selected)
        ui.update_select("rc_new_template", choices=sorted(
            {v.variable for v in rc_versions.get() if not v.is_draft}
        ))

    @reactive.Effect
    def _rc_boot():
        # Once, at session start: seed the pickers the UI declared empty.
        with reactive.isolate():
            _rc_refresh_versions()

    @reactive.Effect
    @reactive.event(input.rc_refresh)
    def _rc_on_refresh():
        _rc_refresh_versions(select=input.rc_recipe() or None)

    @reactive.Effect
    @reactive.event(input.rc_recipe)
    def _rc_on_recipe_change():
        version_dir = _rc_selected_dir()
        files = recipe_editing.recipe_files(version_dir) if version_dir else []
        choices = {f.rel_path: f"{f.rel_path}  ·  {f.kind}" for f in files}
        first = next(iter(choices), None)
        ui.update_select("rc_file", choices=choices, selected=first)
        if version_dir and first:
            ui.update_text_area("rc_editor",
                                value=recipe_editing.read_file(version_dir, first))
        rc_save_status_value.set("")

    @reactive.Effect
    @reactive.event(input.rc_file)
    def _rc_on_file_change():
        version_dir = _rc_selected_dir()
        rel = input.rc_file() or ""
        if version_dir and rel:
            try:
                ui.update_text_area("rc_editor",
                                    value=recipe_editing.read_file(version_dir, rel))
            except (OSError, ValueError) as unreadable:
                rc_save_status_value.set(f"Cannot open {rel}: {unreadable}")

    @reactive.Effect
    @reactive.event(input.rc_save)
    def _rc_on_save():
        version_dir = _rc_selected_dir()
        rel = input.rc_file() or ""
        if not version_dir or not rel:
            rc_save_status_value.set("Pick a recipe and a file first.")
            return
        try:
            refusal = recipe_editing.save_file(version_dir, rel, input.rc_editor() or "")
        except (OSError, ValueError) as failed:
            rc_save_status_value.set(f"Nothing was saved. {failed}")
            return
        if refusal:
            # The edit is already rolled back; the words on screen are the operator's
            # to fix, so the editor keeps them.
            rc_save_status_value.set(
                f"Not saved — the recipe loader refused this edit and the file on "
                f"disk is unchanged: {refusal}"
            )
            return
        rc_save_status_value.set(
            f"Saved {rel}. Runs sealed with the old content now refuse to continue "
            "extraction — take the change to a fresh run: junior extract --new-run."
        )

    @reactive.Effect
    @reactive.event(input.rc_finish)
    def _rc_on_finish():
        version_dir = _rc_selected_dir()
        if not version_dir:
            rc_save_status_value.set("Pick a recipe first.")
            return
        refusal = recipe_editing.finish_draft(version_dir)
        if refusal:
            rc_save_status_value.set(f"Still a draft — {refusal}")
            return
        _rc_refresh_versions(select=str(version_dir))
        rc_save_status_value.set(
            "Draft finished — the loader accepts it, so it can run now. Add it to "
            "`recipes:` in your project settings, or tick it on the Start tab."
        )

    @reactive.Effect
    @reactive.event(input.rc_create)
    def _rc_on_create():
        name = (input.rc_new_name() or "").strip()
        template = (input.rc_new_template() or "").strip()
        if not name or not template:
            rc_create_status_value.set("Name the new variable and pick a template.")
            return
        try:
            destination = recipe_editing.scaffold_new_recipe(name, template)
        except ValueError as refused:
            rc_create_status_value.set(f"Nothing was created. {refused}")
            return
        _rc_refresh_versions(select=str(destination))
        ui.update_text("rc_new_name", value="")
        rc_create_status_value.set(
            f"Created {name} from {template} — a draft, on purpose: the wiring is "
            "done, the words are not. Edit the prompts and schema above, then press "
            "Finish draft."
        )

    @reactive.Effect
    @reactive.event(input.rc_new_version)
    def _rc_on_new_version():
        version_dir = _rc_selected_dir()
        if not version_dir:
            rc_save_status_value.set("Pick a recipe first.")
            return
        try:
            destination = recipe_editing.scaffold_new_version(version_dir.parent.name)
        except ValueError as refused:
            rc_save_status_value.set(f"Nothing was created. {refused}")
            return
        _rc_refresh_versions(select=str(destination))
        rc_save_status_value.set(
            f"Created {destination.parent.name}/{destination.name} from "
            f"{version_dir.name} — sealed runs keep reading {version_dir.name}; new "
            "runs resolve the highest version."
        )

    @output
    @render.ui
    def rc_draft_notice():
        version_dir = _rc_selected_dir()
        if not version_dir or not recipe_editing.is_draft(version_dir):
            return ui.tags.span()
        return ui.HTML(
            '<div class="readiness"><b>This recipe is a draft.</b> Its prompts and '
            'schema still describe the variable it was copied from, and the pipeline '
            'refuses to run it. Edit the files below, then press Finish draft.</div>'
        )

    @output
    @render.ui
    def rc_save_status():
        msg = rc_save_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        color = "#b03a2e" if ("Not saved" in msg or "Nothing was" in msg
                              or "Still a draft" in msg) else "#1d4f9e"
        return ui.tags.p(msg, style=f"font-size: 12px; color: {color}; margin-top: 8px;")

    @output
    @render.ui
    def rc_create_status():
        msg = rc_create_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        color = "#b03a2e" if "Nothing was" in msg else "#1d4f9e"
        return ui.tags.p(msg, style=f"font-size: 12px; color: {color}; margin-top: 8px;")

    # ── Start tab, steps 1–2: scan an input folder, select patients, preflight + ingest ──
    # The app opens ON a scanned cohort, patients already ticked, rather than on an
    # empty picker under a folder box that is already filled in. Scanning the folder
    # the app just told you it is using is not a decision anybody makes differently —
    # it is a click between the operator and every button that needs a patient, and
    # the buttons below all answer "Scan a folder first" until it happens. Retyping
    # the folder and pressing Scan still does exactly what it did.
    opening_scan = cohort_ingest.scan_cohort(COHORT_INPUT_DEFAULT)
    cohort_scan_value: reactive.Value = reactive.Value(opening_scan)   # cohort_ingest.ScanResult | None
    cohort_status_value = reactive.Value("")
    # Step 1 answers under its own button, not at the foot of the tab. A scan that
    # finds nothing renders an EMPTY picker, so its only reply was a line roughly a
    # screen further down, past steps 2 and 3 — and a reply nobody can see reads as a
    # button that does nothing. Reported exactly that way.
    cohort_scan_status_value: reactive.Value = reactive.Value(_scan_message(opening_scan))
    cohort_results_value: reactive.Value = reactive.Value(None)   # ("preflight", rows) | ("ingest", outcome) | None

    def _selected_patients() -> list[str]:
        """Currently ticked patient ids; empty before the picker has rendered."""
        try:
            return list(input.cohort_patients() or [])
        except Exception:
            return []

    def _selected_variables() -> list[str]:
        """Currently ticked variable names; empty when no recipes were found (the
        checkbox group is then not in the UI at all)."""
        try:
            return list(input.run_variables() or [])
        except Exception:
            return []

    @reactive.Effect
    @reactive.event(input.cohort_scan)
    def _on_cohort_scan():
        folder = (input.cohort_folder() or "").strip() or COHORT_INPUT_DEFAULT
        result = cohort_ingest.scan_cohort(folder)
        cohort_scan_value.set(result)
        cohort_results_value.set(None)
        cohort_scan_status_value.set(_scan_message(result))

    @output
    @render.ui
    def cohort_picker():
        result = cohort_scan_value.get()
        if result is None or not result.patients:
            return ui.tags.div()
        choices = {
            p.patient_id: f"{p.patient_id}  ·  {p.file_count} file{'s' if p.file_count != 1 else ''}"
            for p in result.patients
        }
        extras = []
        if result.empty_dirs:
            shown = ", ".join(result.empty_dirs[:6])
            more = f" (+{len(result.empty_dirs) - 6} more)" if len(result.empty_dirs) > 6 else ""
            extras.append(ui.tags.p(
                f"{len(result.empty_dirs)} subfolder(s) had no source files at this level "
                f"and were skipped: {shown}{more}. Point deeper to reach their patients.",
                style="font-size: 11.5px; color: #8a8a8a; margin-top: 8px;",
            ))
        return ui.tags.div(
            ui.tags.div(
                ui.input_action_link("cohort_all", "Select all"),
                ui.tags.span(" · ", style="color: #ccc;"),
                ui.input_action_link("cohort_none", "Clear"),
                style="font-size: 12px; margin: 8px 0 2px 0;",
            ),
            ui.input_checkbox_group(
                "cohort_patients", None, choices=choices, selected=list(choices),
            ),
            *extras,
        )

    @reactive.Effect
    @reactive.event(input.cohort_all)
    def _on_cohort_all():
        result = cohort_scan_value.get()
        if result and result.patients:
            ui.update_checkbox_group(
                "cohort_patients", selected=[p.patient_id for p in result.patients]
            )

    @reactive.Effect
    @reactive.event(input.cohort_none)
    def _on_cohort_none():
        ui.update_checkbox_group("cohort_patients", selected=[])

    @reactive.Effect
    @reactive.event(input.vars_all)
    def _on_vars_all():
        ui.update_checkbox_group("run_variables", selected=AVAILABLE_VARIABLES)

    @reactive.Effect
    @reactive.event(input.vars_none)
    def _on_vars_none():
        ui.update_checkbox_group("run_variables", selected=[])

    @output
    @render.ui
    def readiness_panel():
        if not READINESS_PROBLEMS:
            return ui.tags.div()
        return ui.tags.div(
            ui.tags.b("A full run can't complete in this checkout yet:"),
            ui.tags.ul(*[ui.tags.li(p) for p in READINESS_PROBLEMS]),
            ui.tags.p(
                "Preflight and ingest-only still work — they need no model.",
                style="margin: 8px 0 0 0;",
            ),
            class_="readiness",
        )

    @reactive.Effect
    @reactive.event(input.cohort_preflight)
    def _on_cohort_preflight():
        result = cohort_scan_value.get()
        selected = _selected_patients()
        if result is None or not result.patients:
            cohort_status_value.set("Scan a folder first.")
            return
        if not selected:
            cohort_status_value.set("Select at least one patient.")
            return
        rows = cohort_ingest.preflight_cohort(result.folder, selected)
        cohort_results_value.set(("preflight", rows))
        ok = sum(1 for r in rows if r.ok)
        cohort_status_value.set(f"Preflight: {ok} OK, {len(rows) - ok} blocked.")

    @reactive.Effect
    @reactive.event(input.cohort_ingest)
    def _on_cohort_ingest():
        result = cohort_scan_value.get()
        selected = _selected_patients()
        if result is None or not result.patients:
            cohort_status_value.set("Scan a folder first.")
            return
        if not selected:
            cohort_status_value.set("Select at least one patient.")
            return
        outcome = cohort_ingest.ingest_cohort(result.folder, selected)
        cohort_results_value.set(("ingest", outcome))
        if outcome.error:
            cohort_status_value.set(outcome.error)
            return
        ok = sum(1 for r in outcome.rows if r.ok)
        cohort_status_value.set(
            f"Ingest {outcome.run_id}: {ok}/{len(outcome.rows)} ok — parquets written to "
            "structured/ in each patient's run folder. That was Step 1 only: no variables "
            "were extracted. Use ▶ Run pipeline for the full pipeline."
        )

    # ── Full pipeline run (ingest → … → extract). The work happens in a subprocess
    #    (pipeline_launcher); here we only start it, poll it, and adopt the finished
    #    run into the review picker. ──
    active_run: reactive.Value = reactive.Value(None)   # pipeline_launcher.PipelineRun | None
    run_message = reactive.Value("")
    adopted_run_id = reactive.Value("")
    # The run the stage buttons are working in. Ingest starts one; embed, index and
    # extract continue it by name. Without that they each start a fresh run of their
    # own — deliberate CLI behaviour on the bundled settings, where continuing whatever
    # is newest on disk could append to a cohort nobody asked about — and Extract then
    # extracted from an empty run it had just minted and called it finished.
    #
    # Seeded from the project's newest run, not left empty, because this value used to
    # live only for as long as the browser tab did: ingest a cohort, reload the page,
    # and Index refused with "run 1 Ingest first" while the ingested run sat on disk.
    # The caution the CLI applies to "newest on disk" is about a stray command in an
    # unknown directory; the app was pinned to this project when it was launched.
    stage_run_id = reactive.Value(_newest_run_here())

    @reactive.Effect
    @reactive.event(input.run_pipeline)
    def _on_run_pipeline():
        current = active_run.get()
        if current is not None and current.is_running:
            run_message.set("A run is already going — stop it first.")
            return
        scan = cohort_scan_value.get()
        patients = _selected_patients()
        variables = _selected_variables()
        if scan is None or not scan.patients:
            run_message.set("Scan a folder first (step 1).")
            return
        if not patients:
            run_message.set("Select at least one patient (step 1).")
            return
        if not variables:
            run_message.set("Select at least one variable (step 2).")
            return
        run_message.set("")
        active_run.set(pipeline_launcher.start_run(
            scan.folder, patients, variables, run_results.DATA_ROOT,
        ))

    def _start_stage(stage: str, new_run: bool = False) -> None:
        """Everything a stage button needs before it may spawn its command. The same
        guards the Run button uses, minus the variable ticks — a stage command has no
        --variable, so requiring one would be the app inventing a rule the CLI has not."""
        current = active_run.get()
        if current is not None and current.is_running:
            run_message.set("A run is already going — stop it first.")
            return
        scan = cohort_scan_value.get()
        patients = _selected_patients()
        if scan is None or not scan.patients:
            run_message.set("Scan a folder first (step 1).")
            return
        if not patients:
            run_message.set("Select at least one patient (step 1).")
            return
        variables = _selected_variables()
        if stage == "extract" and not variables:
            run_message.set("Select at least one variable (step 2).")
            return
        # Ingest opens a run; the stages after it continue that one by name. Starting
        # over is the CLI's --new-run, not an empty run id: inside a project a bare
        # `junior ingest` continues the newest run, which is the opposite of starting
        # over — and the app is always inside a project, because `junior workbench`
        # pinned one when it launched.
        carry_on_with = "" if (new_run or stage == "ingest") else stage_run_id.get()
        if stage != "ingest" and not carry_on_with:
            run_message.set(
                f"No run to {stage} yet — run 1 Ingest first, or use ▶ Run pipeline."
            )
            return
        run_message.set("")
        active_run.set(pipeline_launcher.start_stage(
            stage, patients, [p.patient_id for p in scan.patients], run_results.DATA_ROOT,
            run_id=carry_on_with, variables=variables, new_run=new_run,
        ))

    # One effect per stage. A loop needs the stage bound per iteration, or all four
    # buttons close over whichever name the loop finished on.
    def _bind_stage_button(stage: str) -> None:
        @reactive.Effect
        @reactive.event(input[f"run_{stage}"])
        def _on_stage():
            _start_stage(stage)

    for _stage in pipeline_launcher.STAGES:
        _bind_stage_button(_stage)

    @reactive.Effect
    @reactive.event(input.run_new)
    def _on_new_run():
        """Start over: a fresh run with the charts read into it, which is what the CLI
        itself points at when a run needs redoing (`junior ingest --new-run`)."""
        _start_stage("ingest", new_run=True)

    @reactive.Effect
    @reactive.event(input.run_stop)
    def _on_run_stop():
        run = active_run.get()
        if run is None or not run.is_running:
            run_message.set("No run in flight.")
            return
        run.stop()
        run_message.set("Stop requested.")

    # Poll while the child is alive; once it has finished successfully, point the
    # review picker at the run it produced and switch to the Review tab — the whole
    # reason the operator pressed Run.
    @reactive.Effect
    def _adopt_finished_run():
        run = active_run.get()
        if run is None:
            return
        if run.is_running:
            reactive.invalidate_later(2)
            return
        # Whatever it was, the next stage button continues the run it used.
        if run.run_id:
            with reactive.isolate():
                if stage_run_id.get() != run.run_id:
                    stage_run_id.set(run.run_id)
        if not run.succeeded or not run.run_id or not run.produces_values:
            return
        with reactive.isolate():
            if adopted_run_id.get() == run.run_id:
                return
            adopted_run_id.set(run.run_id)
        patients = run_results.list_patients_with_extract(run.run_id)
        if not patients:
            run_message.set(
                f"Run {run.run_id} finished but produced no extract output — see the log."
            )
            return
        runs = run_results.list_reviewable_runs()
        sel_run.set(run.run_id)
        sel_patient.set(patients[0])
        ui.update_select("review_run", choices=runs, selected=run.run_id)
        ui.update_select("review_patient", choices=patients, selected=patients[0])
        ui.update_navs("main_tabs", selected="Review")

    @output
    @render.ui
    def stage_run_line():
        run_id = stage_run_id.get()
        words = (f"Stages continue run {run_id}." if run_id
                 else "No run yet — 1 Ingest starts one.")
        return ui.tags.p(words,
                         style="font-size: 12px; color: #6b6b6b; margin: 8px 0 0 0;")

    @output
    @render.ui
    def run_status():
        run = active_run.get()
        message = run_message.get()
        parts = []
        if run is not None:
            if run.is_running:
                reactive.invalidate_later(2)
            color = "#1d4f9e" if (run.is_running or run.succeeded) else "#b03a2e"
            parts.append(ui.tags.p(
                run.status_line(),
                style=f"font-size: 13px; color: {color}; margin: 10px 0 0 0;",
            ))
        if message:
            parts.append(ui.tags.p(
                message, style="font-size: 13px; color: #b03a2e; margin: 6px 0 0 0;"))
        return ui.tags.div(*parts)

    @output
    @render.ui
    def run_log():
        run = active_run.get()
        if run is None:
            return ui.tags.div()
        if run.is_running:
            reactive.invalidate_later(2)
        log_text = "\n".join(run.log())
        return ui.HTML(f'<div class="run-log">{html.escape(log_text)}</div>')

    @output
    @render.ui
    def cohort_scan_status():
        level, message = cohort_scan_status_value.get()
        if not message:
            return ui.tags.div()
        color = "#b03a2e" if level == "problem" else "#1d4f9e"
        return ui.tags.p(message,
                         style=f"font-size: 13px; color: {color}; margin: 10px 0 0 0;")

    @output
    @render.ui
    def cohort_status():
        msg = cohort_status_value.get()
        if not msg:
            return ui.tags.p("", style="font-size: 12px;")
        return ui.tags.p(msg, style="font-size: 13px; color: #1d4f9e; margin-top: 10px;")

    @output
    @render.ui
    def cohort_results():
        payload = cohort_results_value.get()
        if not payload:
            return ui.tags.div()
        kind, data = payload
        if kind == "preflight":
            return _cohort_preflight_html(data)
        return _cohort_ingest_html(data)


def _cohort_preflight_html(rows) -> ui.HTML:
    """Render the preflight rows as an OK/blocked table (reuses the extract-table style)."""
    body = []
    for r in rows:
        status = (
            "<span style='color:#1e7e34;font-weight:600;'>OK</span>" if r.ok
            else "<span style='color:#b03a2e;font-weight:600;'>blocked</span>"
        )
        problems = html.escape("; ".join(r.problems)) if r.problems else "—"
        advisories = html.escape("; ".join(r.advisories)) if r.advisories else "—"
        body.append(
            f"<tr><td class='var-name'>{html.escape(r.patient_id)}</td>"
            f"<td>{status}</td><td class='var-quote'>{problems}</td>"
            f"<td class='var-quote' style='color:#8a6d1f;'>{advisories}</td></tr>"
        )
    return ui.HTML(
        "<table class='extract-table'><thead><tr>"
        "<th>Patient</th><th>Preflight</th><th>Problems</th><th>Heads-up</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _cohort_ingest_html(outcome):
    """Render the ingest outcome: run id + a per-patient status/result table."""
    if outcome.error:
        return ui.tags.p(outcome.error, style="color:#b03a2e; font-size:13px;")
    body = []
    for r in outcome.rows:
        if r.ok:
            state = "cached" if r.cached else "ingested"
            state_html = f"<span style='color:#1e7e34;font-weight:600;'>{state}</span>"
            detail = (f"{r.files_written} file{'s' if r.files_written != 1 else ''}, "
                      f"{r.total_rows} rows")
        else:
            state_html = "<span style='color:#b03a2e;font-weight:600;'>failed</span>"
            detail = html.escape(r.error or "")
        body.append(
            f"<tr><td class='var-name'>{html.escape(r.patient_id)}</td>"
            f"<td>{state_html}</td><td class='var-quote'>{detail}</td></tr>"
        )
    return ui.tags.div(
        ui.tags.p(f"Run {outcome.run_id}",
                  style="font-size:12px; color:#5a6273; margin:8px 0 4px 0;"),
        ui.HTML(
            "<table class='extract-table'><thead><tr>"
            "<th>Patient</th><th>Status</th><th>Result</th>"
            "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
        ),
    )


app = App(app_ui, server)
