"""Single CLI entry point — every ``jr-pipeline`` command is defined here.

``jr-pipeline run --config <c>`` is the normal way in: it performs all three phases
below for a whole cohort in one process. The individual phase commands exist so a
cluster can fan the middle phase out to one job per patient; they are the same
sealed path split apart, not a second way to run.

A run always happens in three ordered phases, the same on a laptop, on Nero/GCP,
or on a SLURM cluster:

    1. Seal — ``jr-pipeline seal --config <c> [--patients-file <f>]``. Freezes a
       snapshot of the code + recipes and writes the run's manifest. The per-patient
       stage commands refuse to run until it exists. ``run`` seals for you; the
       standalone command exists because a cluster run fans out to one job per
       patient that all start at once, and if each job rebuilt the snapshot itself
       they would delete and recopy the same ``code/`` directory on top of each
       other. Sealing once, before the fan-out, avoids that collision.

    2. Stages — ``jr-pipeline ingest|embed|index|... --config <c> --patient <p>``,
       one call per patient (run directly, or one parallel job per patient on a
       cluster). Before doing any work, each call checks that the seal is still
       present, that the code on disk still matches it, and that the config it was
       handed agrees with the sealed config on the values that must be identical
       across the whole run (its id, its output and recipe paths, ...) plus the
       provider/retrieval fingerprints.

    3. Summarize — ``jr-pipeline summarize --run-root <r>``, once every patient is
       done. Merges the logs, writes ``summary.json``, flips the run from
       "running" to finished, and compacts the run's NO_PHI exhaust shards into
       shareable parquet + manifest (``jr-pipeline export-metadata`` ships them).

To add a new stage you touch three small things: a ``_run_<name>`` adapter, one entry
in ``STAGES``, and a module exposing
``run_<name>_one(cfg, patient_id, code_lock_hash, force)``.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import click
import yaml

from apps_and_interfaces.console_questions import (
    BACK_WORDS,
    CANCEL_WORDS,
    Question,
    ask_one,
    ask_sequence,
    describe_controls,
    is_interactive,
)
from jr_pipeline import __version__
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_to_data_cross_link import (
    code_bundle_for,
    verify_code_bundle,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    build_and_seal,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    resolve_repo_root as _resolve_repo_root,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
    build_manifest,
    build_roster,
    model_sha256_from_cfg,
    write_manifest,
    write_roster,
)

# The run-invariant key set and the drift questions live with the other provenance
# code in src (sealed_run_continuity): `junior run` and the engine ask the identical
# questions before re-entering a run, and two copies of the list is how one door
# would drift open. Re-exported under the name this module's tests and guardrails
# import (F401: imported for them, not for this file).
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
    RUN_INVARIANT_CFG_KEYS as _RUN_INVARIANT_CFG_KEYS,  # noqa: F401
)
from jr_pipeline.runtime_infrastructure.config_loading import load_config
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    data_root,
    ensure_layout,
    phi_intermediate_run_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.project_context import (
    CONFIG_ENVIRONMENT_VARIABLE,
    project_config_name,
)

_log = get_logger("cli")


def _bind_data_root(cfg: dict) -> Path:
    """Bind JR_DATA_ROOT from cfg['output_root'] (or fall back to ambient)."""
    # Unconditional assignment, not setdefault: a stale env value from a
    # previous run must not redirect writes to the wrong filesystem.
    import os
    configured = cfg.get("output_root")
    if not configured:
        return data_root().resolve()
    output_root = Path(configured).expanduser().resolve()
    prior = os.environ.get("JR_DATA_ROOT")
    if prior and Path(prior).expanduser().resolve() != output_root:
        # Worth saying — output is not going where the environment implies — but as a
        # sentence, not a structured log record. This lands in the middle of a
        # command's own output, where raw JSON reads as something having gone wrong.
        # Compared after resolving, so a symlinked or trailing-slash spelling of the
        # same directory does not raise a false alarm.
        click.secho(f"  note    writing to {output_root}, not the location left over from "
                    f"an earlier run ({prior})", err=True, dim=True)
    os.environ["JR_DATA_ROOT"] = str(output_root)
    return output_root


def _run_ingest(cfg: dict, patient_id: str, code_lock_hash: str | None, force: bool) -> dict:
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import run_ingest_one

    return run_ingest_one(
        cfg=cfg, patient_id=patient_id, code_lock_hash=code_lock_hash, force=force
    )


def _run_embed(cfg: dict, patient_id: str, code_lock_hash: str | None, force: bool) -> dict:
    from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import run_embed_one

    return run_embed_one(
        cfg=cfg, patient_id=patient_id, code_lock_hash=code_lock_hash, force=force
    )


def _run_index(cfg: dict, patient_id: str, code_lock_hash: str | None, force: bool) -> dict:
    from jr_pipeline.pipeline_steps.step_3_build_vector_index.build_index import run_index_one

    return run_index_one(
        cfg=cfg, patient_id=patient_id, code_lock_hash=code_lock_hash, force=force
    )


def _run_retrieve(cfg: dict, patient_id: str, code_lock_hash: str | None, force: bool) -> dict:
    from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import run_retrieve_one

    return run_retrieve_one(
        cfg=cfg, patient_id=patient_id, code_lock_hash=code_lock_hash, force=force
    )


def _run_extract(cfg: dict, patient_id: str, code_lock_hash: str | None, force: bool) -> dict:
    from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import run_extract_one
    from jr_pipeline.runtime_infrastructure.extraction_progress import report_variables

    # The same display `junior run` uses. Extraction is the one stage where a patient
    # takes minutes, so which variable is being worked on is the only thing that tells
    # a slow run from a stuck one — and it has to be the same display in both commands,
    # or the one nobody watches is the one that goes wrong.
    return run_extract_one(
        cfg=cfg, patient_id=patient_id, code_lock_hash=code_lock_hash, force=force,
        on_variable=report_variables(),
    )


STAGES: dict[str, Callable[..., dict]] = {
    "ingest": _run_ingest,
    "embed": _run_embed,
    "index": _run_index,
    "retrieve": _run_retrieve,
    # Reranking is not a standalone stage -- it runs inside extract (step 7).
    # The extract module + pipeline_steps/step_7_extract_variables/ defaults config +
    # recipes/ are wired and tested locally, but there is deliberately no extract.sh
    # under deployment/Local_SLURM_Cluster/slurm/, so the cluster batch cannot
    # launch extraction by accident.
    "extract": _run_extract,
}

# The two halves of a run, in order, with the one-line description each stage shows in
# help. Setting up a cohort is the expensive part and is done once per patient;
# extraction is cheap and re-run whenever a recipe changes. Splitting them here is what
# lets someone re-extract without rebuilding a corpus.
#
# The single source of the stage order and numbering: the command registration loop,
# `junior --help`, and the interactive prompt all render from this, so a stage cannot
# appear in one place with a number it does not have in another.
PIPELINE_PHASES: dict[str, dict[str, str]] = {
    "Run": {
        "ingest": "read the chart files in",
        "embed": "make the text searchable by meaning",
        "index": "build each patient's search index",
        # Retrieval and reranking happen inside this stage, per recipe, because a
        # recipe chooses its own retriever and query. `retrieve` exists as its own
        # command for probing that subsystem (ADR 0010), not as a step you run.
        #
        # Steps 1-3 build the corpus once; this one is re-run whenever a recipe
        # changes, which is the only distinction between them and not worth a
        # heading of its own.
        "extract": "pull the variables out of the charts",
    },
}

PIPELINE_STAGE_DESCRIPTIONS: dict[str, str] = {
    name: description
    for phase in PIPELINE_PHASES.values()
    for name, description in phase.items()
}

# The rest of the surface, each group labelled by what it is for rather than when you
# reach it. Unnumbered: none of these advance a run, so numbering would imply a
# sequence that is not there.
START_COMMANDS: dict[str, str] = {
    "new-project": "set up a cohort — where the charts are, where output goes",
    "columns": "map your export's column names to Junior's",
}

# Commands that report rather than advance a run. Shown in the footer beside `help`
# and `quit`, because they are things you reach for at any point in the flow rather
# than steps in it.
META_COMMANDS: dict[str, str] = {
    "status": "which config and run you are on, and how far along",
    "project": "which cohort you are working on, and how to switch",
    "recipes": "every recipe you can extract with, and which this project uses",
}

APP_COMMANDS: dict[str, str] = {
    "workbench": "open the browser workbench — run, review, label, inspect",
}

SHARE_COMMANDS: dict[str, str] = {
    "export-metadata": "zip a run's shareable summary — no patient data",
}




def _recipes_that_changed_since_sealing(cfg: dict, code_dir: Path) -> list[str]:
    """Recipes whose live content differs from this run's sealed copy — the shared
    check, under the signature this module's callers and tests already use."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
        recipes_that_changed_since_sealing,
    )

    return recipes_that_changed_since_sealing(cfg.get("recipes_root"), code_dir / "recipes")


def _seal_bundle(
    cfg: dict,
    run_root: Path,
    *,
    step: str,
    config_alias: str,
    target_patients: list[str] | None = None,
) -> str:
    """Write the run's code bundle, manifest, and roster; return the code_lock_hash.

    Shared by the `seal` command and by auto-sealing, so a run started either way is
    equally complete. The manifest matters beyond provenance: `summarize` reads it to
    close a run out, so a bundle without one leaves the run unfinishable."""
    repo_root = _resolve_repo_root()
    recipes_root = _resolve_recipes_root(cfg, repo_root)
    if not recipes_root.is_dir():
        raise click.ClickException(
            f"Cannot find the variable recipes at {recipes_root}.\n"
            "Each variable is a folder there saying how to extract it. Point "
            "`recipes_root:` in your config at that folder."
        )
    sealed = build_and_seal(
        run_id=cfg["run_id"],
        run_root=run_root,
        repo_root=repo_root,
        cfg=cfg,
        entry_point={
            "argv_program": "junior",
            "argv": sys.argv[:],
            "step": step,
            "config_alias": config_alias,
        },
        recipes_root=recipes_root,
        allowlist_path=(
            Path(cfg["allowlist_path"]).resolve() if cfg.get("allowlist_path") else None
        ),
    )
    code_lock_hash = sealed["code_lock_hash"]

    patients = list(target_patients or [])
    write_manifest(run_root, build_manifest(
        run_id=cfg["run_id"],
        code_lock_hash=code_lock_hash,
        entry_point_name="apps_and_interfaces.command_line_interface." + step,
        config_alias=config_alias,
        target_patients=patients,
        model_sha256=model_sha256_from_cfg(cfg),
    ))
    # The raw roster stays PHI-side and is never exported; the manifest holds a count.
    write_roster(run_root, build_roster(run_id=cfg["run_id"], target_patients=patients))
    return code_lock_hash


def _ensure_sealed_bundle(*, cfg: dict, run_root: Path) -> str:
    """Seal this run if it has not been sealed yet, then check it; return code_lock_hash.

    Sealing is not something an operator should have to know about, so the first stage
    command to run a given run_id writes the bundle. It is done under a lock because a
    SLURM array starts every patient's job at once and they would otherwise each
    delete and recopy the same code/ directory on top of one another.

    Only an ABSENT bundle is filled in. If one exists, the drift and content checks
    below still run — auto-sealing must never paper over a config that changed
    underneath a half-finished run."""
    from filelock import FileLock

    code_lock_path = run_root / "code" / "code.lock.json"
    if not code_lock_path.is_file():
        run_root.mkdir(parents=True, exist_ok=True)
        with FileLock(str(run_root / ".seal.lock"), timeout=300):
            # Re-check inside the lock: whoever held it first may have just sealed.
            if not code_lock_path.is_file():
                # No roster: a stage command knows its own patient, not the cohort.
                # `seal --patients-file` is how a cluster run declares one up front.
                _seal_bundle(
                    cfg, run_root, step="auto_seal",
                    config_alias=cfg.get("project") or "run",
                )
    return _require_sealed_bundle(cfg=cfg, run_root=run_root)


def _require_sealed_bundle(*, cfg: dict, run_root: Path) -> str:
    """Refuse to run a step without a matching sealed code bundle; return its code_lock_hash."""
    # We deliberately do NOT compare the full ``config_hash``: each step has its own
    # config shape. The shared continuity checks compare run-invariant keys, scoped
    # sub-hashes (provider/retrieval), live recipe content, and the planned variables
    # — the same questions `junior run` and `junior seal` ask before re-entering a run.
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
        why_a_sealed_run_cannot_continue,
    )

    code_dir = run_root / "code"
    lock_path = code_dir / "code.lock.json"
    if not lock_path.is_file():
        raise click.ClickException(
            f"This run has no saved copy of the code and settings, and one could not be "
            f"written to {code_dir}.\nCheck that folder is writable, then try again."
        )

    hashes_path = code_dir / "hashes.json"
    if not hashes_path.is_file():
        raise click.ClickException(
            f"This run's saved copy of the code is incomplete — {hashes_path.name} is "
            f"missing.\nStart over:  junior ingest --new-run   then  junior embed, index, extract."
        )

    try:
        changed = why_a_sealed_run_cannot_continue(cfg, run_root)
    except RuntimeError as recipes_unhashable:
        raise click.ClickException(str(recipes_unhashable)) from recipes_unhashable

    if changed:
        raise click.ClickException(
            f"Run '{cfg['run_id']}' was started with different settings than the ones "
            f"you are using now.\n"
            f"  what changed: {', '.join(changed)}\n"
            f"A run is locked to the settings it began with, so its results stay "
            f"reproducible. Either:\n"
            f"  • put those settings back, and carry on with this run; or\n"
            f"  • changed a recipe only?  junior extract --new-run   reuses this run's "
            f"corpus under a fresh run id; or\n"
            f"  • start over under the new settings:  junior ingest --new-run   then  junior embed, index, extract"
        )

    # Recompute the bundle's own content hash against the sealed code.lock.json. This
    # catches tampering with the SNAPSHOT — a bundle altered in transit, on a share, or
    # after archiving. It does NOT look at the working tree: the comment here used to
    # claim it caught "code edited AFTER sealing (src/, recipes/)", and the test that
    # covered it edited a file inside the bundle, so a gate that was blind to the live
    # tree looked tested. Live-tree recipe drift is the check above.
    ok, detail = verify_code_bundle(run_root)
    if not ok:
        raise click.ClickException(
            "The code changed after this run started.\n"
            f"  what changed: {detail}\n"
            "A run records the exact code that produced it, so it will not continue "
            "under different code. Either put the change back, or start over:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )
    return code_bundle_for(run_root).code_lock_hash


def _finalize_exhaust_best_effort(run_id: str, dr: Path | None = None) -> dict | None:
    """Compact the run's NO_PHI exhaust shards into ``<record_type>.parquet`` + the
    shareable manifest. Best-effort — exhaust is telemetry; a finalize failure must
    not sink the calling command (mirrors the cohort runner's run-end behavior).
    Returns the manifest dict, or None if finalize failed."""
    try:
        from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust
        return finalize_exhaust(run_id, dr)
    except Exception as e:
        click.echo(f"  (note: the run's shareable summary was not finalized: {e})", err=True)
        return None


def _resolve_recipes_root(cfg: dict, repo_root: Path) -> Path:
    """Resolve the recipe tree to snapshot into the sealed code bundle."""
    configured = cfg.get("recipes_root")
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (repo_root / path).resolve()

    current_default = repo_root / "var_extraction_recipes"
    if current_default.is_dir():
        return current_default
    return repo_root / "recipes"


class _CuratedHelpGroup(click.Group):
    """A group whose help lists commands once, in the order you would run them.

    Click appends its own alphabetical `Commands:` block below the docstring. With a
    curated list already there, that produced two lists on one screen that disagreed
    on wording, ordering, and which commands even exist — the auto block includes
    every unhidden command and truncates each description mid-sentence."""

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        return None


@click.group(cls=_CuratedHelpGroup, invoke_without_command=True)
@click.version_option(__version__, prog_name="junior")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Junior — auditable per-patient clinical extraction.

    \b
    This CLI runs the pipeline: here, or on SLURM / a cloud VM. One
    command per stage, one patient at a time, so a cluster can fan the stages
    out across jobs. The run's code snapshot is written for you on the first
    stage you run. Type `junior` with no arguments for a prompt.

    \b
    Every command finds its config and its run for you — a junior.yaml here or
    above, else the bundled one; the newest run, unless you name another. Pass
    --config / --run-id to be explicit. Stages run the whole cohort; --patient
    narrows to one, which is what a SLURM array task passes.

    \b
    Set up new cohort:
      new-project      point Junior at your charts and choose where output goes
      columns          draft this site's column map from your export's headers

    \b
    Work with existing cohort — run these in order, once per patient:
      1. ingest        read the chart files into structured parquet
      2. embed         turn each text chunk into a vector
      3. index         build the per-patient vector index
      4. extract       pull the values out — re-run whenever a recipe changes

    \b
    Step 4 closes the run out when it finishes: it summarizes the run and checks
    it. There is no separate step to remember.

    \b
    Review a run:
      workbench        open the browser workbench — run, review, label, inspect

    \b
    Share a run:
      export-metadata  zip up the no-PHI summary of a run, to share or archive

    \b
    status             which config and run you are on, and how far along
    project            which cohort you are working on, and how to switch
    recipes            every recipe you can extract with, and which this project uses
    """
    # Bare `junior` opens a prompt rather than printing help and exiting. The prompt
    # dispatches back into this same group, so it adds a way in, not a second CLI.
    # The workbench is reached by clicking the app, or by the `workbench` command.
    if ctx.invoked_subcommand is None:
        from apps_and_interfaces.interactive_session import start

        start(main, __version__)


def _runs_with_extracted_values(data_root: Path) -> list[str]:
    """Runs under this project that produced values worth opening, newest first.

    A run id is a fixed-width timestamp, so sorting by name is sorting by date. Runs
    with no extract output are left out rather than offered: the workbench shows its
    bundled example when it is handed a run with nothing in it, and a reviewer looking
    at the example has no way to tell it is not their cohort."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        extract_output_dir,
        pipeline_run_receipts_root,
    )
    from jr_pipeline.runtime_infrastructure.project_context import RUN_ID_PATTERN

    receipts = pipeline_run_receipts_root(data_root)
    if not receipts.is_dir():
        return []
    found: list[str] = []
    for run_dir in sorted(receipts.iterdir(), key=lambda child: child.name, reverse=True):
        if not run_dir.is_dir() or not RUN_ID_PATTERN.fullmatch(run_dir.name):
            continue
        patients = run_dir / "patients"
        if not patients.is_dir():
            continue
        if any(
            any(extract_output_dir(patient).glob("*/result.json"))
            for patient in patients.iterdir()
            if patient.is_dir()
        ):
            found.append(run_dir.name)
    return found


def _free_port_from(port: int) -> int:
    """``port`` when nothing is listening on it, else the first free port in the
    next twenty. A workbench left running from an earlier session still holds its
    port, and failing with "address already in use" leaves the operator to find
    and kill a process themselves."""
    import socket

    def is_free(candidate: int) -> bool:
        with socket.socket() as probe:
            return probe.connect_ex(("127.0.0.1", candidate)) != 0

    if is_free(port):
        return port
    for candidate in range(port + 1, port + 21):
        if is_free(candidate):
            click.secho(f"  Port {port} is already in use; using {candidate}.",
                        err=True, fg="yellow")
            return candidate
    raise click.ClickException(
        f"Ports {port}–{port + 20} are all in use. Stop one of those servers, or "
        "pass a different --port."
    )


@main.command(help="Open the workbench in your browser: run, review, label, inspect.")
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Config file. Found automatically if omitted.")
@click.option("--run-id", "run_id", default=None,
              help="Run to open. The newest run with extracted values if omitted.")
@click.option("--port", default=8765, show_default=True, help="Port to serve it on.")
def workbench(config_path: Path | None, run_id: str | None, port: int) -> None:
    """Open the workbench on this project's runs.

    The app finds runs under JR_DATA_ROOT, so this resolves the project the same way
    every other command does and pins that variable to THIS cohort's output tree.
    Without it the app would fall back to the checkout's own data/ and open, in
    silence, on whichever unrelated run happened to be newest there — a reviewer
    would be correcting values and labelling passages on somebody else's patients."""
    import importlib.util
    import os
    import subprocess

    if importlib.util.find_spec("shiny") is None:
        raise click.ClickException(
            "The workbench runs on Shiny, which is not installed in this environment.\n"
            "From the repo root:  .venv/bin/python -m pip install -e '.[app]'"
        )
    app_file = _resolve_repo_root() / "apps_and_interfaces" / "shiny_review_app" / "app.py"
    if not app_file.is_file():
        raise click.ClickException(
            f"Workbench app not found at {app_file}. It ships with the source "
            "checkout; run `junior` from a clone rather than a bare wheel install."
        )

    cfg, config_origin = load_cohort_config(config_path, run_id)
    data_root_path = Path(cfg["output_root"])
    reviewable = _runs_with_extracted_values(data_root_path)
    if run_id and run_id not in reviewable:
        # A named run that has nothing in it is a wrong answer to a question the
        # operator actually asked, so name the ones that do rather than opening
        # something else and letting them find out on screen.
        listed = (
            "Runs in this project with values to look at:\n    " + "\n    ".join(reviewable)
            if reviewable
            else "No run in this project has been through `junior extract` yet."
        )
        raise click.ClickException(
            f"Run '{run_id}' has no extracted values under {data_root_path}.\n{listed}"
        )
    opening = run_id or (reviewable[0] if reviewable else None)

    # Pin the resolved project for the app process AND its children: the Start tab's
    # Run button launches `junior run` in a subprocess, and that child must resolve
    # the SAME project this command resolved — not whatever config discovery happens
    # to find from the server's working directory.
    # The PATH is re-resolved from discovery, never parsed out of the display
    # string — a project folder with " (" in its name made the parse cut inside the
    # origin's reason and pinned a garbage path.
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    os.environ[CONFIG_ENVIRONMENT_VARIABLE] = str(find_config(config_path).path)
    os.environ["JR_DATA_ROOT"] = str(data_root_path)
    if opening:
        os.environ["JR_REVIEW_RUN_ID"] = opening
    else:
        # Clear rather than leave: inside the prompt this process is reused, so a
        # selection left over from an earlier `workbench` would reopen a run the
        # operator has since moved away from.
        os.environ.pop("JR_REVIEW_RUN_ID", None)
    # A previous launch's patient is never right for a new one, and nothing here sets
    # it — the app picks the run's first patient and the reviewer switches in the app.
    os.environ.pop("JR_REVIEW_PATIENT_ID", None)

    port = _free_port_from(port)
    click.secho(f"  config  {config_origin}", err=True, dim=True)
    click.secho(f"  reading {data_root_path}", err=True, dim=True)
    if opening:
        click.secho(f"  run     {opening}", err=True, dim=True)
    else:
        click.secho(
            "  This project has no extracted values yet, so the workbench opens with\n"
            "  nothing selected. Run the pipeline from its Start tab, or `junior run`.",
            err=True, fg="yellow",
        )
    # The address, said by us. There were two ways to reach the app and both fail
    # quietly: --launch-browser goes through Python's webbrowser module, which returns
    # without opening anything when it cannot find a handler, and the only place the
    # URL otherwise appeared was uvicorn's "Uvicorn running on ..." line — an INFO
    # line, suppressed by the --log-level below. So a workbench that started perfectly
    # well showed a banner and nothing else, with no address to fall back on.
    click.secho(f"  open    http://127.0.0.1:{port}", err=True, fg="green")
    command = [
        sys.executable, "-m", "shiny", "run", str(app_file),
        "--host", "127.0.0.1", "--port", str(port), "--launch-browser",
        # A page load prints ~20 INFO lines, which scrolls the "stop with Ctrl-C"
        # banner away — and "how do I get out" was the actual question asked.
        "--log-level", "warning",
    ]
    if _in_interactive_session():
        # Never exec from inside the prompt: exec replaces this process, so the
        # session is gone for good and `quit` is never reached. Run it as a child
        # instead, so Ctrl-C stops the server and hands the prompt back.
        click.secho("  Starting the workbench. If your browser does not open, use the "
                    "address above.\n  Ctrl-C stops it and returns you here.", dim=True)
        try:
            subprocess.run(command, check=False)
        except KeyboardInterrupt:
            pass
        # Said on the way out as well as on the way in. The server owns the terminal
        # while it runs, so the line above is minutes and a page of output old by the
        # time anybody wants it, and "how do I get back" is asked at the end.
        click.secho("\n  workbench stopped — you are back at the junior prompt.", dim=True)
        return
    # From a shell, exec is right: the server runs in the foreground and Ctrl-C
    # should stop it directly, not land on a Python wrapper in between.
    os.execv(sys.executable, command)


DEFAULT_CONFIG_RELATIVE_PATH = Path("deployment") / "local" / "laptop.yaml"


@main.command(help="Run every stage for a cohort in one go, in one process.")
@click.argument("patient_folders", required=False,
                type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Pipeline config YAML. Defaults to the bundled local config.")
@click.option("--variable", "variables", multiple=True,
              help="Variable to extract; repeatable. Overrides the config's `recipes`.")
@click.option("--patient", "patients", multiple=True,
              help="Only these patients; repeatable. Omit to run every patient folder.")
@click.option("--run-id", "run_id", default=None,
              help="Work in one particular run. Rarely needed.")
@click.option("--output", "output_root", default=None,
              type=click.Path(file_okay=False, path_type=Path),
              help="Where to write outputs. Defaults to the config's output_root.")
def run(patient_folders: Path | None, config_path: Path | None,
        variables: tuple[str, ...], patients: tuple[str, ...],
        run_id: str | None, output_root: Path | None) -> None:
    """Seal the code bundle, then walk ingest -> embed -> index -> extract.

    PATIENT_FOLDERS is a directory holding one subfolder per patient; each
    subfolder's name is that patient's id. Omit it to use the config's source_root.
    """
    from jr_pipeline.runtime_infrastructure.cohort_runner import (
        check_phi_containment,
        cohort_settings_from_config,
        run_cohort,
        view_results,
    )

    # Found the same way the stage commands find theirs. Loading the bundled config
    # outright meant that standing in a cohort folder and typing `junior run` read the
    # checkout's own examples/ and wrote to the checkout's own data/ — the settings of
    # a project the operator was standing in were never consulted, and nothing said so.
    cfg, config_origin = load_cohort_config(config_path, run_id)
    if patient_folders is not None:
        cfg["source_root"] = str(patient_folders.expanduser().resolve())
    if patients:
        cfg["patients"] = list(patients)
    if output_root is not None and str(output_root.expanduser().resolve()) != cfg["output_root"]:
        # Only an --output that actually MOVES the destination resets the run choice:
        # the run was chosen by looking for the newest one under the config's output
        # root, and continuing a run picked from a directory nobody looked in is the
        # mistake this avoids. Spelling the config's own output root explicitly (the
        # app's Run button does, as its promise that results land where it reads)
        # changes nothing and so keeps the same continue-the-newest-run behaviour as
        # a bare `junior run` typed in the project.
        cfg["output_root"] = str(output_root.expanduser().resolve())
        if run_id is None:
            from jr_pipeline.runtime_infrastructure.project_context import resolve_run_id

            cfg["run_id"] = resolve_run_id(
                {}, None, data_root=Path(cfg["output_root"]), continue_newest=False
            )
    _bind_data_root(cfg)
    # Loud here rather than null values later: a variable with no recipe on disk is
    # dropped from the plan and the run reports success with the column simply absent.
    _planned_variables({**cfg, "recipes": list(variables) or cfg.get("recipes")})
    settings = cohort_settings_from_config(cfg, variables=variables)
    click.secho(f"  config  {config_origin}", err=True, dim=True)
    click.secho(f"  run     {cfg['run_id']}", err=True, dim=True)
    click.echo(f"Patients from: {settings.input_folder}")
    result = run_cohort(settings, run_id=cfg["run_id"])
    view_results(result, settings)
    check_phi_containment(settings)


@contextmanager
def _stage_monitor(
    stage: str,
    patient_id: str,
    run_root: Path,
    source_root: str | None,
    known_position: tuple[int, int] | None = None,
):
    """Show cohort position and a projected finish while a stage works.

    A raw clock only says time has passed. What an operator needs is how far through
    the cohort this is and roughly how long the rest will take, so the line reads
    ``ingest · patient 12 of 40 · 3s · ~6m left``. Position and the per-patient
    average come from the run's own state log, which means this works the same for a
    local loop and for a cluster where each patient is a separate process.

    Writes to stderr with a carriage return so it overwrites itself and never lands in
    piped output; when stderr is not a terminal it prints a start and a done line
    rather than animating into a logfile."""
    import threading
    import time

    from jr_pipeline.runtime_infrastructure.run_progress import (
        count_patient_folders,
        format_duration,
        read_stage_progress,
    )

    if known_position is not None:
        # Looping in this process: the position is known outright. Reading it back from
        # the log would undercount, because a patient that failed never records a
        # completion and would leave the counter stuck.
        index, total = known_position
        progress = read_stage_progress(run_root, stage, total=total)
        progress = replace(progress, completed=index - 1)
    else:
        progress = read_stage_progress(
            run_root, stage, total=count_patient_folders(source_root)
        )
    position = (
        f"patient {progress.completed + 1} of {progress.total}"
        if progress.total
        else f"patient {progress.completed + 1}"
    )
    # The patient is on this line because on a terminal this line is the only one that
    # survives: it owns its row with a carriage return, so anything printed underneath
    # it is overwritten. Extraction's per-variable display is stood down while it runs
    # (see cohort_runner), which leaves one moving line saying who, how far, how long.
    prefix = f"  {stage} · {position} · {patient_id}"

    animated = sys.stderr.isatty()
    started = time.monotonic()
    stop = threading.Event()

    said_it_is_busy = False

    def _somebody_is_typing() -> bool:
        """Whether keystrokes are sitting unread at the terminal.

        Asked, never answered: this deliberately does not READ them. A stage can be
        interrupted, and the Ctrl-C confirmation reads stdin for its y/n — a ticker
        draining the buffer once a second would race it for that keypress and eat the
        answer to "stop this run?". The keystrokes stay put and are discarded when the
        command ends, which is already handled. All this does is notice."""
        try:
            import select

            return bool(sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0])
        except (OSError, ValueError, ImportError):
            return False

    def _tick() -> None:
        nonlocal said_it_is_busy
        while not stop.wait(1.0):
            elapsed = time.monotonic() - started
            left = progress.estimated_seconds_left(elapsed)
            # No estimate on the first patient of a stage — there is nothing to
            # average yet, and a made-up number is worse than saying so.
            tail = f" · ~{format_duration(left)} left" if left else " · estimating"
            if not said_it_is_busy and _somebody_is_typing():
                # Said WHILE somebody is waiting on it, not after the stage ends. This
                # runs for minutes; a note that arrives at the end answers a question
                # they gave up on. Typing at this point is somebody trying to start
                # another command, and the terminal echoing it looks like it worked.
                said_it_is_busy = True
                # Two truths for two doors. Inside the prompt, the session discards
                # type-ahead when the command ends, so "ignored" is a promise it
                # keeps. At a plain shell nothing here can discard anything — the
                # SHELL will run those keystrokes the moment this finishes, and
                # telling the operator they are ignored is the opposite of a warning
                # they need.
                what_happens_to_it = (
                    "What you type now is ignored."
                    if _in_interactive_session()
                    else "What you type now will run in your shell when it finishes."
                )
                click.secho(
                    f"\r\033[K  still running — one command at a time. {what_happens_to_it}",
                    err=True, fg="yellow",
                )
            click.echo(f"\r{prefix} · {format_duration(elapsed)}{tail}   ", nl=False, err=True)

    def _clear_ticker() -> None:
        stop.set()
        if animated:
            click.echo("\r\033[K", nl=False, err=True)

    if animated:
        threading.Thread(target=_tick, daemon=True).start()
    else:
        click.echo(f"{prefix} …", err=True)

    try:
        yield
    except BaseException as e:
        # Only a completed stage may report "done". A bare finally would print a
        # success line for a Ctrl-C or a crash, which is the worst thing this
        # display could do — it would say a patient finished when nothing was written.
        _clear_ticker()
        if isinstance(e, KeyboardInterrupt):
            # "Nothing was written" was false exactly where stopping is most likely:
            # extract records each finished variable as it goes, and those results
            # stay on disk and resume. Only the work in flight is unrecorded.
            click.secho(
                f"{prefix} · stopped after {format_duration(time.monotonic() - started)}"
                " — the work in flight was not recorded; anything already finished "
                "is kept and will resume",
                err=True, fg="yellow",
            )
        raise
    _clear_ticker()
    done = f"{prefix} · done in {format_duration(time.monotonic() - started)}"
    if progress.remaining:
        done += f" · {progress.remaining} to go"
    click.secho(done, err=True, dim=True)


def _anchor_to_repo(value: str, repo_root: Path) -> Path:
    """Resolve a config path against the checkout, not the caller's directory.

    Absolute paths and ones the caller clearly meant relative to home (``~``) are left
    alone; only a bare relative path is anchored, because that is the form the shipped
    configs use to mean "inside this repo"."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


# One sentence per stage, shown as it starts. A stage that prints nothing but a clock
# leaves the operator guessing what it is doing to their data.
STAGE_EXPLANATIONS: dict[str, str] = {
    "ingest": "Reading each patient's chart files and saving them as tables. "
              "Nothing is interpreted yet — this is a faithful copy.",
    "embed": "Turning the text of each chart into vectors, so passages can be found "
             "by meaning rather than exact words.",
    "index": "Building each patient's search index over those vectors.",
    "retrieve": "Fetching the passages a query matches, for inspection.",
    "extract": "Finding the evidence for each variable and reading the answer out of "
               "it. Every value keeps the chart passage it came from.",
}


def _one_line_result(stage: str, summary: dict) -> str:
    """What a stage did for one patient, in a phrase rather than a JSON dump."""
    if summary.get("cached"):
        return "already done — reused what was there"
    def _ingested(summary: dict) -> str:
        """How many of this patient's files became tables, out of how many were asked for.

        The bare count cannot be read: one patient's 17 is seventeen source files, and
        another's 17 is nineteen files with two missing, and ingest is the stage whose
        whole job is telling those apart."""
        written = len(summary.get("files_written") or [])
        asked_for = summary.get("files_asked_for")
        if asked_for is None or asked_for == written:
            return f"{written} of {written} files saved as tables"
        return f"{written} of {asked_for} files saved as tables — {asked_for - written} did not arrive"

    counted = {
        "ingest": _ingested,
        "embed": lambda s: f"{s.get('chunks', '?')} passages embedded",
        "index": lambda s: f"{s.get('size', '?')} passages indexed",
        "extract": lambda s: f"{len(s.get('variables') or {})} variables read",
    }.get(stage)
    return counted(summary) if counted else "done"


def _quiet_model_loading_chatter() -> None:
    """Stop the model libraries narrating their own startup to the operator.

    Loading a model prints, unasked: an OpenMP warning about /tmp, a deprecation
    notice about torch_dtype, and a table headed "LOAD REPORT" listing weights as
    UNEXPECTED. The last is the worst — it is normal (those weights belong to a
    training head this pipeline does not use) but it reads as a broken model, and
    the footnote explaining it away is itself jargon.

    The progress bar goes too: it redraws with carriage returns, which a terminal
    handles and a SLURM logfile turns into one enormous line. Junior prints its own
    progress, and -v restores all of this."""
    import os

    os.environ.setdefault("KMP_WARNINGS", "off")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        # No transformers installed, or its logging API moved. Quieting output is a
        # courtesy; failing to do it must never stop a run.
        pass


def _nothing_was_found(answer: Any) -> bool:
    """Whether a recipe's answer field amounts to no value.

    An empty list or string is how several recipes spell absence, so treating only
    None as absent reported them as found. Written by type rather than by falsiness
    because 0 and False are answers."""
    return answer is None or (isinstance(answer, (str, list, dict)) and not answer)


def _report_extracted_variables(summary: dict) -> None:
    """Say which variables came back with a value and which did not.

    Waiting minutes for a model and then being told only "1 variables read" leaves
    the operator unable to tell a finished run from a run that found nothing. The
    VALUES stay off screen — they are patient data, and a terminal outlives the run
    in scrollback and SLURM logs — so this reports presence and confidence, and
    points at the app for the values themselves.

    Whether a recipe found anything is asked the way the answer tables ask it, not by
    looking for a field named after the variable. That test only works for the three
    recipes whose answer happens to share its name — date_of_birth, date_of_death,
    date_of_diagnosis — and the other fifteen fell through to "read: open it to see
    what it holds", which is a display declining to answer its own question on 83% of
    the recipe book. It was written that way for a good reason: an earlier version
    guessed the answer field by taking the first name without an underscore, picked up
    the model's `rationale` — which every prompting recipe fills in — and reported
    `date_of_birth: found` for a chart with no date of birth in it.

    The rule step 8 uses for provenance does not have that failure: a rationale, a
    certainty score, an evidence pointer and a null placeholder are all structural and
    none of them is a finding. It works on any shape, so a table of rows and a nested
    object answer as readily as a bare field."""
    from jr_pipeline.runtime_infrastructure.cohort_runner import _stdout_values_allowed
    from jr_pipeline.runtime_infrastructure.values_table import (
        _any_value_found,
        _flatten,
        _record_list_fields,
    )

    variables = summary.get("variables") or {}
    if not variables:
        return
    show_values = _stdout_values_allowed()
    for variable, outcome in sorted(variables.items()):
        data = outcome.get("data") or {}
        if not outcome.get("ok"):
            click.secho(f"       {variable}: failed", fg="red")
        # _flatten leaves a list of records to the caller, so a recipe that answers
        # with a table -- pathology_table's rows, treatment_lines' lines -- would look
        # empty on its patient-level fields alone. Records ARE the finding there.
        elif not _any_value_found(_flatten(data)) and not any(
            data.get(field) for field in _record_list_fields(data)
        ):
            click.secho(f"       {variable}: nothing found in this chart", fg="yellow")
        elif show_values and variable in data and not _nothing_was_found(data[variable]):
            click.secho(f"       {variable}: {data[variable]}", fg="green")
        else:
            click.secho(f"       {variable}: found", fg="green")
    if not show_values:
        click.secho("       (values are patient data — see them with `junior workbench`)",
                    dim=True)


def _patients_with_a_corpus(run_id: str) -> list[str]:
    """Patients the given run actually built a corpus for."""
    patients_dir = phi_intermediate_run_dir(run_id) / "patients"
    if not patients_dir.is_dir():
        return []
    return sorted(
        child.name for child in patients_dir.iterdir()
        if child.is_dir() and (child / "chunk_index.parquet").is_file()
    )


def _record_inherited_corpus(
    run_id: str, source_run_id: str, inherited: list[dict]
) -> None:
    """Write down that this run's corpus came from another one.

    A run whose vectors it did not build looks identical on disk to one that did, and
    "where did these come from" is exactly the question the two-stream provenance model
    exists to answer. The per-artifact proof is already there and is not copied here:
    each inherited sidecar still carries its own content hash and the produced_by block
    naming the run and code_lock_hash that built it, so this file says which run was
    inherited from and the sidecars say what was inherited, each hash still checkable
    against the bytes beside it."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        atomic_write_json,
    )

    run_root = phi_intermediate_run_dir(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    # On a CHAIN of inheritances (run C inherits from B, which inherited from A) the
    # builder of the vectors is A, not B — so each patient's corpus_source_run_id is
    # carried forward from the source's own record rather than restamped. parent_run_id
    # stays the run this one continues from; the two answer different questions.
    source_record_path = phi_intermediate_run_dir(source_run_id) / "corpus_inheritance.json"
    original_builder_by_patient: dict[str, str] = {}
    if source_record_path.is_file():
        try:
            source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
            original_builder_by_patient = {
                entry["patient_id"]: entry["corpus_source_run_id"]
                for entry in source_record.get("patients", [])
                if entry.get("patient_id") and entry.get("corpus_source_run_id")
            }
        except (OSError, json.JSONDecodeError):
            original_builder_by_patient = {}
    patients = [
        {**entry, "corpus_source_run_id": original_builder_by_patient.get(
            entry["patient_id"], entry["corpus_source_run_id"],
        )}
        for entry in inherited
    ]
    builders = sorted({entry["corpus_source_run_id"] for entry in patients})
    atomic_write_json(run_root / "corpus_inheritance.json", {
        "run_id": run_id,
        "parent_run_id": source_run_id,
        # The single run that built every patient's vectors, or None when patients
        # trace to different builders — then the per-patient entries are the answer.
        "corpus_source_run_id": builders[0] if len(builders) == 1 else None,
        "patients": patients,
    })


def _sealed_config_of(run_id: str) -> dict:
    """The config a run was sealed with — what actually produced its artifacts."""
    import yaml

    sealed = phi_intermediate_run_dir(run_id) / "code" / "config_resolved.yaml"
    if not sealed.is_file():
        return {}
    try:
        return yaml.safe_load(sealed.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _sealed_at_of(run_id: str) -> float | None:
    """When the run's bundle was sealed, as an epoch timestamp; None if unreadable."""
    from datetime import datetime

    lock_path = phi_intermediate_run_dir(run_id) / "code" / "code.lock.json"
    if not lock_path.is_file():
        return None
    try:
        sealed_at = json.loads(lock_path.read_text(encoding="utf-8"))["payload"]["sealed_at"]
        return datetime.fromisoformat(sealed_at).timestamp()
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def _refuse_if_the_corpus_cannot_be_inherited(
    cfg: dict, source_run_id: str, patients: list[str],
    requested_patient: str | None = None,
) -> None:
    """Every reason `extract --new-run` cannot go ahead, asked before anything is made.

    Separate from the copying, and called first, because a refusal that arrives after a
    run directory exists is worse than the problem it reports. Declining the extract
    confirmation used to leave a directory holding a full corpus and no code/ — and that
    became the newest run, so the next `--new-run` inherited from it, found no sealed
    config, and refused every time after. A --patient with no corpus was the same trap
    one level down: the check on `patients` passed (the list held the name), the copy
    then failed after the run directory existed, and THAT empty run became the newest."""
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
        INCOMPLETE_INHERITANCE_MARKER,
        patients_whose_charts_changed,
        why_the_corpus_cannot_be_reused,
    )

    sealed = _sealed_config_of(source_run_id)
    if not sealed:
        # Distinguished from "your settings changed", which is what this said before.
        # With no runs on disk, resolve_run_id mints an id for a run that never existed,
        # and the operator was told their chart-reading settings had changed.
        raise click.ClickException(
            f"There is no finished run to extract from — {source_run_id} has no sealed "
            "settings.\nBuild a corpus first:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )
    if (phi_intermediate_run_dir(source_run_id) / INCOMPLETE_INHERITANCE_MARKER).exists():
        raise click.ClickException(
            f"Run {source_run_id} holds a half-copied corpus — an inheritance into it "
            "was interrupted.\nDelete that run's folder, then run "
            "`junior extract --new-run` again."
        )
    with_corpus = _patients_with_a_corpus(source_run_id)
    if requested_patient is not None and requested_patient not in with_corpus:
        listed = (
            "Patients with a finished corpus in that run:\n    " + "\n    ".join(with_corpus)
            if with_corpus
            else f"Run {source_run_id} has no patient with a finished corpus."
        )
        raise click.ClickException(
            f"'{requested_patient}' has no finished corpus in run {source_run_id}, so "
            f"there is nothing to inherit for them.\n{listed}\n"
            "Build theirs first:  junior ingest --new-run   then  junior embed, index, "
            "extract"
        )
    if not patients:
        raise click.ClickException(
            f"Run {source_run_id} has no patient with a finished corpus to reuse.\n"
            "Build one first:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )
    if requested_patient is None and cfg.get("source_root"):
        # The new run's roster is what can be inherited, so a patient in the cohort
        # WITHOUT a corpus would not fail — they would silently not exist in the new
        # run, and the cohort would shrink with nothing said.
        absent = sorted(set(cohort_patients(cfg)) - set(patients))
        if absent:
            raise click.ClickException(
                f"{len(absent)} patient(s) in this cohort have no finished corpus in "
                f"run {source_run_id}, so an inherited run would silently leave them "
                f"out: {', '.join(absent[:6])}{', …' if len(absent) > 6 else ''}\n"
                "Either build their corpus first (junior ingest, embed, index), or "
                "narrow this extraction:  junior extract --new-run --patient <id>"
            )
    changed = why_the_corpus_cannot_be_reused(
        cfg, sealed, source_sealed_at=_sealed_at_of(source_run_id),
    )
    if changed:
        raise click.ClickException(
            "This changes how the charts themselves are read, so the passages and "
            "vectors have to be rebuilt:\n"
            + "".join(f"  · {key}\n" for key in changed)
            + "Rebuild it, a stage at a time:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )
    moved = patients_whose_charts_changed(
        source_run_id=source_run_id, patients=list(patients),
        source_root=cfg.get("source_root") or "",
    )
    if moved:
        raise click.ClickException(
            "The chart files for "
            f"{', '.join(moved[:6])}{', …' if len(moved) > 6 else ''} changed after run "
            f"{source_run_id} read them, so that corpus no longer describes their "
            "charts.\nRe-ingest them first:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )


def _inherit_the_corpus_into_a_fresh_run(
    cfg: dict, source_run_id: str, patients: list[str]
) -> None:
    """Carry ingest/embed/index output from `source_run_id` into this new run.

    `extract --new-run` exists so that changing a recipe does not rebuild a corpus the
    change cannot have touched. Whether that is honest is decided by
    _refuse_if_the_corpus_cannot_be_inherited, which runs first; this only copies.

    A marker file stands in the new run for the whole copy: written before the first
    file moves, removed only after the provenance record is down. A crash mid-copy
    otherwise leaves a half-corpus indistinguishable from a whole one, which a later
    bare `junior extract` would resume and answer from whichever patients happened to
    finish. Every door back into a run refuses while the marker stands."""
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
        INCOMPLETE_INHERITANCE_MARKER,
        inherit_corpus_for_patient,
    )

    run_root = phi_intermediate_run_dir(cfg["run_id"])
    run_root.mkdir(parents=True, exist_ok=True)
    marker = run_root / INCOMPLETE_INHERITANCE_MARKER
    marker.write_text(
        f"inheriting from {source_run_id} — if this file is still here, the copy was "
        "interrupted; delete this run's folder and run `junior extract --new-run` "
        "again\n",
        encoding="utf-8",
    )
    inherited = []
    for patient in patients:
        try:
            inherited.append(inherit_corpus_for_patient(
                source_run_id=source_run_id,
                destination_run_id=cfg["run_id"],
                patient_id=patient,
            ))
        except FileNotFoundError as missing:
            raise click.ClickException(
                f"{missing}\n"
                "Build the corpus first:\n"
                "  junior ingest --new-run   then  junior embed, index, extract"
            ) from missing

    how = sorted({method for record in inherited for method in record["how"]})
    click.secho(
        f"  corpus  reused from {source_run_id} — {len(inherited)} patient(s), "
        f"nothing re-embedded ({', '.join(how)})",
        err=True, dim=True,
    )
    _record_inherited_corpus(cfg["run_id"], source_run_id, inherited)
    marker.unlink()


def _one_run_or_the_other(run_id: str | None, new_run: bool) -> None:
    """--new-run and --run-id answer the same question two ways, so only one may ask.

    They are not variations on a theme: --new-run means a run that does not exist yet,
    --run-id means one that does. Letting both through would mean silently honouring
    one and discarding the other, and either choice is wrong half the time."""
    if new_run and run_id:
        raise click.UsageError(
            "--new-run starts a run; --run-id picks one that already exists. "
            "Use one:\n"
            "  junior extract --new-run        start over\n"
            "  junior extract --run-id <id>    go back to a particular run"
        )


def load_cohort_config(config_path: Path | None, run_id: str | None) -> tuple[dict, str]:
    """Resolve which config and which run a stage command is operating on.

    Returns the config with run_id, source_root, and output_root filled in, plus a
    sentence saying where the config came from — a command that picks a file for you
    has to say which one, or a wrong cohort is silent."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config, resolve_run_id

    # Everything below funnels through here, so this is where a broken settings file
    # or a missing one becomes a sentence rather than a traceback.
    try:
        choice = find_config(config_path)
        cfg = load_config(choice.path)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    except ValueError as e:
        raise click.ClickException(
            f"{e}\nFix that file and run this again."
        ) from e

    # A project owns its own folder, which is what keeps two of them apart. A settings
    # file sitting directly in the home directory instead captures every command run
    # anywhere below it — an unrelated checkout, a scratch directory, another project
    # that has no settings file of its own — and nothing about typing `junior ingest`
    # would suggest a cohort three directories up had claimed it. Discovery still
    # honours it, because someone may have put it there deliberately, but it says so.
    if choice.path.parent == Path.home():
        click.secho(
            f"  Using {choice.path}, which is sitting in your home folder instead of in "
            f"a folder of its own.\n"
            f"  Junior looks in the folder you are in, then every folder above it — and "
            f"your home folder is\n"
            f"  above nearly all of them, so this is what it finds almost anywhere, "
            f"including folders that\n"
            f"  have nothing to do with this cohort. Moving that file into a folder of "
            f"its own is what stops it.\n"
            f"  `junior project` lists the others on this machine.",
            err=True, fg="yellow",
        )

    repo_root = _resolve_repo_root()
    # The bundled config leaves these unset so a null reads as "not configured" rather
    # than a wrong path. Filling them here is what lets a bare `junior ingest` work on
    # a fresh clone without exporting three environment variables first.
    if not cfg.get("source_root"):
        cfg["source_root"] = str(repo_root / "examples")
    if not cfg.get("output_root"):
        cfg["output_root"] = str(repo_root / "data")

    # The shipped configs name repo locations as "./models/...", "./var_extraction_recipes".
    # Those only ever meant "in the checkout", but a bare relative path resolves against
    # the caller's directory, so running from anywhere else used to find nothing. That
    # failed quietly rather than loudly — a missing model yields an empty answer, not an
    # error — so anchor them to the checkout here, where the mistake is still cheap.
    for key in ("recipes_root", "allowlist_path"):
        if cfg.get(key):
            cfg[key] = str(_anchor_to_repo(cfg[key], repo_root))
    # The column map is the one path a PROJECT owns: `junior columns` writes the map
    # beside the settings file and records the bare name, so a relative spelling
    # resolves against the config's own folder first — which also survives the
    # project folder being moved. A repo spelling (deployment/<site>/...) does not
    # exist there and still anchors to the checkout.
    if cfg.get("chart_columns_file"):
        named = Path(cfg["chart_columns_file"]).expanduser()
        beside_the_config = choice.path.parent / named
        if not named.is_absolute() and beside_the_config.is_file():
            cfg["chart_columns_file"] = str(beside_the_config.resolve())
        else:
            cfg["chart_columns_file"] = str(_anchor_to_repo(cfg["chart_columns_file"], repo_root))
    encoder = cfg.get("encoder")
    if isinstance(encoder, dict) and encoder.get("model_id"):
        encoder["model_id"] = str(_anchor_to_repo(encoder["model_id"], repo_root))

    data_root = _bind_data_root(cfg)
    # Written back, not just bound. `output_root` is what a settings file was typed with,
    # so it can be `~/cohort` or a relative path, and the readers that join a run onto it
    # then look somewhere that does not exist — a run the app is happily serving gets
    # reported as "no run in this project", which reads as "your extraction produced
    # nothing". One expansion here rather than one at each reader, none of which can be
    # relied on to remember.
    cfg["output_root"] = str(data_root)

    # A project pins its own folder as output_root when it is created. Move or rename
    # that folder and the settings file travels with it, so every command still finds
    # the project — but the pinned path is gone, the layout is recreated empty there,
    # and `status` reports a cohort with no runs while its receipts sit in the folder
    # the operator is standing in. Said here because the next stage would otherwise
    # start the whole cohort again, in silence, somewhere nobody is looking.
    if not data_root.is_dir():
        beside_the_config = choice.path.parent
        if not (beside_the_config / "CONTAINS_PHI" / "pipeline_run_receipts").is_dir():
            # Gone, and not merely moved. `new-project` creates this tree, so a config
            # naming a folder that is not there means it was deleted, or the file was
            # written by hand. Said rather than repaired: this is the operator's file,
            # and Junior choosing a new output_root for them would be a worse surprise
            # than a missing one. Every screen that prints this path — `status` most of
            # all — was otherwise stating a location that does not exist as plain fact.
            click.secho(
                f"  This project writes to {data_root}, which is not there.\n"
                f"  The next stage you run creates it. If that is not where you meant "
                f"it to go, set `output_root:` in {choice.path.name}.",
                err=True, fg="yellow",
            )
        else:
            click.secho(
                f"  This project writes to {data_root}, which is not there any more.\n"
                f"  Its runs are in {beside_the_config} — the folder looks moved or "
                f"renamed.\n"
                f"  Set  output_root: {beside_the_config}  in {choice.path.name}, or the "
                f"next stage starts the cohort over.",
                err=True, fg="yellow",
            )

    # Continuing "the newest run on disk" is right inside a project — it is what makes
    # `ingest` then `embed` land together. It is DANGEROUS with the bundled fallback:
    # that points at the checkout's own data/, so a stray command in an unrelated
    # directory would try to continue whatever real run happened to be newest there,
    # appending to someone's clinical audit trail. Outside a project, always start
    # fresh, and say plainly where the work is going.
    inside_a_project = choice.reason != "bundled default"
    cfg["run_id"] = resolve_run_id(
        cfg, run_id, data_root=data_root, continue_newest=inside_a_project
    )
    if not inside_a_project and config_path is None:
        click.secho(
            f"  no project here — using the bundled example settings\n"
            f"          reading {cfg['source_root']}\n"
            f"          writing {cfg['output_root']}\n"
            f"          `junior new-project` sets up your own.",
            err=True, fg="yellow",
        )
    return cfg, f"{choice.path} ({choice.reason})"


def cohort_patients(cfg: dict) -> list[str]:
    """Every patient folder under the configured source root, in a stable order.

    Three different mistakes used to produce the same "none found" message: the path
    does not exist, it exists but is empty, and — much the most common — it is one
    patient's folder rather than the folder of patients. They are told apart here,
    because the fix for each is different."""
    source_root = Path(cfg["source_root"]).expanduser()
    if not source_root.exists():
        raise click.ClickException(
            f"{source_root} does not exist.\n"
            "That is where Junior expects your charts. Create it and put one folder "
            f"per patient inside, or point `source_root:` in "
            f"{_this_projects_settings_name()} "
            "somewhere else."
        )
    if not source_root.is_dir():
        raise click.ClickException(f"{source_root} is a file, not a folder of patients.")

    patients = sorted(
        child.name
        for child in source_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )
    if not patients:
        loose_files = sorted(
            child.name for child in source_root.iterdir()
            if child.is_file() and not child.name.startswith(".")
        )
        if loose_files:
            shown = ", ".join(loose_files[:3]) + ("…" if len(loose_files) > 3 else "")
            raise click.ClickException(
                f"{source_root} has no patient folders in it, but it does hold "
                f"{len(loose_files)} file(s) directly: {shown}\n"
                "That looks like ONE patient's folder. Junior wants the folder ABOVE "
                f"it — the one holding a folder per patient — so try {source_root.parent}, "
                "or run this one patient with --patient."
            )
        raise click.ClickException(
            f"{source_root} is empty.\n"
            "Put one folder per patient inside it; each folder's name becomes that "
            "patient's id."
        )
    return patients


def patients_for_stage(stage: str, cfg: dict, run_root: Path) -> list[str]:
    """Which patients this stage should run over when no --patient was given.

    Ingest reads the source tree, and screens it first: a directory of directories
    (an export bundle sitting beside the patients) is not a patient, and reporting it
    as a failure every run trains the operator to ignore failures.

    Every later stage reads the run, not the source tree — its cohort is whoever
    actually made it through ingest, which is also what run_cohort feeds them."""
    if stage != "ingest":
        patients_dir = run_root / "patients"
        if not patients_dir.is_dir():
            raise click.ClickException(
                f"Nothing ingested in run {run_root.name} yet.\n"
                "Run `junior ingest` first."
            )
        patients = sorted(child.name for child in patients_dir.iterdir() if child.is_dir())
        # The same gate in the other direction: index and extract read what embed
        # writes, and running them on a run with no embeddings produced one red
        # failure per patient plus a --force retry hint that fails identically. A
        # whole-cohort miss is an ordering mistake, and its fix is a stage, not a
        # retry. (Some patients embedded and some not is a different situation —
        # per-patient reporting handles it.)
        if stage in ("index", "extract", "retrieve") and not any(
            (patients_dir / pid / "chunk_index.parquet").is_file() for pid in patients
        ):
            next_steps = (
                "Run `junior embed` first."
                if stage == "index"
                else "Run `junior embed`, then `junior index`, first."
            )
            raise click.ClickException(
                f"Nothing embedded in run {run_root.name} yet.\n{next_steps}"
            )
        return patients


    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import preflight_patients

    candidates = cohort_patients(cfg)
    preflight = preflight_patients(cfg=cfg, patients=candidates)
    # Preflight tells seven kinds of problem apart and attaches what to do about each.
    # Reporting one dim sentence for all of them — "not a readable patient folder" —
    # was true of exactly one kind, and being wrong about the rest is what made the
    # line worth ignoring: a patient whose export has two-digit years, or whose free
    # text is in a column the map does not name, was called unreadable and then went
    # missing from every later stage and from the exported dataset.
    reason_for: dict[str, Any] = {}
    for issue in preflight.issues:
        if issue.blocking:
            reason_for.setdefault(issue.patient_id, issue)
    for blocked in preflight.blocked:
        blocking_issue = reason_for.get(blocked)
        where = (
            f" ({blocking_issue.file})"
            if blocking_issue is not None and blocking_issue.file else ""
        )
        detail = (
            blocking_issue.detail if blocking_issue is not None
            else "not a readable patient folder"
        )
        click.secho(f"  – skipping {blocked}{where}: {detail}", err=True, fg="yellow")

    _report_what_ingest_is_about_to_settle(preflight, cfg)
    return list(preflight.ok)


def _report_what_ingest_is_about_to_settle(preflight: Any, cfg: dict) -> None:
    """Say what this ingest is deciding permanently, and nothing else.

    Two kinds of thing land in the parquet and are inherited by every later stage.
    Malformation — a column whose name matches two different chart fields, a date that
    could be read two ways — and the chart-field mapping itself.

    The mapping used to be left out here, on the reasoning that it belonged to the stages
    that consume those fields. That reasoning was wrong about this pipeline:
    ``resolve_metadata_columns_for`` runs at INGEST (step_1/ingest.py) and writes its
    answer to each file's sidecar; embed reads the sidecar and re-derives only when it is
    absent or names a column the table does not have. A field resolved to None is not
    "missing" to embed, it is settled. So the one place that stayed quiet was the one
    place where staying quiet is permanent.

    The other half of that old reasoning still holds, and is what the filtering below is
    for: most unresolved fields are not a problem. A lab table has no author because lab
    tables do not have authors, and saying so once per file per field, multiplied by
    every patient, is how a report becomes something to scroll past. Only files that
    carry free text are reported — those are the ones whose chunks reach retrieval
    wearing this metadata, and on the shipped example that is the difference between one
    line and eight."""
    from collections import Counter

    malformations = Counter(
        advisory.detail for advisory in preflight.advisories if not advisory.fields
    )
    for detail, how_many in malformations.most_common(3):
        click.secho(f"  note  {detail} ({how_many} file(s))", err=True, dim=True)
    if len(malformations) > 3:
        click.secho(f"  note  and {len(malformations) - 3} more of the same kind, in "
                    f"run_log.jsonl", err=True, dim=True)

    # Grouped by field rather than listed per file: a 500-patient cohort has the same
    # handful of unmapped fields 500 times over, and the count is the useful part.
    unmapped: Counter = Counter()
    for advisory in preflight.advisories:
        if advisory.fields and advisory.file_contributes_text:
            for field_name in advisory.fields:
                unmapped[field_name] += 1
    for field_name, how_many in unmapped.most_common(3):
        click.secho(
            f"  note  no column holds {field_name} in {how_many} file(s) that carry text "
            f"— a recipe filtering on it drops their evidence",
            err=True, dim=True,
        )
    if len(unmapped) > 3:
        click.secho(f"  note  and {len(unmapped) - 3} more field(s) with no column, in "
                    f"run_log.jsonl", err=True, dim=True)

    if unmapped and not cfg.get("chart_columns_file"):
        # One line, because an operator at a keyboard is about to be asked this properly
        # by _confirm_column_mapping and a paragraph here would only be the same warning
        # scrolling past before the question. This line is for the runs nobody is
        # watching — piped, scripted, a SLURM task — where the confirmation never fires
        # and the log is the only record that the mapping was guessed.
        #
        # Said only when there is no map at all. With one set, unresolved fields are
        # ordinary — the shipped example map leaves the same eight unresolved on the
        # example cohort — so treating those as an alarm every run would teach the
        # operator to scroll past the run where it mattered.
        click.secho(
            "  note  no column map is set, so those fields were looked for under generic "
            "names",
            err=True, fg="yellow",
        )


def _common_step(name: str, help_text: str):
    """Register a step subcommand with consistent flags."""
    @main.command(name=name, help=help_text)
    @click.option("--config", "config_path", default=None,
                  type=click.Path(exists=True, dir_okay=False, path_type=Path),
                  help="Config file. Found automatically if omitted.")
    @click.option("--patient", "patient_id", default=None, type=str,
                  help="One patient. Omit to run the whole cohort.")
    @click.option("--new-run", "new_run", is_flag=True, default=False,
                  help="Start over in a fresh run instead of continuing this one.")
    @click.option("--run-id", "run_id", default=None,
                  help="Work in one particular run. Rarely needed — see --new-run.")
    @click.option("--variable", "variables", multiple=True, hidden=(name != "extract"),
                  help="Variable to extract; repeatable. Overrides the project's "
                       "`recipes:` list. Extract only.")
    @click.option("--force/--no-force", default=False, help="Recompute even if cached outputs exist.")
    @click.option("--verbose", "-v", is_flag=True,
                  help="Show each structured event as it happens, not just the summary.")
    @click.pass_context
    def _cmd(ctx: click.Context, config_path: Path | None, patient_id: str | None,
             run_id: str | None, new_run: bool, variables: tuple[str, ...],
             force: bool, verbose: bool) -> None:
        _one_run_or_the_other(run_id, new_run)
        if variables and name != "extract":
            raise click.ClickException(
                f"--variable chooses what to extract, and {name} extracts nothing. "
                f"Pass it to `junior extract`."
            )
        # Loaded as the run this command WOULD have continued, then re-pointed. That
        # order matters for `extract --new-run`: it is a fresh run for the answers and a
        # continuation of the corpus, so the run being inherited from has to be known
        # before a new id exists, and minting one is precisely what makes it unfindable.
        cfg, config_origin = load_cohort_config(config_path, run_id)
        if variables:
            # What `junior run --variable` does, said the same way here: the names
            # asked for win over the project's `recipes:`, and _planned_variables
            # still refuses any that have no recipe on disk. Without this the
            # workbench's variable ticks reached nothing — extract ran the project's
            # list and the tick looked like it had simply not worked.
            cfg["recipes"] = list(variables)
        inherit_from: str | None = None
        if new_run:
            from jr_pipeline.runtime_infrastructure.project_context import resolve_run_id

            if name == "extract":
                inherit_from = str(cfg["run_id"])
            cfg["run_id"] = resolve_run_id({}, None, start_fresh=True)
        run_root = phi_intermediate_run_dir(cfg["run_id"])

        click.secho(f"\n  {STAGE_EXPLANATIONS[name]}", italic=True)
        click.secho(f"  config  {config_origin}", err=True, dim=True)
        click.secho(f"  run     {cfg['run_id']}", err=True, dim=True)

        _refuse_if_this_project_points_at_nothing(cfg, config_origin, name)

        # A run whose inheritance was interrupted holds a half-corpus that looks whole
        # patient by patient. No stage may resume it — least of all extract, which
        # would answer from whichever patients happened to finish copying.
        from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
            INCOMPLETE_INHERITANCE_MARKER,
        )

        if (run_root / INCOMPLETE_INHERITANCE_MARKER).exists():
            raise click.ClickException(
                f"Run {cfg['run_id']} holds a half-copied corpus — the inheritance "
                "that was building it was interrupted.\nDelete that run's folder, "
                "then run `junior extract --new-run` again."
            )

        # Checked before anything is created, done only once the answer is yes. Doing
        # the whole thing up front left a declined command with a run directory holding
        # a full corpus and no code/ — and the NEXT `--new-run` took that as its source,
        # found no sealed config to compare against, and refused every time after.
        inheritable: list[str] = []
        if inherit_from:
            inheritable = [patient_id] if patient_id else _patients_with_a_corpus(inherit_from)
            _refuse_if_the_corpus_cannot_be_inherited(
                cfg, inherit_from, inheritable, requested_patient=patient_id,
            )

        patients = list(inheritable) if inherit_from else (
            [patient_id] if patient_id else patients_for_stage(name, cfg, run_root)
        )
        if not patients:
            raise click.ClickException(
                f"No patients to {name}.\n"
                "`junior status` shows which folder Junior is reading and how many "
                "patients it found there."
            )

        if name == "ingest" and patient_id is None and not _confirm_column_mapping(
            cfg, run_root, len(patients)
        ):
            click.echo("  cancelled — nothing was ingested")
            return

        if name == "embed" and not _confirm_embedding_model(
            cfg, run_root, len(patients), config_origin.rsplit(" (", 1)[0]
        ):
            click.echo("  cancelled — nothing was embedded")
            return

        if name == "extract":
            # Resolved for every caller, including a cluster task: a requested variable
            # with no recipe has to fail the same way everywhere. Only the confirmation
            # is skipped when --patient narrowed this to one task of a fan-out, where
            # the choice was made by whoever launched the array.
            planned = _planned_variables(cfg)
            if patient_id is None and not _confirm_variables(
                cfg, planned, len(patients), Path(config_origin.rsplit(" (", 1)[0]).name
            ):
                click.echo("  cancelled — nothing was extracted")
                return

        # Created only once the operator has said yes. Doing it earlier meant declining
        # a stage still left a run behind — a numbered, empty receipts folder that
        # `status` then reported and the next command carried on into, under a message
        # that had just said nothing was done.
        ensure_layout(cfg["run_id"])

        # After the confirmation and after the layout exists, so a declined command
        # leaves nothing: no run holding a corpus, and in particular no run holding a
        # corpus with no code/ — which the next `--new-run` would inherit from and then
        # refuse against for good.
        if inherit_from:
            _inherit_the_corpus_into_a_fresh_run(cfg, inherit_from, inheritable)

        code_lock_hash = _ensure_sealed_bundle(cfg=cfg, run_root=run_root)
        runner = STAGES[name]

        # Tee structured events into the run's own log, the way the cohort runner does.
        # Without this a stage driven from the CLI leaves no durable record — its
        # events live only in a terminal scrollback or a lost SLURM stdout.
        from jr_pipeline.runtime_infrastructure.json_event_logging import (
            clear_run_log_file,
            set_console_quiet,
            set_run_log_file,
        )
        set_run_log_file(run_root / "run_log.jsonl")
        # A stage logs one event per source file per patient. Nineteen tables across a
        # cohort buries the progress line the operator is actually watching, so the
        # terminal gets the summary and run_log.jsonl keeps every event.
        set_console_quiet(not verbose)
        if not verbose:
            _quiet_model_loading_chatter()

        failures: list[tuple[str, str]] = []
        # Variable-level outcomes, which `failures` above does not see: a patient whose
        # every variable failed still returns a summary and is counted as done.
        values_failed: Counter = Counter()
        values_attempted: Counter = Counter()
        try:
          with _confirm_before_stopping(name, len(patients)):
            for index, patient in enumerate(patients, 1):
                # Looping here means the position is known exactly; only a narrowed
                # single-patient call has to infer it from the run's state log.
                position = None if patient_id else (index, len(patients))
                try:
                    with _stage_monitor(name, patient, run_root, cfg.get("source_root"), position):
                        summary = runner(cfg, patient, code_lock_hash, force)
                except Exception as e:
                    # One patient's bad chart must not strand the rest of the cohort;
                    # collect and report at the end so the exit code is still honest.
                    # The exception class goes to the log, not the operator's screen.
                    failures.append((patient, f"{type(e).__name__}: {e}"))
                    click.secho(f"  ✗ {patient}: {e}", err=True, fg="red")
                    continue
                # The full summary lists every file with its row and column counts —
                # a screenful per patient. Say what it amounts to; -v has the rest,
                # and the receipts on disk have it whether or not anyone asked.
                if name == "extract":
                    # Counted across the cohort, because `failures` only holds patients
                    # whose extract RAISED. A patient whose every variable failed returns
                    # a summary like any other, so a run where nothing was read at all
                    # closed out clean and printed a tick.
                    for variable, outcome in (summary.get("variables") or {}).items():
                        values_attempted[variable] += 1
                        if not outcome.get("ok"):
                            values_failed[variable] += 1
                if verbose:
                    # A stage summary can carry extracted values and value-quoting
                    # error strings; the verbose dump goes to the same stdout the
                    # one-line result redacts, so it passes the same gate.
                    from jr_pipeline.runtime_infrastructure.cohort_runner import (
                        redact_summary_for_stdout,
                    )

                    click.echo(json.dumps(
                        redact_summary_for_stdout(summary), indent=2, sort_keys=True,
                    ))
                else:
                    click.secho(f"     {_one_line_result(name, summary)}", dim=True)
                    # Eighteen lines per patient is the detail a log wants and a
                    # terminal does not: on screen the ticker already says who and how
                    # far, and what an operator needs at the end is the failures, which
                    # _report_stage_finished gives once for the whole stage. Kept when
                    # piped, and -v still prints the whole summary.
                    if name == "extract" and not sys.stderr.isatty():
                        _report_extracted_variables(summary)
        finally:
            # Stop teeing, so a second command in the same process (the prompt, tests)
            # does not append to a stale run's log.
            clear_run_log_file()

        # Extraction is the last stage, so a whole-cohort extract closes the run out
        # rather than leaving that a command to remember. NOT when --patient narrowed
        # it: that caller is one task of a fan-out and the other patients are still
        # running, so summarizing would be premature and a sibling's bad artifact
        # would fail this task for something it did not do.
        # Said before the run is closed out, not after. `finish` ends on what to do
        # next, and guidance that arrives before the result it is about reads as though
        # it belongs to something else.
        if not failures:
            _report_stage_finished(name, len(patients), values_failed, values_attempted)

        closing_the_cohort = patient_id is None and not failures
        if name == "extract" and closing_the_cohort:
            ctx.invoke(finish, run_root=run_root)

        if failures:
            named = ", ".join(patient for patient, _ in failures)
            raise click.ClickException(
                f"{len(failures)} of {len(patients)} patient(s) failed {name}: {named}\n"
                f"The reason for each is above, and in run_log.jsonl under\n"
                f"  {run_root}\n"
                f"After fixing one, re-run just that patient:\n"
                f"  junior {name} --patient <id> --force"
            )

    return _cmd


# Every stage in STAGES gets a command. The numbered ones carry their position in
# their help; `retrieve` is dispatchable but is not a step in the flow, so it gets a
# plain description instead of a number it would then have to keep in sync.
_STAGE_POSITIONS = {name: position for position, name in enumerate(PIPELINE_STAGE_DESCRIPTIONS, 1)}
_UNNUMBERED_STAGE_HELP = {
    "retrieve": "Probe the retrieval subsystem. Extraction retrieves on its own.",
}
for _stage_name in STAGES:
    _position = _STAGE_POSITIONS.get(_stage_name)
    _common_step(
        _stage_name,
        f"Step {_position}: {PIPELINE_STAGE_DESCRIPTIONS[_stage_name]}."
        if _position
        else _UNNUMBERED_STAGE_HELP[_stage_name],
    )


# Asking is shared with the interactive prompt; see console_questions for the words
# that work at a question (`back`, `cancel`) and the never-block-when-piped rule.
_is_interactive = is_interactive
_ask_with_default = ask_one


def _things_each_stage_leaves(under_a_patient: bool) -> list[tuple[str, str]]:
    """Per stage, the things it writes, read off the guide a run carries in its own
    folder — so what is promised here and what that folder says cannot drift apart.

    Only the first segment of each path is kept, because the question being answered is
    which stage produced which pile, not what every file inside it is called. The guide
    itself answers that, and every run has a copy."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        RUN_ARTIFACT_GUIDE,
    )

    by_stage: dict[str, list[str]] = {}
    for stage_label, path_pattern, _ in RUN_ARTIFACT_GUIDE:
        is_a_patients_path = path_pattern.startswith("patients/")
        if is_a_patients_path != under_a_patient:
            continue
        if path_pattern.startswith("NO_PHI__shareable/"):
            continue  # named on its own below; it is the one tree that may be shared
        remainder = path_pattern.split("/", 2)[2] if is_a_patients_path else path_pattern
        head = remainder.split("/")[0]
        named = f"{head}/" if "/" in remainder else head
        leaves = by_stage.setdefault(stage_label, [])
        if named not in leaves:
            leaves.append(named)
    listed = []
    for stage_label, leaves in by_stage.items():
        shown = ", ".join(leaves[:3]) + (", …" if len(leaves) > 3 else "")
        listed.append((stage_label, shown))
    return listed


def _describe_layout(data_root: Path, input_folder: Path) -> None:
    """Show what this cohort will read and where each kind of output will land.

    Which directory holds what is the thing an operator actually has to decide — one
    of these trees is PHI and one is shareable, and the difference is the point."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        RUN_ARTIFACT_GUIDE_NAME,
    )

    click.echo()
    click.secho("  Reads from", bold=True)
    click.echo(f"    {input_folder}")
    click.secho("       one folder per patient; the folder name is the patient id", dim=True)
    click.echo()
    click.secho("  Writes to", bold=True)
    click.echo(f"    {data_root}")
    click.echo("      CONTAINS_PHI/pipeline_run_receipts/<run>/patients/<id>/")
    for stage_label, leaves in _things_each_stage_leaves(under_a_patient=True):
        click.secho(f"          {stage_label.ljust(10)}{leaves}", dim=True)
    click.echo("      CONTAINS_PHI/pipeline_run_receipts/<run>/")
    for _, leaves in _things_each_stage_leaves(under_a_patient=False):
        click.secho(f"          {'the run'.ljust(10)}{leaves}", dim=True)
    click.echo("      NO_PHI__shareable/<run>/")
    click.secho("          run-level metadata only — the tree you may share", dim=True)
    click.echo()
    click.secho(f"    Every run carries a {RUN_ARTIFACT_GUIDE_NAME} saying what each "
                f"of those is.", dim=True)


@main.command("columns",
              short_help="Draft this site's column map from your export's headers.")
@click.argument("patient_folder", required=False,
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output", "output", default=None,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Write the map here instead of asking. Printed to stdout if piped.")
@click.option("--edit", "edit", is_flag=True,
              help="Open this project's column map in your editor instead of drafting one.")
def columns(patient_folder: str, output: Path | None, edit: bool) -> None:
    """Draft this site's column map from one patient's file headers.

    \b
    Junior filters evidence by fixed names — document_date, author, doc_type,
    specialty ... — but your export names its columns whatever it names them, so each
    site maps them once. This reads the HEADER LINES ONLY of one patient's files
    (never a row of data), guesses what it recognizes, writes a map, and opens it so
    you can fill in what it could not match.

    \b
    Then point a run at it with `chart_columns_file:` in your config, and preflight
    the cohort — it refuses to run a patient with no free text to embed, and flags
    any metadata field it could not find a column for.

    \b
    Piped or redirected, it prints the map instead, so this still works:
      junior columns /path/to/cohort/PATIENT_1 > deployment/ucla/column_map.yaml

    \b
    `--edit` skips all of that and opens the map this project already uses, which is
    what a screen telling you to add a `text_columns:` line is asking you to do.
    """
    from jr_pipeline.runtime_infrastructure.column_map_drafting import (
        render,
        resolve_patient_folder,
    )

    if edit:
        _edit_this_projects_column_map()
        return

    # Most sites are already mapped. Ask which one this export is before asking a
    # single question about drafting a new map — reading headers to rediscover a
    # mapping that ships with the repo is work nobody needs to do.
    if output is None and _is_interactive():
        existing = _known_column_maps()
        if existing and not patient_folder:
            chosen = _choose_column_map(existing, _column_map_in_use())
            if chosen is None:
                click.echo("  cancelled")
                return
            if chosen != _DRAFT_A_NEW_MAP:
                _point_config_at_column_map(existing[chosen])
                return

    if not patient_folder:
        # Defaulted to the charts this project already declares. "." meant the project
        # folder, whose only subfolder is CONTAINS_PHI — Junior's own output — so taking
        # the default sent the drafter looking for an export inside the results.
        asked = ask_sequence([Question(
            "folder", "Which folder holds the chart files? (one patient is enough)",
            lambda _answers: _this_projects_charts() or ".",
        )])
        if asked is None:
            click.echo("  cancelled — nothing was written")
            return
        patient_folder = asked["folder"]
    folder = Path(patient_folder).expanduser()
    if not folder.is_dir():
        raise click.ClickException(
            f"{folder} is not a folder. Point this at your export — a cohort folder, "
            "or one patient's folder inside it."
        )
    read_from = resolve_patient_folder(folder)
    if read_from != folder:
        click.echo(f"# (no data files directly in {folder.name}; read {read_from.name})",
                   err=True)
    try:
        drafted = render(read_from)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e

    # Piped or redirected: print the map and stop, so `junior columns X > map.yaml`
    # keeps working and nothing blocks waiting for an answer.
    if output is None and not _is_interactive():
        click.echo(drafted, nl=False)
        return

    # At a terminal, a draft printed to the screen is not much use — it is a file you
    # have to finish by hand. So write it, then open it, rather than leaving the
    # operator to redirect the output and find an editor themselves.
    #
    # It goes in the PROJECT, beside its settings file. A map drafted into the checkout
    # instead becomes part of the codebase: a throwaway written while trying the tool out
    # is then offered in this menu for good, turns up as an untracked file in the repo,
    # and would be published with it — carrying one site's internal column names into
    # everybody else's copy. deployment/<site>/ is for maps that ship with Junior, which
    # is a decision to add one to the project, not something a draft does on its way past.
    if output is not None:
        destination = Path(output).expanduser()
    else:
        site = _ask_with_default(
            "Which institution is this export from? (used to name the mapping)",
            "my_site",
        ).strip().lower().replace(" ", "_")
        destination = _where_a_drafted_map_goes(site)

    if destination.exists() and _is_interactive():
        if not click.confirm(f"  {destination} exists. Replace it?", default=False):
            click.echo("  left alone")
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(drafted, encoding="utf-8")

    unresolved = drafted.count("# not found:")
    click.echo()
    click.secho(f"✓ {destination.name} is set up and ready — a run works as it is.",
                bold=True)
    if unresolved:
        click.secho(
            f"  {unresolved} table(s) carry a `# not found:` line: a chart field with no\n"
            f"  column behind it. Map one only when a recipe needs to filter on it.",
            dim=True,
        )
    _offer_to_open(destination)
    _point_config_at_column_map(destination)


# What a stage has and has not done, said at the end where the question arises, and
# the step that follows. Ingest is fast and produces tables, which reads like it might
# have been the whole job — so each stage states what it did NOT do as well.
_STAGE_COMPLETION: dict[str, tuple[str, str | None]] = {
    "ingest": ("the charts are saved as tables — nothing has been embedded, "
               "searched or interpreted yet", "embed"),
    "embed": ("every passage now has a vector, so it can be found by meaning", "index"),
    "index": ("each patient has a search index — the corpus is ready", "extract"),
    "extract": ("the variables have been read out, with the passage each came from", None),
    "retrieve": ("retrieval ran; nothing was written to the run", None),
}


def _confirm_column_mapping(cfg: dict, run_root: Path, patients: int) -> bool:
    """Show which column map this cohort will be read with, and get a yes before it sticks.

    Ingest is the point of no return for this choice, in the strong sense: it resolves
    each chart field against a file's real header and records the answer in that file's
    sidecar, and every later stage reads what was recorded. Embed does not revisit it —
    a field recorded as None is settled, not missing. So this is the last moment the
    answer is cheap, and unlike the model or the variable list it is a choice nobody
    made: a project inherits an unset map from the bundled settings.

    The default carries the recommendation. With a map, this is an ordinary confirmation
    and Enter takes it. Without one, Enter declines, because proceeding writes a cohort
    whose chart fields were guessed and cannot be corrected in place.

    Asked once per run — a re-run for a late-arriving patient must not re-ask, since a
    plain re-run cannot change the answer anyway (the ingest cache is keyed on whether
    the source files changed; `--force` is what rewrites a sidecar)."""
    interactive = is_interactive()

    from jr_pipeline.runtime_infrastructure.run_progress import read_stage_progress

    # `completed` is a COUNT of patients, not a flag. Treating it as "this run has been
    # ingested" meant one finished patient silenced the question for the whole cohort —
    # so `junior ingest --patient P1` to try one, then `junior ingest` for the rest, and
    # the gate was gone for every patient whose mapping had not been decided yet.
    done = read_stage_progress(run_root, "ingest", total=None).completed
    # Something already ingested means a mapping is already recorded. If the config has
    # changed since, a plain re-run reuses those patients, prints a tick, and applies
    # none of it — the one moment this question is worth most.
    drifted = _mapping_this_run_would_no_longer_produce(cfg, run_root) if done else []
    if not drifted and done >= patients:
        # Every patient about to be handled is already done and agrees with the config,
        # so nothing is left to settle and asking would imply it could change.
        # `completed` is a COUNT: comparing it against the patient count rather than
        # testing it for truthiness is what stops one finished patient silencing the
        # question for a whole cohort that has not been decided yet.
        return True
    if drifted:
        click.echo()
        click.secho("  This run was ingested with a different column mapping", bold=True,
                    fg="yellow")
        for line in drifted[:6]:
            click.secho(f"    {line}", dim=True)
        if len(drifted) > 6:
            click.secho(f"    and {len(drifted) - 6} more", dim=True)
        click.secho(
            "\n  Ingest reuses a patient whose chart files have not changed, and the "
            "mapping is not\n"
            "  part of that check — so running it now would report success and change "
            "none of the\n"
            "  above. `junior ingest --force` re-reads the charts and records the new "
            "mapping; a new\n"
            "  --run-id starts clean and leaves this run as it is.",
            fg="yellow",
        )
        click.echo()
    # Piped, on SLURM, or under the workbench's stage buttons: nothing there can
    # answer a question. What this stage is about to use is not a function of who
    # is watching, though, so it is printed above either way and only the QUESTION
    # is skipped. Bailing out before the display meant an operator clicking Embed
    # never saw which model was about to be locked into the run.
        if not interactive:
            return True
        return click.confirm("  Run anyway, reusing what is already there?", default=False)

    column_map = cfg.get("chart_columns_file")
    click.echo()
    click.secho("  About to ingest", bold=True)
    click.secho(f"    patients   {patients} from {cfg.get('source_root')}", dim=True)
    click.secho("    columns    ", nl=False, dim=True)
    if column_map:
        click.echo(Path(column_map).name)
    else:
        click.secho("no column map — document_date, author, doc_type and specialty are "
                    "guessed from generic names", fg="yellow")
        click.secho("               recorded per file now; changing it later needs "
                    "junior ingest --force", dim=True)
    click.secho(f"    writing    {run_root.name}/patients/<patient>/structured/", dim=True)
    click.echo()
    if column_map:
        if not interactive:
            return True
        return click.confirm(f"  Ingest {patients} patient(s)?", default=True)

    # Without a map, the question itself says so and names the way out. Carrying that
    # only in the default -- [y/N] rather than [Y/n] -- asks somebody to notice a
    # capital letter and infer a recommendation from it. Ingest is the point of no
    # return for this choice, so it is worth a sentence.
    click.secho("  No column map is set. `junior columns` writes one.", fg="yellow")
    if not interactive:
        return True
    return click.confirm(f"  Are you sure — ingest {patients} patient(s) without one?",
                         default=False)


def _confirm_embedding_model(
    cfg: dict, run_root: Path, patients: int,
    settings_file: str = "this project's settings file",
) -> bool:
    """Show what is about to be embedded and by what, then ask.

    Two settings decide the corpus here — which columns hold the text, and which model
    turns them into vectors — and each is named with the file it lives in, because a
    screen that shows a setting without saying where it lives leaves the reader hunting
    for a file.

    Embedding is the point of no return for this choice: every patient in a run must
    be embedded by the same model or their passages are not comparable, so the run
    locks it in. It is also the first slow step. Confirming here is the last moment
    the answer is still cheap to change.

    Asked once per run — re-running embed for a late-arriving patient must not
    re-ask, since the choice was already made and cannot differ."""
    encoder = cfg.get("encoder") or {}
    model_id = encoder.get("model_id")
    interactive = is_interactive()
    if not model_id:
        # No encoder at all. Returning True here meant embed started, loaded nothing,
        # and failed once per patient with a message about that patient — when the
        # problem is the project, and is the same for all of them.
        raise click.ClickException(
            "This project has no embedding model set, so there is nothing to turn the "
            "chart text into vectors with.\n"
            "Add an `encoder.model_id:` pointing at a local model folder in this "
            "project's settings file.\n"
            "A project made with `junior new-project` inherits one; a settings file "
            "written by hand needs it added."
        )

    from jr_pipeline.runtime_infrastructure.run_progress import read_stage_progress

    # A count, not a flag — see the note in _confirm_column_mapping. One embedded patient
    # used to silence this for every patient still waiting.
    if read_stage_progress(run_root, "embed", total=None).completed >= patients:
        return True  # this run already chose; asking again implies it could change

    model_path = Path(model_id)
    columns = _columns_that_will_be_embedded(cfg)

    click.echo()
    click.secho("  About to embed", bold=True)
    click.secho("    columns    ", nl=False, dim=True)
    map_name = Path(cfg["chart_columns_file"]).name if cfg.get("chart_columns_file") else None
    if columns is None:
        click.secho(f"unreadable map — {map_name}; embed refuses until it parses",
                    fg="yellow")
    elif columns:
        click.echo(", ".join(columns))
        click.secho(f"               named in {map_name}", dim=True)
    elif map_name:
        # A map IS set and marks nothing as text. Saying "no column map" here was false,
        # and sent the operator to add a setting that was already there.
        click.secho(f"none — {map_name} marks no column as text, so nothing is embedded",
                    fg="yellow")
        click.secho("               add a `text_columns:` line: junior columns --edit",
                    dim=True)
    else:
        text_column = cfg.get("text_column") or "text"
        would_embed = _text_columns_the_default_would_embed(cfg, run_root)
        click.echo(", ".join(would_embed) if would_embed
                   else f"the `{text_column}` column of every table that has one")
        click.secho(
            f"               no column map — only a column named `{text_column}` is "
            f"found; a table naming it anything else is skipped", fg="yellow")
        click.secho(f"               set chart_columns_file in {Path(settings_file).name}, "
                    f"or run junior columns", dim=True)

    click.secho("    model      ", nl=False, dim=True)
    click.echo(f"{model_path.name} · on this machine")
    if not model_path.is_dir():
        click.secho(f"               ⚠ {model_path.parent} does not exist — embed will fail",
                    fg="yellow")
    click.secho(f"               fixed for this run · change encoder.model_id in "
                f"{Path(settings_file).name}", dim=True)
    click.secho("    writing    ", nl=False, dim=True)
    click.echo(f"{run_root.name}/patients/<patient>/ — embeddings.npy + chunk_index.parquet")

    # Enter declines when no map names the text columns, the same way it does at ingest.
    # An unset map is not a choice anyone made — a project inherits it from the bundled
    # settings — and this is the last gate before the slow step, so the path of least
    # resistance must not be the one that embeds a guess.
    settled_columns = bool(columns)
    # Piped, on SLURM, or under the workbench's stage buttons: nothing there can
    # answer a question. What this stage is about to use is not a function of who
    # is watching, though, so it is printed above either way and only the QUESTION
    # is skipped. Bailing out before the display meant an operator clicking Embed
    # never saw which model was about to be locked into the run.
    if not interactive:
        return True
    return click.confirm(f"  Embed {patients} patient(s)?", default=settled_columns)


def _planned_variables(cfg: dict) -> Any:
    """Resolve this project's `recipes:` list into the recipes extraction will run.

    Checked here, before anything is written, because the planner returns a name it
    could not find on disk in its `missing` list and carries on. That is right for a
    dependency — the recipe that wanted it gets a null upstream value — and wrong for a
    name the operator asked for: a typo in `recipes:` produced a run that reported
    success with the column simply absent, which is the same "we never looked" reading
    as a null value. Requested names must exist."""
    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_execution_order import plan

    requested = list(cfg.get("recipes") or [])
    if not requested:
        raise click.ClickException(
            "This project does not say which variables to extract.\n"
            "Add a `recipes:` list to its junior.yaml — `junior recipes` shows the "
            "names you can use."
        )
    recipes_root = _resolve_recipes_root(cfg, _resolve_repo_root())
    try:
        resolved = plan(recipes_root=recipes_root, requested=requested)
    except Exception as e:
        raise click.ClickException(
            f"One of this project's variable recipes could not be read.\n"
            f"  {e}\n"
            f"They live under {recipes_root}."
        ) from e
    unknown = [name for name in requested if name in resolved.missing]
    if unknown:
        raise click.ClickException(
            f"No recipe for {', '.join(unknown)} under {recipes_root}.\n"
            "Each variable is a folder there. `junior recipes` lists the ones that "
            f"exist; correct the `recipes:` list in {_this_projects_settings_name()}."
        )
    return resolved


def _say_what_running_it_here_means(extras: dict) -> None:
    """Spell out that this model runs on this machine, and what that costs.

    A remote endpoint sends the charts somewhere with a signed agreement behind it and
    answers with an error when a prompt is too big. A local one loads the weights into
    this process, on this machine's memory, and an oversized prompt does not return an
    error — Metal refuses the allocation and the process dies with nothing in the log.

    So the ceiling that prevents that is worth showing before a run, together with the
    hardware it was measured on, because a number with no provenance is one nobody can
    safely raise. Whoever moves this model to a bigger machine needs to know that 8000
    was a laptop's answer and not the model's — the model's own context window is four
    times that."""
    cap = extras.get("max_prompt_tokens_cap")
    click.secho("    memory     ", nl=False, dim=True)
    here = _how_much_memory_this_machine_has()
    on_this_machine = f"on this machine{f' ({here})' if here else ''}"
    if not cap:
        click.secho(f"{on_this_machine} · no ceiling set — an oversized prompt kills "
                    f"the run", fg="yellow")
        return
    # `measured_on` is the operator's own note about their hardware, if they wrote one.
    # Nothing ships with a value: a number measured on somebody else's laptop, printed
    # as fact to whoever cloned the repo, is worse than no number.
    measured = extras.get("measured_on")
    where = f" · {measured}" if measured else ""
    click.echo(f"{on_this_machine}{where} · refuses over {cap} prompt tokens")


def _how_much_memory_this_machine_has() -> str:
    """Installed RAM, read at runtime, or "" if it cannot be determined.

    Read rather than configured, because the alternative was a shipped file describing
    the machine it happened to be measured on — true for one person and wrong for
    everyone who cloned the repo."""
    import os

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return ""
    return f"{round(total / 2**30)} GiB" if total > 0 else ""


def _describe_the_model_that_reads_the_chart(cfg: dict) -> None:
    """Name the language model extraction will actually call, not the file listing it.

    Naming only the allow-list left the one choice that most determines the values
    unstated, and the two shipped allow-lists differ in a way that matters: one loads a
    3B model from a folder on this machine, the other names a model id on a public hub
    and opts into downloading it. An operator confirming a run on a machine holding
    patient data has to be told which of those is about to happen."""
    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_allowlist import (  # noqa: E501
        AllowlistError,
        load_allowlist,
    )

    configured = cfg.get("allowlist_path")
    if not configured:
        click.secho("none set — the variables above will fail", fg="yellow")
        return
    try:
        allowlist = load_allowlist(configured)
    except AllowlistError as e:
        click.secho(f"⚠ {e}", fg="yellow")
        return
    except Exception:
        # This describes; it does not decide. Extraction refuses on its own when a
        # variable needs a model and none can be built, so a screen that cannot be
        # drawn must not be what stops a cohort — it would take the run down for a
        # sentence nobody depends on.
        click.secho(f"listed in {Path(configured).name}", dim=True)
        return

    wanted = cfg.get("model_override")
    endpoints = [
        endpoint for endpoint in allowlist.endpoints()
        if not wanted or endpoint.name == wanted
    ]
    if not endpoints:
        click.secho(f"⚠ {Path(configured).name} lists no endpoint named {wanted}",
                    fg="yellow")
        return
    for position, endpoint in enumerate(endpoints):
        target = endpoint.default_model or endpoint.url
        # The folder name, not the path to it: the path wraps the terminal and which
        # model it is is the part being confirmed. A hub id is left whole — it looks
        # path-shaped but its first segment is the publisher, and dropping that leaves
        # a name that no longer says whose model this is.
        looks_like_a_path_on_disk = target.startswith(("/", "./", "../", "~"))
        shown = Path(target).name if looks_like_a_path_on_disk else target
        line = f"{endpoint.name} — {shown}"
        if position == 0:
            click.echo(line, nl=False)
            click.secho(f"    change in {Path(configured).name}", dim=True)
        else:
            # An allow-list may hold several endpoints and each recipe names its own
            # (`llm.model`), so every candidate is shown — an early return here left
            # every endpoint after the first unprinted, and the screen could name a
            # model no planned recipe was going to call.
            click.echo(f"               {line}")
        extras = endpoint.extras or {}
        if extras.get("allow_download"):
            click.secho("               ⚠ downloads its weights from a public hub the "
                        "first time it runs", fg="yellow")
        elif endpoint.provider == "local_hf" and not Path(target).is_dir():
            click.secho(f"               ⚠ {target} is not there, so these will fail",
                        fg="yellow")
        if endpoint.provider == "local_hf":
            _say_what_running_it_here_means(extras)


def _mapping_this_run_would_no_longer_produce(cfg: dict, run_root: Path) -> list[str]:
    """Fields whose column has changed since this run was ingested, as readable lines.

    Ingest records the mapping it resolved beside each table, and the cache is keyed on
    whether the SOURCE FILES changed — not on the mapping. So pointing a project at a
    column map and re-running ingest reuses every patient, reports a tick, and applies
    nothing. The gap between what the config says now and what the run recorded is the
    only evidence that happened, and it is on disk.

    Empty when they agree, or when there is nothing recorded to compare against."""
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
        resolve_metadata_columns_for,
    )

    patients_dir = run_root / "patients"
    if not patients_dir.is_dir():
        return []
    for patient in sorted(p for p in patients_dir.iterdir() if p.is_dir()):
        # One patient is enough: an export has the same shape for every patient, and a
        # cohort-wide scan to say "the map changed" costs a lot to say the same thing.
        sidecars = sorted((patient / "structured").glob("*.parquet.meta.json"))
        if not sidecars:
            continue
        drifted: list[str] = []
        compared = False
        for sidecar in sidecars:
            stem = sidecar.name.split(".parquet")[0]
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8")).get("payload") or {}
            except (OSError, ValueError):
                continue
            recorded = payload.get("metadata_columns")
            columns = [column.get("name") for column in payload.get("columns") or []]
            if not isinstance(recorded, dict) or not columns:
                continue
            compared = True
            would_be = resolve_metadata_columns_for(columns, cfg, stem)
            for field_name, column in sorted(would_be.items()):
                was = recorded.get(field_name)
                if column != was:
                    drifted.append(
                        f"{stem}.{field_name}: recorded {was or 'nothing'}, "
                        f"this map says {column or 'nothing'}"
                    )
        # Only stop once a patient has actually yielded a comparison. Returning after
        # the first patient that merely HAD sidecars reported "no drift" whenever that
        # one was unreadable, or came from a run predating recorded mappings — and "no
        # drift" is what the caller takes as permission to stay quiet.
        if compared:
            return drifted
    return []


def _variables_this_run_already_has(cfg: dict) -> list[str] | None:
    """Variables this run has already written results for, or None if it has none yet.

    Read off the run rather than from the config, because that is the question an edited
    settings file raises: the config says what you want extracted, the run says what was.
    None and [] are different answers — nothing extracted yet is not the same as nothing
    to compare against — so a fresh run says nothing rather than listing everything as
    new."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    try:
        run_root = phi_intermediate_run_dir(str(cfg["run_id"]))
    except (KeyError, ValueError, OSError):
        return None
    patients_dir = run_root / "patients"
    if not patients_dir.is_dir():
        return None
    found: set[str] = set()
    for patient in sorted(p for p in patients_dir.iterdir() if p.is_dir()):
        extract_dir = patient / "extract"
        if not extract_dir.is_dir():
            continue
        found.update(
            variable.name for variable in extract_dir.iterdir()
            if variable.is_dir() and (variable / "result.json").is_file()
        )
    return sorted(found) if found else None


def _text_columns_the_default_would_embed(cfg: dict, run_root: Path) -> list[str] | None:
    """What gets embedded when no map names the text columns, read off the ingested run.

    Not a guess: with no map, embed takes the column named by ``text_column`` (default
    ``text``) from every ingested table that has one. Ingest already recorded each
    table's columns in its sidecar, so the answer is on disk and can be shown rather
    than described. None when nothing has been ingested yet — then there is nothing to
    read and the caller has to speak generally."""
    text_column = cfg.get("text_column") or "text"
    patients_dir = run_root / "patients"
    if not patients_dir.is_dir():
        return None
    # One patient is enough: an export has the same shape for every patient, which is
    # the same assumption `junior columns` makes when it drafts from one folder.
    for patient in sorted(p for p in patients_dir.iterdir() if p.is_dir()):
        sidecars = sorted((patient / "structured").glob("*.parquet.meta.json"))
        if not sidecars:
            continue
        found = []
        for sidecar in sidecars:
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8")).get("payload", {})
            except (OSError, ValueError):
                continue
            names = [column.get("name") for column in payload.get("columns") or []]
            if text_column in names:
                found.append(f"{sidecar.name.split('.parquet')[0]}.{text_column}")
        return sorted(found)
    return None


def _confirm_variables(
    cfg: dict, resolved: Any, patients: int, settings_file: str = "this project's settings"
) -> bool:
    """Show what is about to be extracted, from what, using what — then ask.

    Three facts, because three are what an operator can check at a glance: the variables,
    where they are coming from, and which model will read the charts. Each says where to
    change it, since a screen that shows a setting without saying where it lives leaves
    the reader hunting for a file."""
    interactive = is_interactive()

    from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import (
        _recipe_needs_a_model,
    )

    requested = list(cfg.get("recipes") or [])
    pulled_in = [recipe.name for recipe in resolved.recipes if recipe.name not in requested]
    needing_model = [
        recipe.name for recipe in resolved.recipes if _recipe_needs_a_model(recipe)
    ]

    click.echo()
    click.secho("  About to extract", bold=True)

    named = ", ".join(requested)
    if pulled_in:
        named += f" (+ {', '.join(pulled_in)}, needed by them)"
    click.secho("    variables  ", nl=False, dim=True)
    click.echo(named, nl=False)
    click.secho(f"    change in {settings_file}", dim=True)
    if resolved.missing:
        # Not the requested names — those already refused above. These are recipes the
        # requested ones depend on, and a null upstream value changes the answer.
        click.secho(f"               ⚠ {', '.join(resolved.missing)} has no recipe, so "
                    f"whatever depends on it reads as empty", fg="yellow")

    click.secho("    source     ", nl=False, dim=True)
    click.echo(f"{patients} patient(s) in run {cfg.get('run_id')}")

    # Which of these this run already answered, and which are new since. A settings file
    # edited between runs — or while one was in flight, which reads its config once at
    # the start — otherwise leaves "did my edit take effect" answerable only by going and
    # looking at the output folder.
    already = _variables_this_run_already_has(cfg)
    if already is not None:
        planned = [recipe.name for recipe in resolved.recipes]
        fresh = [name for name in planned if name not in already]
        stale = [name for name in already if name not in planned]
        if fresh and already:
            click.secho(f"               {len(already)} already extracted in this run; "
                        f"new here: {', '.join(fresh)}", dim=True)
        elif not fresh and already:
            click.secho("               all of these were already extracted in this run — "
                        "re-running skips them; --force re-extracts", dim=True)
        if stale:
            click.secho(f"               {', '.join(stale)} was extracted here but is no "
                        f"longer in your settings; its values stay as they are", dim=True)

    # How much chart text each question gets to see. This is the setting that decides
    # whether a run finishes on this machine, and it was the one thing the screen did
    # not say — it showed the endpoint's token ceiling, which is the backstop, not the
    # budget anyone actually tunes.
    ceiling = cfg.get("max_chunks_per_prompt")
    asked_for = max(
        (int(getattr(recipe.reranking, "top_n", 0)) for recipe in resolved.recipes),
        default=0,
    )
    click.secho("    evidence   ", nl=False, dim=True)
    if ceiling and asked_for > ceiling:
        click.echo(f"{ceiling} chunks per prompt", nl=False)
        click.secho(f" · these recipes ask for up to {asked_for}, capped here for this "
                    f"machine    change in {settings_file}", dim=True)
    elif ceiling:
        click.echo(f"{ceiling} chunks per prompt", nl=False)
        click.secho(f"    change in {settings_file}", dim=True)
    else:
        click.echo(f"up to {asked_for} chunks per prompt", nl=False)
        click.secho(" · no machine ceiling set (max_chunks_per_prompt)", fg="yellow")

    click.secho("    model      ", nl=False, dim=True)
    if needing_model:
        _describe_the_model_that_reads_the_chart(cfg)
    else:
        click.echo("none — every recipe here reads a table directly")

    # Piped, on SLURM, or under the workbench's stage buttons: nothing there can
    # answer a question. What this stage is about to use is not a function of who
    # is watching, though, so it is printed above either way and only the QUESTION
    # is skipped. Bailing out before the display meant an operator clicking Embed
    # never saw which model was about to be locked into the run.
    if not interactive:
        return True
    return click.confirm(
        f"\n  Extract {len(requested)} variable(s) from {patients} patient(s)?", default=True
    )


def _columns_that_will_be_embedded(cfg: dict) -> list[str] | None:
    """The `table.column` pairs this project's column map marks as text.

    Empty when no map is set — embed then detects a text column per file, which is a
    different and less predictable thing, so the caller says so rather than implying
    a fixed list exists.

    None when a map IS set and could not be read. Collapsing that into "no map set"
    told the operator, on the one screen that exists to state the truth before the
    point of no return, that a run would fall back to per-file detection when in fact
    the run will refuse to start."""
    map_file = cfg.get("chart_columns_file")
    if not map_file:
        return []
    try:
        parsed = yaml.safe_load(Path(map_file).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    mapping = parsed.get("chunk_metadata_columns")
    if not isinstance(mapping, dict):
        return None
    return [
        f"{source}.{column}"
        for source, fields in sorted(mapping.items())
        if isinstance(fields, dict)
        for column in (fields.get("text_columns") or [])
    ]


def _report_stage_finished(
    stage: str, patients: int,
    values_failed: Counter | None = None, values_attempted: Counter | None = None,
) -> None:
    """Close a stage by saying what it did, what it did not, and what comes next.

    "What it did not" includes values that failed. Those never reached the stage's own
    failure list — that holds only patients whose run raised — so a cohort where every
    variable failed for eight of nine patients printed a tick and "the variables have
    been read out", which was not true of 24 of the 27 values."""
    described, follows = _STAGE_COMPLETION.get(stage, ("done", None))
    # Normalized once rather than `or {}` at each use: an empty dict has no most_common,
    # which the type checker is right to object to even though `failed` guards the call.
    how_many_failed: Counter = values_failed if values_failed is not None else Counter()
    how_many_tried: Counter = values_attempted if values_attempted is not None else Counter()
    failed = sum(how_many_failed.values())
    attempted = sum(how_many_tried.values())
    click.echo()
    if failed:
        click.secho(f"⚠ {stage} finished, but {failed} of {attempted} value(s) failed",
                    bold=True, fg="yellow")
        worst = ", ".join(
            f"{variable}: {count} of {how_many_tried[variable]}"
            for variable, count in how_many_failed.most_common(4)
        )
        click.secho(f"  {worst}", dim=True)
        click.secho(
            "  A value that failed was not read at all — that is not the same as a "
            "chart having no answer.\n"
            "  The reason for each is in run_log.jsonl. `junior extract` retries just "
            "these; the values that succeeded are left alone.\n"
            "  A value that fails the same way twice will keep doing so — extraction is "
            "deterministic,\n"
            "  so the same evidence gives the same answer. Change the recipe and extract "
            "again.",
            dim=True,
        )
        if follows:
            position = list(PIPELINE_STAGE_DESCRIPTIONS).index(follows) + 1
            click.secho("  Next", bold=True)
            click.echo(f"    ·  junior {follows}"
                       f"{' ' * max(1, 12 - len(follows))}"
                       f"{PIPELINE_STAGE_DESCRIPTIONS[follows]}  (step {position})")
        return
    click.secho(f"✓ {stage} finished — {patients} patient(s)", bold=True)
    click.secho(f"  {described}", dim=True)
    if follows:
        position = list(PIPELINE_STAGE_DESCRIPTIONS).index(follows) + 1
        click.secho("  Next", bold=True)
        click.echo(f"    ·  junior {follows}"
                   f"{' ' * max(1, 12 - len(follows))}"
                   f"{PIPELINE_STAGE_DESCRIPTIONS[follows]}  (step {position})")
    elif stage == "extract":
        click.secho("  Next", bold=True)
        click.echo("    ·  junior workbench    look at the values and their evidence")


def _what_next_after_columns() -> None:
    """End the command by saying what to do next, the way new-project does.

    A command that finishes into a bare prompt leaves the operator to remember where
    they were in a sequence they have run once."""
    click.echo()
    click.secho("  Next", bold=True)
    click.echo("    ·  junior ingest       read the charts in  (step 1)")
    click.echo("    ·  junior recipes      choose what to extract")
    click.secho("    ·  or `help` for the full list", dim=True)


def _this_projects_charts() -> str:
    """This project's declared source_root, for use as a default answer."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
        if choice.reason == "bundled default":
            return ""
        source = (load_config(choice.path) or {}).get("source_root")
    except (FileNotFoundError, ValueError, OSError):
        return ""
    return str(source) if source and Path(str(source)).is_dir() else ""


def _this_projects_settings_name() -> str:
    """What to call this project's settings file on screen: its real name if there is
    one, a generic phrase if there is no project. `new-project` names the file after the
    project, so "junior.yaml" is a file most projects do not have."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
    except FileNotFoundError:
        return "this project's settings file"
    if choice.reason == "bundled default":
        return "this project's settings file"
    return choice.path.name


# What each stage needs to exist before it starts, as (config key, what it is, how to
# fix it). A stage reads these once for the whole cohort, so a missing one is a fault in
# the project, not in a patient.
_WHAT_EACH_STAGE_NEEDS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "ingest": (
        ("chart_columns_file", "column map",
         "`junior columns` points this project at one that exists. Deleting that line "
         "falls back to generic column names."),
    ),
    "embed": (
        ("chart_columns_file", "column map",
         "`junior columns` points this project at one that exists. Deleting that line "
         "falls back to generic column names."),
        ("encoder.model_id", "embedding model folder",
         "Point `encoder.model_id:` at a local model folder. Embed never downloads one."),
    ),
    "retrieve": (
        ("encoder.model_id", "embedding model folder",
         "Point `encoder.model_id:` at a local model folder. Embed never downloads one."),
    ),
    "extract": (
        ("recipes_root", "recipes folder",
         "Point `recipes_root:` at the folder holding one folder per variable."),
        ("allowlist_path", "LLM allow-list",
         "Point `allowlist_path:` at the file naming the endpoints this project may use."),
    ),
}


@contextmanager
def _confirm_before_stopping(stage: str, patients: int):
    """Ctrl-C asks before it throws away the work in flight.

    Stopping a stage has to stay possible — a cohort can be hours, and finding out the
    settings were wrong two patients in should not mean waiting it out. But Ctrl-C is
    also what a hand reaches for reflexively, and it is one keystroke away from
    discarding an afternoon of GPU time, so it asks first.

    Installed only where somebody is at a keyboard to answer. A SLURM task, a piped run
    or the cohort runner must die on SIGINT immediately and without a question, because
    there is nobody to read it and the scheduler is waiting.

    Answering the question is itself interruptible: the default handler is restored
    while it is up, so a second Ctrl-C stops the run the old way rather than recursing
    into another question."""
    if not is_interactive():
        yield
        return
    import signal

    def _ask(_signum, _frame):
        previous_during_question = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            # Whatever was typed at the progress line is in the buffer and would answer
            # this question for you. It gets thrown away first — see the prompt's own
            # discard, for the same reason.
            _forget_type_ahead()
            click.echo(err=True)
            stop = click.confirm(
                f"  Stop {stage}? {patients} patient(s) in this run", default=False
            )
        finally:
            signal.signal(signal.SIGINT, previous_during_question)
        if stop:
            raise KeyboardInterrupt
        click.secho("  carrying on", err=True, dim=True)

    previous = signal.signal(signal.SIGINT, _ask)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _forget_type_ahead() -> None:
    """Drop terminal input typed while something else was running. See the prompt's
    copy of this: keystrokes during a long stage are still buffered afterwards, and the
    next thing to read stdin takes them as an answer."""
    try:
        import sys
        import termios
    except ImportError:
        return
    try:
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
    except (termios.error, OSError, ValueError):
        return


def _refuse_if_this_project_points_at_nothing(
    cfg: dict, config_origin: str, stage: str
) -> None:
    """Stop before touching patients when a setting names something that is not there.

    Everything a stage reads once for the whole cohort — the column map, the model, the
    recipes, the allow-list — fails identically for every patient when it is missing. But
    the per-patient loop wraps each one in a try/except and reports "skipping <patient>",
    or "1 of 500 patient(s) failed", with a remedy of re-running that patient. So one
    deleted line in a settings file arrived as a cohort's worth of blame on the patients,
    and the advice on screen could not fix it.

    Checked once, here, so it is said once and names the setting. Deliberately not in
    load_cohort_config: `status` must still describe a project whose model has moved
    rather than refuse to speak about it. Reporting is what status is for; refusing is
    what running is for."""
    settings_name = Path(config_origin.rsplit(" (", 1)[0]).name
    for key, what_it_is, how_to_fix in _WHAT_EACH_STAGE_NEEDS.get(stage, ()):
        named: Any = cfg
        for part in key.split("."):
            named = named.get(part) if isinstance(named, dict) else None
        if not named:
            continue  # unset is a different question, and other screens ask it
        there = Path(str(named))
        if there.exists():
            continue
        raise click.ClickException(
            f"This project's {what_it_is} is not there:\n"
            f"  {named}\n"
            f"  named by `{key.split('.')[-1]}:` in {settings_name}\n"
            f"Every patient would fail on it the same way, so nothing ran.\n"
            f"{how_to_fix}"
        )


def _column_map_in_use() -> Path | None:
    """The map this project reads its charts with right now, if it names one."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
        if choice.reason == "bundled default":
            return None
        named = (load_config(choice.path) or {}).get("chart_columns_file")
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not named:
        return None
    resolved = _resolve_column_map(str(named), choice.path)
    return resolved if resolved.is_file() else None


def _resolve_column_map(named: str, config_path: Path) -> Path:
    """Where a `chart_columns_file:` value actually points, for a given settings file.

    The same order `load_cohort_config` uses, in one place rather than three: a bare
    relative name means the file beside the settings file, because that is where
    `junior columns` writes one and it survives the project folder being moved; anything
    else anchors to the checkout, which is how the shipped deployment/<site>/ maps are
    spelled. Re-deriving this with `_anchor_to_repo` alone made `junior columns --edit`
    refuse a map that ingest was reading happily, and print a remedy that would have
    broken a config that was already correct."""
    expanded = Path(named).expanduser()
    beside_the_config = config_path.parent / expanded
    if not expanded.is_absolute() and beside_the_config.is_file():
        return beside_the_config.resolve()
    return Path(_anchor_to_repo(named, _resolve_repo_root()))


def _where_a_drafted_map_goes(site: str) -> Path:
    """Beside this project's settings file, or the current folder if there is no project.

    Never inside the checkout. A map belongs to the cohort that uses it — which is also
    what `load_cohort_config` assumes when it resolves a bare `chart_columns_file:` name
    against the config's own folder before anywhere else."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    name = f"{site}_column_map.yaml"
    try:
        choice = find_config()
    except FileNotFoundError:
        return Path.cwd() / name
    if choice.reason == "bundled default":
        return Path.cwd() / name
    return choice.path.parent / name


def _edit_this_projects_column_map() -> None:
    """Open the map this project already points at, so `text_columns:` can be added.

    Every screen that says "name its text column under that table" is asking for an edit
    to a file whose path the operator then has to go and find. This is that step."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
        settings = load_config(choice.path)
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    named = settings.get("chart_columns_file")
    if not named:
        raise click.ClickException(
            f"{choice.path.name} does not point at a column map yet, so there is "
            f"nothing to open.\n"
            "Run `junior columns` to pick one or draft one, then `junior columns "
            "--edit` to change it."
        )
    map_path = _resolve_column_map(str(named), choice.path)
    if not map_path.is_file():
        raise click.ClickException(
            f"{choice.path.name} names a column map that is not there:\n  {map_path}\n"
            "Fix `chart_columns_file:` in that settings file, or run `junior columns` "
            "to make a new map."
        )

    # A map inside the checkout is shared by every project that points at it, and it is
    # a tracked file — editing it in place changes other people's cohorts and shows up
    # as a repo diff. Copying it into the project first is almost always what was meant,
    # so that is offered before the editor opens, not explained afterwards.
    if _is_interactive() and _resolve_repo_root() in map_path.parents:
        project_copy = choice.path.parent / map_path.name
        click.secho(
            f"  {map_path.name} ships with Junior and is shared by every project that "
            f"points at it.",
            fg="yellow",
        )
        if not project_copy.exists() and click.confirm(
            f"  Copy it into {choice.path.parent.name} and edit that instead?",
            default=True,
        ):
            import shutil

            shutil.copy2(map_path, project_copy)
            _set_one_setting(choice.path, "chart_columns_file", str(project_copy))
            click.secho(f"✓ {choice.path.name} now uses your own copy", bold=True)
            click.secho(f"  {project_copy}", dim=True)
            map_path = project_copy

    click.echo()
    click.secho(f"  {map_path}", bold=True)
    click.secho(
        "  Each table is a block. `text_columns:` under a table is what makes it "
        "searchable by\n"
        "  meaning; the other keys map that table's date, author and so on. Re-run "
        "`junior embed`\n"
        "  after editing — and `junior ingest --force` too, if you changed anything "
        "but text_columns.",
        dim=True,
    )
    _offer_to_open(map_path)


def _set_one_setting(config_path: Path, key: str, value: str) -> None:
    """Change one top-level key in a settings file, leaving the rest of the text alone.

    Splicing rather than re-dumping, because a settings file is not just its values. It
    carries the operator's own comments — beside a key, indented inside a block, written
    while working something out — and it may carry `${oc.env:...}` interpolations that
    let one file serve a laptop and a cluster. Loading it and dumping the result back
    keeps only the comments that begin a line, hoists them to the top where they annotate
    the wrong key, and resolves every interpolation to whatever this machine happened to
    have in its environment. That is a one-way loss of a file people are invited to edit
    freely, performed by a command that only meant to record one path."""
    written = yaml.safe_dump({key: value}, default_flow_style=False).strip()
    lines = config_path.read_text(encoding="utf-8").splitlines()
    for position, line in enumerate(lines):
        if not line.startswith((" ", "\t", "#")) and line.split(":", 1)[0].strip() == key:
            lines[position] = written
            break
    else:
        lines.append(written)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _offer_to_open(path: Path) -> None:
    """Open a file for editing only in a way that cannot surprise the operator.

    Handing a path to the system's default application launches whatever is
    registered for .yaml — a spreadsheet suite, on at least one machine, which then
    crashed. A CLI has no business launching a GUI the operator did not ask for, and
    a file it may have open elsewhere is one Junior is about to rewrite.

    So: honour $EDITOR / $VISUAL, because that is the operator saying which editor
    they want. Failing that, give them a line to paste AND say where to paste it. A
    path on its own leaves "open it how?" as the reader's problem; and the rule against
    printing a command that cannot be run where it is read is answered by naming the
    window it CAN be run in, not by withholding the command."""
    import os
    import shlex
    import sys

    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if editor and _is_interactive() and click.confirm(f"  Open it in {editor}?", default=True):
        try:
            click.edit(filename=str(path))
            return
        except click.ClickException as e:
            click.secho(f"  (could not open {editor}: {e})", err=True, dim=True)

    # `open -e` pins TextEdit rather than whatever is registered for .yaml, which is the
    # part that misbehaved; nano is the editor that is there on every other machine.
    quoted = shlex.quote(str(path))
    command = f"open -e {quoted}" if sys.platform == "darwin" else f"nano {quoted}"

    click.secho(
        "  To edit it, paste this in a second terminal (this one is running Junior):"
        if _in_interactive_session()
        else "  To edit it:",
        dim=True,
    )
    click.echo(f"    {command}")


_DRAFT_A_NEW_MAP = "__draft__"

# Set by the interactive prompt while it is running. Commands consult it to decide
# whether "cd somewhere" is advice the reader can act on — inside the prompt it is not.
_INTERACTIVE_SESSION = False


def set_interactive_session(active: bool) -> None:
    """Tell commands they are being run from the prompt rather than a shell."""
    global _INTERACTIVE_SESSION
    _INTERACTIVE_SESSION = active


def _in_interactive_session() -> bool:
    return _INTERACTIVE_SESSION


def _is_a_column_map(candidate: Path) -> bool:
    """Whether this file is a column map, by content rather than by filename.

    Naming a file is a convention people break; what makes it a map is what is inside
    it. Parsed rather than grepped — the bundled laptop.yaml documents the key in a
    comment, and a substring match offered that as a mapping."""
    try:
        parsed = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get("chunk_metadata_columns"), dict)


def _known_column_maps() -> dict[str, Path]:
    """The maps worth offering: the ones Junior ships, and the ones cohorts here use.

    Two sources, because those are the two ways a map earns a place on a menu. A map
    under deployment/<site>/ ships with the code and is meant to be reused. A map some
    project on this machine points at is proven — a real cohort was read with it.

    Nothing else counts. Offering every yaml that happens to sit in the checkout meant a
    map drafted once while trying the tool out was proposed for good, long after its
    author deleted what they thought was the only copy."""
    found: dict[str, Path] = {}
    for candidate in sorted((_resolve_repo_root() / "deployment").glob("*/*.y*ml")):
        if _is_a_column_map(candidate):
            found.setdefault(candidate.parent.name, candidate)

    from apps_and_interfaces.project_registry import project_name, remembered_projects

    for config_path in remembered_projects():
        try:
            named = (load_config(config_path) or {}).get("chart_columns_file")
        except (ValueError, OSError):
            continue
        if not named:
            continue
        in_use = _resolve_column_map(str(named), config_path)
        if not in_use.is_file() or in_use in found.values() or not _is_a_column_map(in_use):
            continue
        # Keyed by the cohort using it, since a project-local map has no site folder to
        # be named after and "which mapping matches your export" is answered as well by
        # "the one breast2026 uses" as by an institution's name.
        found.setdefault(project_name(config_path), in_use)

    # Maps sitting in a project's folder but not currently pointed at. Without these,
    # switching a project to a different map orphaned its own file: still on disk, gone
    # from this menu, and reachable again only by hand-editing the settings — which is
    # the one repair the people this menu exists for cannot make.
    for config_path in remembered_projects():
        for candidate in sorted(config_path.parent.glob("*.y*ml")):
            if candidate in found.values() or candidate == config_path:
                continue
            if _is_a_column_map(candidate):
                found.setdefault(f"{project_name(config_path)}/{candidate.stem}", candidate)
    return found


def _choose_column_map(existing: dict[str, Path], in_use: Path | None = None) -> str | None:
    """Ask which site's mapping this export uses. Returns a key, or None to cancel.

    The one already in use is marked and is what Enter takes. Defaulting to the first
    entry instead meant re-running this command and pressing Enter — which is what
    re-running a command usually costs — silently swapped a project's own map for
    another institution's, reported a tick, and cut what gets embedded. There is no
    undo, and the replaced map then dropped off this menu, so the way back was to
    hand-edit the settings file."""
    click.echo()
    click.secho("  Which mapping matches your export?", bold=True)
    sites = list(existing)
    default = "1"
    for position, site in enumerate(sites, 1):
        current = in_use is not None and existing[site] == in_use
        if current:
            default = str(position)
        click.echo(f"    {position}. {site}" + ("   ← in use now" if current else ""))
        click.secho(f"       {existing[site].name}", dim=True)
    click.echo(f"    {len(sites) + 1}. none of these — read my export's headers "
               f"and draft a new one")

    answer = ask_one("Pick one", default).strip()
    # `cancel` works at every other question, so it must not be reported here as an
    # invalid choice and then honoured anyway — two opposite verdicts in a row.
    if answer.lower() in CANCEL_WORDS or answer.lower() in BACK_WORDS:
        return None
    if not answer.isdigit() or not 1 <= int(answer) <= len(sites) + 1:
        click.secho(f"  '{answer}' is not one of the choices — pick 1 to {len(sites) + 1}.",
                    fg="yellow")
        return None
    picked = int(answer)
    return _DRAFT_A_NEW_MAP if picked == len(sites) + 1 else sites[picked - 1]


def _describe_what_gets_embedded(map_path: Path) -> None:
    """Say which COLUMNS get embedded, and which tables therefore contribute none.

    The unit is a column, not a table: `text_columns:` names the column inside a
    table that holds free text. A table can offer more than one, and a table with
    none contributes nothing searchable even though it is fully ingested. Saying
    "3 of 19 tables will be searched" mis-stated what the mapping actually decides."""
    try:
        parsed = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return
    mapping = parsed.get("chunk_metadata_columns")
    if not isinstance(mapping, dict):
        return

    embedded: list[str] = []
    structured_only: list[str] = []
    for source, fields in sorted(mapping.items()):
        columns = (fields or {}).get("text_columns") if isinstance(fields, dict) else None
        if columns:
            # Named table.column, because that is the thing being embedded — and a
            # table listing two columns contributes two, not one.
            embedded.extend(f"{source}.{column}" for column in columns)
        else:
            structured_only.append(source)

    click.echo()
    click.secho(f"  {len(embedded)} column(s) will be embedded", bold=True)
    click.secho("  Only these become searchable text; a variable's evidence is quoted "
                "from them:", dim=True)
    for entry in embedded:
        click.secho(f"    ✓ {entry}", fg="green")
    if not embedded:
        click.secho("    none — no column in this mapping is marked as text, so nothing "
                    "will be searchable", fg="red")

    if structured_only:
        click.echo()
        click.secho(f"  {len(structured_only)} table(s) have no text column", bold=True)
        click.secho("  A recipe still reads values straight out of these and cites the "
                    "row it took them\n  from — that is how date_of_birth works. What "
                    "they cannot do is be found by\n  meaning, since nothing in them was "
                    "embedded:", dim=True)
        click.secho(f"    {', '.join(structured_only)}", dim=True)

    click.echo()
    click.secho(f"  To make one searchable by meaning, add a `text_columns:` line naming "
                f"its\n  text column, under that table in {map_path.name}:", dim=True)
    click.echo("    ·  junior columns --edit    open that file here")


def _point_config_at_column_map(map_path: Path) -> None:
    """Record the chosen map in this cohort's config, so a run actually uses it.

    Telling someone to add a line to a YAML file is the step that gets skipped, and
    skipping it fails quietly: ingest falls back to generic column guesses and only
    says so as a preflight advisory."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        choice = find_config()
    except FileNotFoundError:
        choice = None

    if choice is None or choice.reason == "bundled default":
        click.echo()
        click.secho("  Add this to your project's junior.yaml:", bold=True)
        click.echo(f"    chart_columns_file: {map_path}")
        return

    # Read for the report below only. What gets WRITTEN is one spliced line — see
    # _set_one_setting — because this runs at the end of every `junior columns`, and
    # re-dumping a loaded config here is what silently ate an operator's comments and
    # froze their ${oc.env:...} interpolations to this machine.
    try:
        settings = load_config(choice.path)
    except (FileNotFoundError, ValueError) as e:
        # The map is already written; only the pointer is left. A YAML typo in the
        # settings file is the ordinary hand-edit mistake, and it reached the operator
        # as a traceback right after a tick saying the map was ready.
        raise click.ClickException(
            f"{map_path} is written, but {choice.path.name} could not be read to point "
            f"at it:\n  {e}\nFix that file, then add:  chart_columns_file: {map_path}"
        ) from e
    _set_one_setting(choice.path, "chart_columns_file", str(map_path))
    settings["chart_columns_file"] = str(map_path)
    click.echo()
    click.secho(f"✓ {choice.path.name} now uses {map_path.name}", bold=True)
    click.secho(f"  chart_columns_file: {map_path}", dim=True)
    _say_what_this_map_cannot_reach(settings)
    _describe_what_gets_embedded(map_path)
    _what_next_after_columns()


def _say_what_this_map_cannot_reach(settings: dict) -> None:
    """Warn that runs already ingested keep the mapping they were ingested with.

    Ingest resolves each chart field against a file's real header and records the answer
    in that file's sidecar; later stages read what was recorded. So pinning a map now is
    a fact about the NEXT ingest, not about the receipts already on disk — and a plain
    re-run will not revisit them, because the ingest cache is keyed on whether the source
    files changed, which they have not. Said here rather than discovered at extract,
    where the symptom is evidence quietly missing its dates."""
    output_root = settings.get("output_root")
    if not output_root:
        # Only a project that says where it writes can be asked what it has written.
        # Falling back to the ambient data root would answer about somebody else's runs.
        return
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        SENSITIVE_LABEL,
        pipeline_run_receipts_root,
    )
    from jr_pipeline.runtime_infrastructure.project_context import RUN_ID_PATTERN

    receipts = (
        Path(output_root).expanduser() / SENSITIVE_LABEL / pipeline_run_receipts_root().name
    )
    if not receipts.is_dir():
        return
    already = [
        directory for directory in receipts.iterdir()
        if directory.is_dir() and RUN_ID_PATTERN.fullmatch(directory.name)
    ]
    if not already:
        return
    click.echo()
    click.secho(
        f"  This project already has {len(already)} run(s). They keep the mapping they "
        f"were ingested with —\n"
        f"  ingest recorded it against each file, and a plain re-run sees the charts are "
        f"unchanged and keeps it.\n"
        f"  To put this map into a run:  junior ingest --force   (or start a fresh one "
        f"with --run-id)",
        fg="yellow",
    )


def ensure_layout_directories(data_root: Path) -> None:
    """Create the PHI and shareable trees for a cohort's output root."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        NON_SENSITIVE_LABEL,
        SENSITIVE_LABEL,
        pipeline_run_receipts_root,
    )

    (data_root / SENSITIVE_LABEL / pipeline_run_receipts_root().name).mkdir(
        parents=True, exist_ok=True
    )
    (data_root / NON_SENSITIVE_LABEL).mkdir(parents=True, exist_ok=True)


def _write_project_config(
    destination: Path,
    *,
    name: str,
    source_root: Path,
    output_root: Path,
    variables: list[str] | None,
) -> None:
    """Write the cohort's junior.yaml — the file every other command then discovers.

    Built from the bundled config rather than a second literal, so the encoder, chunker
    and index settings a cohort inherits stay defined in exactly one place. Only the
    three things that are this cohort's own get replaced."""
    bundled = load_config(_resolve_repo_root() / DEFAULT_CONFIG_RELATIVE_PATH)
    repo_root = _resolve_repo_root()

    settings = dict(bundled)
    settings["project"] = name
    # A run id is minted per run, not per cohort — pinning one here would make every
    # run in this cohort collide in the same directory.
    settings["run_id"] = None
    settings["source_root"] = str(source_root)
    settings["output_root"] = str(output_root)
    if variables:
        settings["recipes"] = variables
    # The bundled config names repo locations relatively; this file will be read from
    # somewhere else, so pin them now rather than rely on where it is run from.
    for key in ("recipes_root", "allowlist_path", "chart_columns_file"):
        if settings.get(key):
            settings[key] = str(_anchor_to_repo(settings[key], repo_root))
    encoder = settings.get("encoder")
    if isinstance(encoder, dict) and encoder.get("model_id"):
        encoder["model_id"] = str(_anchor_to_repo(encoder["model_id"], repo_root))

    header = (
        f"# Junior cohort: {name}\n"
        f"#\n"
        f"# Every `junior` command run from this folder (or any folder below it) uses\n"
        f"# this file. Edit it and the next run picks the change up — except mid-run:\n"
        f"# a run is locked to the settings it started with.\n"
        f"#\n"
        f"# `recipes` are the variables to extract. `junior recipes` lists every name\n"
        f"# you can put there and marks the ones this project uses. Change the list and\n"
        f"# re-run `junior extract` — the corpus built by ingest/embed/index is reused.\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        header + yaml.safe_dump(settings, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


@main.command("new-project",
              help="Start a new project: name it, choose where it lives, "
                   "point it at your charts.")
@click.argument("name", required=False)
@click.option("--input", "input_folder", default=None,
              help="Folder holding one subfolder per patient. Defaults to a folder "
                   "inside the project.")
@click.option("--into", "into", default=None,
              help="Where to create the project folder. The project gets its own "
                   "folder named after it. Defaults to here.")
@click.option("--variables", default=None,
              help="Comma-separated list of variables to extract "
                   "(default: date_of_birth). Must match recipe folder names.")
@click.option("--overwrite", is_flag=True,
              help="Replace an existing project of the same name.")
def new_project_cmd(name: str | None, input_folder: str | None, into: str | None,
                    variables: str | None, overwrite: bool) -> None:
    """Create a cohort's folder, its settings file, and its data directories."""
    # Only ask for what was not given on the command line. Asked in this order because
    # the input folder's default is derived from the name and the location, so going
    # `back` to change either refreshes the suggestion rather than leaving a path under
    # a tree you just moved away from.
    def _project_dir(answers: dict) -> Path:
        parent = Path(answers.get("into", into or ".")).expanduser()
        return (parent / answers.get("name", name or "my_cohort")).resolve()

    pending = [
        question for question, already_given in (
            (Question("name", "Name for this project",
                      lambda _answers: "my_project"), name),
            # The answer is the PARENT. Giving a project its own folder is what lets
            # several projects live side by side — each with its own junior.yaml — so
            # "/Desktop" means "/Desktop/<name>", not "put my PHI tree on the Desktop".
            (Question("into", "Where should this project's folder go?",
                      lambda _answers: "."), into),
            (Question("input_folder", "Where are the patient folders? (one folder per patient)",
                      lambda answers: str(
                          _project_dir(answers) / "CONTAINS_PHI" / "patient_data_input"
                      )), input_folder),
        ) if already_given is None or already_given == ""
    ]
    if pending:
        if len(pending) > 1 and is_interactive():
            describe_controls()
        answers = ask_sequence(pending)
        if answers is None:
            click.echo("  cancelled — nothing was written")
            return
        name = answers.get("name", name)
        into = answers.get("into", into)
        input_folder = answers.get("input_folder", input_folder)
    # Which variables to extract is deliberately NOT asked. Setting a cohort up means
    # building its corpus — ingest, embed, index — and none of those care. The choice
    # belongs to extraction, which comes later and gets re-run as recipes change; the
    # written config lists the options in a comment for when that time comes.
    vars_list = [v.strip() for v in variables.split(",") if v.strip()] if variables else None
    if vars_list:
        # Checked before the project exists rather than at the first extract, which is
        # after the whole cohort has been ingested, embedded and indexed. A name with
        # no recipe was accepted, written into the settings, and reported as created.
        bundled = load_config(_resolve_repo_root() / DEFAULT_CONFIG_RELATIVE_PATH)
        _planned_variables({
            "recipes": vars_list,
            "recipes_root": str(_resolve_recipes_root(bundled, _resolve_repo_root())),
        })

    # The place to put the project must already exist. Creating it too would happily
    # honour a typo — a stray quote once produced a folder literally named `'` with
    # the rest of the path nested inside it, reported as success. If you cannot open
    # Asked for above when it was not given, so by here there is one. Stated rather
    # than assumed: everything below joins it into a path, and a name that slipped
    # through as nothing would build one silently rather than refuse.
    if not name:
        raise click.ClickException(
            "This project needs a name — it is what its folder and its settings file "
            "are called.\nRun `junior new-project <name>`."
        )

    # the parent in Finder, Junior should not be inventing it.
    parent = Path(into or ".").expanduser()
    if not parent.is_dir():
        raise click.ClickException(
            f"{parent} does not exist, so there is nowhere to put the project.\n"
            "Give a folder that already exists — the project gets its own folder "
            "inside it."
        )
    if any(character in name for character in '\'"/\\'):
        raise click.ClickException(
            f"A project name cannot contain quotes or slashes: {name!r}\n"
            "Use --into to choose where it goes, and a plain name for what it is called."
        )

    # The project's own folder holds everything: its settings, its PHI tree, its
    # shareable tree. Two cohorts are two folders, so neither can overwrite the
    # other's config — which is what happened when every project wrote ./junior.yaml.
    project_dir = (parent / name).resolve()
    config_path = project_dir / project_config_name(name)
    if config_path.exists() and not overwrite:
        click.secho(f"  A project named {name} already lives in {project_dir}.", fg="yellow")
        if not (is_interactive() and click.confirm("  Replace its settings?", default=False)):
            click.echo("  left alone — pick another name, or --into a different folder")
            return

    input_path = (
        Path(input_folder).expanduser().resolve() if input_folder
        else project_dir / "CONTAINS_PHI" / "patient_data_input"
    )
    project_dir.mkdir(parents=True, exist_ok=True)
    _write_project_config(
        config_path, name=name, source_root=input_path,
        output_root=project_dir, variables=vars_list,
    )
    # Create the trees now rather than on first run. Someone who just chose where their
    # patient data goes should be able to open that folder and put charts in it — an
    # empty Desktop after answering three questions reads as nothing having happened.
    ensure_layout_directories(project_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    click.echo()
    click.secho(f"✓ created {project_dir}", bold=True)
    _describe_layout(project_dir, input_path)
    click.echo()
    # A cohort inherits its method from the bundled settings — the embedding model, the
    # recipe list, the allow-list, the column map. Each of those decides what comes out,
    # and answering three questions about folders and being shown only folders leaves
    # an operator with no idea that any of them was chosen on their behalf.
    _report_project_settings(load_config(config_path))
    click.echo()
    # Commands find their config by looking in the current folder and upward. At a
    # shell that means the operator has to cd; inside the prompt there is no cd, so
    # point the session at the new cohort instead of printing a shell command that
    # cannot be run where it is being read.
    import os

    from apps_and_interfaces.project_registry import remember, watch_folder

    # Written down as soon as it exists, so the next prompt opened anywhere on this
    # machine can offer it by name. A project that is not on the list still works; it
    # just has to be reached by cd or by --config, which is what every project needed
    # before the list existed.
    remember(config_path)
    # And the folder it went into is searched from now on, so this project survives
    # being renamed or moved within it — its written path would die with the move —
    # and so the next cohort put beside it turns up without being registered at all.
    watch_folder(project_dir.parent)
    if _in_interactive_session():
        os.environ[CONFIG_ENVIRONMENT_VARIABLE] = str(config_path)
        click.echo()
        click.secho(f"  Now working on {name}.", bold=True)
    click.echo()
    click.secho("  Next", bold=True)
    # Deliberately not numbered. The command listing numbers the pipeline stages, and
    # typing "1" at the prompt runs the first of those — a second numbered list here
    # would mean two different things answer to the same number.
    if not any(child.is_dir() for child in input_path.iterdir()):
        click.echo(f"    ·  put one folder per patient into {input_path}")
    if not _in_interactive_session():
        click.echo(f"    ·  cd {project_dir}")
    click.echo("    ·  junior columns     map your export's column names to Junior's")
    click.echo("    ·  junior ingest      start the run")
    click.echo("    ·  junior workbench   look at the results in a browser")


@main.command("project", help="Show which project you are on, or switch to another.")
@click.argument("which", required=False)
def project_cmd(which: str | None) -> None:
    """Show the current project and the others on this machine, or switch to one.

    Switching inside the prompt takes effect immediately, because the session holds the
    choice for every command after it. At a shell there is nothing to hold it — a
    command cannot change the environment of the shell that started it — so this prints
    the two lines that make the change stick rather than reporting a switch that ended
    when the command did.
    """
    from apps_and_interfaces.project_registry import (
        match_project,
        project_name,
        registry_path,
        remember,
        remembered_projects,
    )
    from jr_pipeline.runtime_infrastructure.project_context import (
        CONFIG_ENVIRONMENT_VARIABLE,
        find_config,
    )

    known = remembered_projects()
    if which:
        chosen = match_project(which, known)
        if chosen is None:
            listed = "".join(f"\n    {project_name(path)}" for path in known)
            raise click.ClickException(
                f"No project called {which!r}."
                + (f"\n  On this machine:{listed}" if known else "")
                + "\n  Give a name from that list, the path to a settings file or its "
                "folder, or run `junior new-project <name>`."
            )
        if _in_interactive_session():
            from apps_and_interfaces.interactive_session import work_on

            work_on(chosen)
            return
        # Naming a project is choosing it, so it joins the list here as well as at the
        # prompt. It is also the only way a cohort made before the list existed — or on
        # another machine, or restored from a backup — gets onto it without cd-ing there
        # and opening a session first.
        remember(chosen)
        click.secho(f"  {project_name(chosen)}", bold=True)
        click.secho(f"  {chosen}", dim=True)
        click.echo()
        click.echo("  To work on it, either:")
        click.echo(f"    cd {chosen.parent}")
        click.echo(f"    export {CONFIG_ENVIRONMENT_VARIABLE}={chosen}")
        return

    try:
        choice = find_config()
    except FileNotFoundError:
        choice = None
    if choice is None or choice.reason == "bundled default":
        click.secho("  No project here.", bold=True)
        click.secho(
            "  Commands run in this folder fall back to the bundled example settings.",
            dim=True,
        )
    else:
        click.secho(f"  {project_name(choice.path)}", bold=True)
        click.secho(f"  {choice.path}  ({choice.reason})", dim=True)
    click.echo()

    if known:
        click.secho("  On this machine", bold=True)
        width = max(len(project_name(path)) for path in known) + 3
        for path in known:
            marker = "·" if choice is not None and path == choice.path else " "
            click.echo(
                f"    {marker} {click.style(project_name(path).ljust(width), bold=True)}"
                f"{click.style(str(path), dim=True)}"
            )
        click.echo()
    click.secho(
        f"  Listed in {registry_path()} — a plain list of paths, edit it freely.\n"
        f"  `junior project <name>` prints how to switch to one — a command cannot "
        f"change the shell\n  that started it. Inside the `junior` prompt it switches "
        f"there and then. A project missing\n  from the list still works by cd or "
        f"--config.",
        dim=True,
    )


@main.command(help="Seal the code bundle and manifest for a run.")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--patients-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "One patient ID per line. The seal step records these in manifest."
        "target_patients so the manifest carries the real cohort. Required "
        "for SLURM-array runs; optional for single-patient local runs."
    ),
)
def seal(config_path: Path, patients_file: Path | None) -> None:
    """Seal the code bundle (recipes, src, configs, hashes) and the manifest.

    Stage commands seal for themselves, so this is only needed to seal ahead of a
    SLURM fan-out — on the login node, where a bad config fails once instead of in
    every array task — or to record the cohort roster with --patients-file."""
    cfg, _ = load_cohort_config(config_path, None)
    run_root = phi_intermediate_run_dir(cfg["run_id"])
    ensure_layout(cfg["run_id"])

    target_patients: list[str] = []
    if patients_file is not None:
        for raw_line in patients_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                target_patients.append(line)

    if (run_root / "code" / "code.lock.json").is_file():
        # A sealed bundle is never rewritten. Re-sealing deletes and recopies code/
        # and rewrites hashes.json — the provenance every existing receipt already
        # claims — and, because resume compares recorded hashes against the sealed
        # index, it silently turns finished work back into pending work. Under the
        # SAME settings a re-seal is a no-op anyway; under CHANGED settings the run
        # must refuse, which is exactly what the continuity check says. The manifest
        # and roster are still refreshed: re-running seal with --patients-file is
        # how a cluster run declares its cohort.
        code_lock_hash = _require_sealed_bundle(cfg=cfg, run_root=run_root)
        write_manifest(run_root, build_manifest(
            run_id=cfg["run_id"],
            code_lock_hash=code_lock_hash,
            entry_point_name="apps_and_interfaces.command_line_interface.seal",
            config_alias=Path(config_path).stem,
            target_patients=target_patients,
            model_sha256=model_sha256_from_cfg(cfg),
        ))
        write_roster(run_root, build_roster(
            run_id=cfg["run_id"], target_patients=target_patients,
        ))
        click.secho("  already sealed — bundle verified and kept, roster updated",
                    err=True, dim=True)
    else:
        code_lock_hash = _seal_bundle(
            cfg, run_root,
            step="seal",
            config_alias=Path(config_path).stem,
            target_patients=target_patients,
        )

    # Ship the scrubbed bundle to the shareable NO_PHI side so the flywheel has
    # the actual method (not just its fingerprint) for cross-site reproduction. The
    # bundle is PHI path/id-scrubbed by build_and_seal; the egress scan is stream-aware
    # (code stream -> path/id scrub, not the clinical-content patterns). Best-effort.
    try:
        import shutil

        from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
            sealed_code_dir,
        )

        dest = sealed_code_dir(cfg["run_id"])
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(run_root / "code", dest)
    except Exception as e:
        click.echo(f"  (note: the shareable copy of this run's saved code was not made: {e}"
                   ". The run itself is unaffected.)")

    click.echo(
        json.dumps(
            {
                "run_id": cfg["run_id"],
                "code_lock_hash": code_lock_hash,
                "n_target_patients": len(target_patients),
            },
            indent=2,
        )
    )


def _results_that_name_a_different_bundle(run_root: Path) -> dict[str, int]:
    """Results whose receipts name a code bundle other than the one in code/.

    verify_code_bundle re-hashes code/ against its own code.lock.json, so it answers
    "is this bundle internally consistent" and nothing else. It cannot see the case
    where the bundle was REPLACED after results were written: the new bundle verifies
    perfectly, and the results it did not produce sit beside it claiming a hash that no
    longer exists anywhere on disk.

    That is not hypothetical. Run 20260820_105033_b74d had nine results stamped
    sha256:dbcc2d3a… and a bundle of sha256:85ef9d23…, because a long-lived session
    re-entered the run and re-sealed it — build_and_seal rmtrees and recopies code/, so
    the bundle that produced those nine results no longer exists. `junior finish`
    reported all three checks green, because nothing compares the two.
    """
    counted: dict[str, int] = {}
    for result_file in sorted(Path(run_root).glob("patients/*/extract/*/result.json")):
        try:
            produced_by = json.loads(
                result_file.read_text(encoding="utf-8")
            ).get("produced_by") or {}
        except (OSError, json.JSONDecodeError):
            continue
        stamped = produced_by.get("code_lock_hash")
        if stamped:
            counted[str(stamped)] = counted.get(str(stamped), 0) + 1
    return counted


@main.command(help="Verify a sealed code bundle's content hash.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify(run_root: Path) -> None:
    ok, msg = verify_code_bundle(run_root)
    report: dict[str, Any] = {"ok": ok, "message": msg}

    # The bundle verifying says nothing about whether it is the bundle that produced
    # this run's values. Asked separately, and reported as its own failure, because a
    # run whose receipts name a bundle that is gone is not reproducible even though
    # every hash in it checks out.
    sealed = None
    lock = Path(run_root) / "code" / "code.lock.json"
    if lock.is_file():
        try:
            sealed = (json.loads(lock.read_text(encoding="utf-8")).get("payload")
                      or {}).get("code_lock_hash")
        except (OSError, json.JSONDecodeError):
            sealed = None
    by_hash = _results_that_name_a_different_bundle(run_root)
    strangers = {h: n for h, n in by_hash.items() if sealed and h != sealed}
    if strangers:
        ok = False
        report["ok"] = False
        report["results_from_another_bundle"] = strangers
        report["message"] = (
            f"{sum(strangers.values())} result(s) were produced by a code bundle this "
            f"run no longer holds. The bundle in code/ is {sealed}; those results name "
            + ", ".join(sorted(strangers))
            + ". Re-sealing a run replaces code/, so the code that produced them is "
            "gone and they cannot be reproduced from this run."
        )
    click.echo(json.dumps(report, indent=2))
    if not ok:
        sys.exit(1)


@main.command("validate-embed", help="Post-task validator for one patient's embed outputs.")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
@click.option("--min-chunks", type=int, default=1, help="Fail if patient has fewer chunks than this.")
@click.option("--norm-tol", type=float, default=1e-3, help="Tolerance for ||v|| ≈ 1 check.")
@click.option("--strict/--no-strict", default=True, help="Exit non-zero on first failure (default) or continue.")
def validate_embed_cmd(
    config_path: Path,
    patient_id: str,
    min_chunks: int,
    norm_tol: float,
    strict: bool,
) -> None:
    from jr_pipeline.pipeline_steps.step_2_embed_chunks.validate_embed_outputs import (
        validate_patient_output,
        write_validation_report,
    )

    cfg, _ = load_cohort_config(config_path, None)
    try:
        report = validate_patient_output(
            patient_id=patient_id,
            cfg=cfg,
            strict=strict,
            min_chunks=min_chunks,
            norm_tol=norm_tol,
        )
    except RuntimeError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    write_validation_report(report, cfg)
    click.echo(json.dumps(report.to_json(), indent=2, sort_keys=True))
    if not report.ok:
        sys.exit(1)


@main.command(help="Validate every artifact in a run against its schema.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def validate(run_root: Path) -> None:
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
        iter_validation_errors,
        known_schemas,
    )

    run_root = Path(run_root)
    schemas = set(known_schemas())
    bad: list[dict] = []
    skipped: list[dict] = []
    checked = 0
    for p in sorted(run_root.rglob("*.json")):
        try:
            env = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(env, dict) or "artifact_type" not in env:
            continue
        atype = env["artifact_type"]
        if atype not in schemas:
            skipped.append({
                "path": str(p.relative_to(run_root)),
                "artifact_type": atype,
                "reason": "no schema registered",
            })
            continue
        checked += 1
        errs = iter_validation_errors(env, atype)
        if errs:
            bad.append({
                "path": str(p.relative_to(run_root)),
                "errors": [str(e.message) for e in errs[:5]],
            })
    click.echo(json.dumps({"checked": checked, "bad": bad, "skipped": skipped}, indent=2))
    if bad:
        sys.exit(1)


@main.command(help="Run the retrieval-quality eval harness against a sealed run.")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--gold", "gold_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def eval(config_path: Path, gold_path: Path) -> None:
    from jr_pipeline.evaluating_pipeline_performance.run import run_eval

    cfg, _ = load_cohort_config(config_path, None)
    run_root = phi_intermediate_run_dir(cfg["run_id"])
    ensure_layout(cfg["run_id"])

    code_lock_hash: str | None = None
    if (run_root / "code" / "code.lock.json").is_file():
        code_lock_hash = code_bundle_for(run_root).code_lock_hash

    summary = run_eval(cfg=cfg, gold_path=gold_path, code_lock_hash=code_lock_hash)
    click.echo(
        json.dumps(
            {
                "candidate_mean_recall_at_10": summary["candidate"]["mean_recall_at_k"].get(10),
                "candidate_mean_mrr": summary["candidate"]["mean_mrr"],
                "bundle_sufficient_rate": summary["bundle"]["bundle_sufficient_rate"],
            },
            indent=2,
        )
    )


@main.command(
    "eval-values",
    help="Characterized accuracy for one or more variables vs the clinician-reviewed "
         "sample: per-field accuracy + Wilson CIs + characterized/underpowered labels, and "
         "(with --gold) retrieval-vs-extraction miss decomposition + systematic patterns.",
)
@click.option("--run-id", required=True, type=str)
@click.option("--variable", "variables", required=True, multiple=True,
              help="Variable to score; repeat for several.")
@click.option("--gold", "gold_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Retrieval-quality gold JSONL — enables retrieval-vs-extraction miss "
                   "decomposition on value mismatches.")
@click.option("--gold-run-id", "gold_run_id", default=None, type=str,
              help="Score this run against ANOTHER run's clinician review, so one "
                   "reviewed sample can characterize a new model, a retrained reranker, "
                   "or a draft recipe. Confirmations are re-scored against this run's "
                   "own answers rather than assumed correct.")
def eval_values(
    run_id: str, variables: tuple[str, ...], gold_path: Path | None, gold_run_id: str | None
) -> None:
    from jr_pipeline.evaluating_pipeline_performance.ground_truth_evaluation import (
        characterized_accuracy_report,
    )

    retrieval_gold = None
    if gold_path is not None:
        from jr_pipeline.evaluating_pipeline_performance.run import load_gold
        retrieval_gold = load_gold(gold_path)

    report = characterized_accuracy_report(
        run_id=run_id, variables=list(variables), retrieval_gold=retrieval_gold,
        gold_run_id=gold_run_id,
    )
    if gold_run_id and gold_run_id != run_id:
        click.echo(f"scoring run {run_id} against the review collected on {gold_run_id}")
    for r in report["per_variable"]:
        n = r["n_evaluated"]
        if n == 0:
            click.echo(f"{r['variable']}: no reviewed cases")
            continue
        unscorable = r.get("n_confirmations_unscorable", 0)
        unscorable_note = f", {unscorable} confirmations unscorable" if unscorable else ""
        click.echo(
            f"{r['variable']}: accuracy {r['accuracy']:.3f} "
            f"[{r['accuracy_ci_low']:.3f}, {r['accuracy_ci_high']:.3f}]  "
            f"(n={n}, correct={r['n_correct']}, confirmed={r.get('n_confirmed', 0)}"
            f"{unscorable_note})  "
            f"[{r['power']}]"
        )
    click.echo(json.dumps(report, indent=2, default=str))


@main.command(
    "collect-feedback",
    help="Save a run's clinician review — corrected values, and which passages "
         "were judged relevant — as training data.",
)
@click.option("--run-id", required=True, type=str)
@click.option("--output-dir", "output_dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
def collect_feedback_cmd(run_id: str, output_dir: Path) -> None:
    from jr_pipeline.model_training.collect_feedback_from_shiny import (
        export_chunk_relevance_labels,
        export_extraction_corrections,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        clinician_feedback_dir,
    )

    feedback_root = clinician_feedback_dir(run_id)
    n_corrections = export_extraction_corrections(
        feedback_root, output_dir / "extraction_corrections.jsonl"
    )
    n_relevance = export_chunk_relevance_labels(
        feedback_root, output_dir / "chunk_relevance.jsonl"
    )
    # A review session also mints NO_PHI relevance_label shards (one per judgment,
    # emitted by the review app alongside each PHI-side chunk_relevance entry).
    # Re-finalize so the run's parquet + manifest include them; finalize is
    # idempotent, so a session with no new shards just recompacts. Gated on the
    # NO_PHI run dir existing — a synthetic demo run id has no exhaust tree and
    # must not get one manufactured here.
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        no_phi_run_dir,
    )

    payload: dict = {
        "run_id": run_id, "extraction_corrections": n_corrections,
        "chunk_relevance": n_relevance, "output_dir": str(output_dir),
    }
    if no_phi_run_dir(run_id).is_dir():
        manifest = _finalize_exhaust_best_effort(run_id)
        if manifest is not None:
            payload["exhaust_relevance_label_rows"] = (
                (manifest["record_types"].get("relevance_label") or {}).get("n_rows", 0)
            )
    click.echo(json.dumps(payload, indent=2))


@main.command(
    "export-metadata",
    help="Package a run's shareable summary — result tables and a record of the "
         "exact code and settings that produced them, with no patient data — into "
         "one zip file. Every file is scanned first, and one hit stops the export.",
)
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Config file. Found automatically if omitted.")
@click.option("--run-id", "run_id", default=None,
              help="Run to export. The newest run in this project if omitted.")
@click.option("--output", "output_path", default=None,
              type=click.Path(dir_okay=False, path_type=Path),
              help="Where to write the zip. Defaults to beside the project's "
                   "shareable tree.")
def export_metadata_cmd(config_path: Path | None, run_id: str | None,
                        output_path: Path | None) -> None:
    from jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata import (
        export_run_metadata,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        no_phi_root,
        no_phi_run_dir,
    )

    # This is the artifact you hand to a collaborator, and it was the one advertised
    # command that never resolved a project: it read whichever data root was last
    # bound in this process, so run it in one cohort's folder after touching another
    # and it zipped up — and finalized exhaust into — the wrong cohort, reporting
    # success. Resolving first is what pins both to the project being asked about.
    cfg, config_origin = load_cohort_config(config_path, run_id)
    data_root = Path(cfg["output_root"])
    run_id = cfg["run_id"]

    shareable = no_phi_run_dir(run_id, data_root)
    if not shareable.is_dir():
        available = sorted(
            child.name for child in no_phi_root(data_root).glob("*") if child.is_dir()
        ) if no_phi_root(data_root).is_dir() else []
        listed = (
            "Runs in this project with a shareable summary:\n    " + "\n    ".join(available)
            if available
            else "No run in this project has one yet — `junior extract` writes it when "
                 "it closes a run out."
        )
        raise click.ClickException(
            f"Run '{run_id}' has no shareable summary under {data_root}.\n{listed}"
        )
    if output_path is None:
        output_path = no_phi_root(data_root) / f"{run_id}_metadata.zip"

    # Compact any un-finalized exhaust shards so the bundle carries current parquet
    # tables + manifest, not a stale inventory.
    _finalize_exhaust_best_effort(run_id, data_root)
    try:
        bundle = export_run_metadata(
            run_id=run_id, output_path=output_path, dr=data_root
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from e
    click.secho(f"  config  {config_origin}", err=True, dim=True)
    click.echo()
    click.secho(f"✓ wrote {bundle}", bold=True)
    click.secho("  Run-level metadata only — no patient data. Every file in it was "
                "scanned before it was written.", dim=True)


@main.command(help="Dump artifacts for a patient / variable in a sealed run.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
@click.option("--step", "step_name", type=str, default=None)
def inspect(run_root: Path, patient_id: str, step_name: str | None) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.run_output_reader import (
        RunLayout,
        load_json,
        source_snapshot,
    )

    layout = RunLayout(run_root)
    out: dict[str, object] = {"patient_id": patient_id}
    out["source_snapshot"] = source_snapshot(run_root, patient_id)["payload"]
    if step_name in (None, "retrieve"):
        out["retrieval_artifacts"] = [
            load_json(p)["payload"] for p in layout.retrieval_artifacts(patient_id)
        ]
    click.echo(pretty(out))


@main.command("diff", help="Diff patient-level artifacts between two runs.")
@click.option("--run-a", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--run-b", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
def diff_cmd(run_a: Path, run_b: Path, patient_id: str) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_run_outputs import compare_run_outputs

    click.echo(pretty(compare_run_outputs(run_a, run_b, patient_id)))


@main.command("compare-retrievers", help="Side-by-side retriever A/B on one query.")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
@click.option("--query", "query_text", required=True, type=str)
@click.option("--retriever-spec", "retriever_spec", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="YAML file with a top-level 'retrievers' list")
@click.option("--relevant", "relevant_ids", type=str, default=None, help="Comma-separated chunk_ids that should be retrieved (for metrics).")
def compare_retrievers_cmd(
    config_path: Path,
    patient_id: str,
    query_text: str,
    retriever_spec: Path,
    relevant_ids: str | None,
) -> None:
    import yaml

    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_retrievers import compare_retrievers

    cfg, _ = load_cohort_config(config_path, None)
    spec = yaml.safe_load(Path(retriever_spec).read_text(encoding="utf-8")) or {}
    retrievers = spec.get("retrievers") or []
    if not isinstance(retrievers, list) or not retrievers:
        raise click.ClickException("retriever-spec YAML must contain a non-empty 'retrievers' list")

    rel_ids = [s.strip() for s in (relevant_ids or "").split(",") if s.strip()] or None
    report = compare_retrievers(
        cfg=cfg,
        patient_id=patient_id,
        query_text=query_text,
        retriever_cfgs=retrievers,
        relevant_chunk_ids=rel_ids,
    )
    click.echo(pretty(report))


@main.command("compare-runs", help="Side-by-side retrieval metrics across two or more sealed runs.")
@click.option("--runs", "runs_csv", required=True, type=str, help="Comma-separated run roots (paths to out/<run_id>/).")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def compare_runs_cmd(runs_csv: str, output_format: str) -> None:
    from jr_pipeline.evaluating_pipeline_performance.compare_runs import compare_runs, format_table

    paths = [Path(p.strip()) for p in runs_csv.split(",") if p.strip()]
    for p in paths:
        if not p.is_dir():
            raise click.ClickException(f"run root is not a directory: {p}")
    report = compare_runs(paths)
    if output_format == "json":
        click.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        click.echo(format_table(report))


@main.command("summarize", help="Build out/<run_id>/summary.json from manifest + state + sidecars.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def summarize_cmd(run_root: Path) -> None:
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.run_summary import write_summary
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        pipeline_run_receipts_root,
    )

    run_root = Path(run_root)
    path = write_summary(run_root)
    # Run end is also when the NO_PHI exhaust shards get compacted into parquet + the
    # shareable manifest (the cohort runner does this at its own run end; CLI-driven
    # stage runs reach theirs here). Derive the data root from the canonical receipts
    # layout <data_root>/<phi_label>/pipeline_run_receipts/<run_id> so finalize writes
    # beside this run even when the ambient JR_DATA_ROOT points elsewhere.
    dr = (
        run_root.parents[2]
        if run_root.parent.name == pipeline_run_receipts_root().name and len(run_root.parents) >= 3
        else None
    )
    manifest = _finalize_exhaust_best_effort(run_root.name, dr)
    out: dict = {"summary_path": str(path)}
    if manifest is not None:
        out["exhaust_record_types"] = {
            rt: entry.get("n_rows") for rt, entry in manifest["record_types"].items()
        }
    click.echo(json.dumps(out, indent=2))


@main.command("recipes", help="List every recipe you can extract with, and which this project uses.")
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Config file. Found automatically if omitted.")
def recipes(config_path: Path | None) -> None:
    """Show every available variable and mark the ones this project extracts.

    Without this there is no way to answer "what can I ask for?" — the names have to
    match folders on disk, and the only way to learn them was to go and look."""
    cfg, _ = load_cohort_config(config_path, None)
    recipes_root = Path(cfg["recipes_root"]).expanduser()
    if not recipes_root.is_dir():
        raise click.ClickException(f"No variable recipes found at {recipes_root}.")

    chosen = set(cfg.get("recipes") or [])
    # A variable is a folder holding at least one version folder; the collection above
    # it (basic/, oncology/...) is how they are grouped, not part of the name.
    by_collection: dict[str, list[str]] = {}
    for version_dir in sorted(recipes_root.glob("*/**/v*/")):
        variable_dir = version_dir.parent
        if variable_dir.name.startswith("_") or variable_dir == recipes_root:
            continue
        collection = variable_dir.relative_to(recipes_root).parts[0]
        by_collection.setdefault(collection, [])
        if variable_dir.name not in by_collection[collection]:
            by_collection[collection].append(variable_dir.name)

    click.echo()
    for collection, names in sorted(by_collection.items()):
        click.secho(f"  {collection}", bold=True)
        for variable in sorted(names):
            mark = "✓" if variable in chosen else " "
            click.echo(f"    {mark} {variable}")
        click.echo()
    click.secho("  ✓ = extracted by this project.", dim=True)
    click.secho(f"  Change the list by editing `recipes:` in "
                f"{_this_projects_settings_name()}, "
                "then re-run `junior extract`.", dim=True)


# `variables` was this command's name, and the two words mean the same thing here: a
# recipe is how one variable gets extracted, and the menu lists one row per recipe. But
# everything else an operator reads says RECIPE — the folder is var_extraction_recipes/,
# the config key is `recipes:`, the run close-out asks "Editing a recipe?" — so looking
# for the list under that word found nothing, which is how somebody concludes there is
# no way to see them. The old name stays: it is what earlier notes and messages say.
main.add_command(click.Command(
    "variables",
    params=list(recipes.params),
    callback=recipes.callback,
    help="Another name for `recipes`.",
))


def _report_project_settings(cfg: dict) -> None:
    """The settings that decide what a run produces, and whether they are still there.

    A project pins the checkout's paths absolutely when it is created and nothing
    re-anchors them, so moving the checkout — or handing the cohort folder to a
    colleague — leaves all of them pointing at nothing. Without this that surfaces one
    stage at a time in the most expensive order: ingest dies on the column map, embed
    on the model, extract on the recipes, each after the whole cohort has already been
    through the stages before it."""
    encoder = cfg.get("encoder") or {}
    entries: list[tuple[str, str | None, bool]] = [
        ("model", encoder.get("model_id"), True),
        ("variables", ", ".join(cfg.get("recipes") or []) or None, False),
        ("recipes", cfg.get("recipes_root"), True),
        ("allow-list", cfg.get("allowlist_path"), True),
        ("column map", cfg.get("chart_columns_file"), True),
    ]
    click.secho("  Method", bold=True)
    click.secho("  what these decide is what comes out; edit them in this project's "
                "settings file", dim=True)
    for label, value, is_a_path in entries:
        if not value:
            click.echo(f"    {label.ljust(11)} ", nl=False)
            click.secho("not set", dim=True)
            continue
        shown = Path(value).name if is_a_path else value
        click.echo(f"    {label.ljust(11)} {shown}")
        if is_a_path and not Path(value).exists():
            click.secho(f"                ✗ {value} is not there", fg="red")


@main.command("status", help="Where things stand: which config, which run, how far along.")
@click.option("--config", "config_path", default=None,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Config file. Found automatically if omitted.")
@click.option("--run-id", "run_id", default=None, help="Run to report on.")
def status(config_path: Path | None, run_id: str | None) -> None:
    """Print the context the stage commands would use, and each stage's progress.

    Every other command picks a config and a run for you. This is where you see which
    ones it picked, before running something that writes."""
    from jr_pipeline.runtime_infrastructure.project_context import describe_run
    from jr_pipeline.runtime_infrastructure.run_progress import read_stage_progress

    cfg, config_origin = load_cohort_config(config_path, run_id)
    run = describe_run(cfg["run_id"], Path(cfg["output_root"]))
    try:
        patients = cohort_patients(cfg)
    except click.ClickException:
        patients = []

    click.secho("  config   ", nl=False, bold=True)
    click.echo(config_origin)
    click.secho("  patients ", nl=False, bold=True)
    click.echo(f"{len(patients)} in {cfg['source_root']}" if patients
               else f"none found in {cfg['source_root']}")
    click.secho("  run      ", nl=False, bold=True)
    if run["exists"]:
        click.echo(run["run_id"])
        click.secho("  outputs  ", nl=False, bold=True)
        click.echo(run["run_root"])
    elif run_id:
        # A named run that is not there is a wrong answer to a question the operator
        # actually asked. Saying "nothing run yet" discards their flag and misreports
        # a project that may well have runs.
        available = sorted(
            child.name for child in
            (Path(cfg["output_root"]) / "CONTAINS_PHI" / "pipeline_run_receipts").glob("*")
            if child.is_dir()
        ) if run["run_root"] else []
        click.secho(f"no run named '{run_id}' here", fg="yellow")
        click.secho("  outputs  ", nl=False, bold=True)
        click.echo(cfg["output_root"])
        click.echo()
        if available:
            click.secho("  Runs in this project:", bold=True)
            for existing in available:
                click.echo(f"    {existing}")
        else:
            click.secho("  This project has no runs yet.", dim=True)
        return
    else:
        # Do NOT print the id resolve_run_id just minted. Nothing has claimed it, and
        # the next command mints its own a second later — showing it would name a run
        # that never comes to exist.
        click.echo("none yet — the first stage you run starts one")
        click.secho("  outputs  ", nl=False, bold=True)
        click.echo(cfg["output_root"])
    click.echo()

    _report_project_settings(cfg)
    click.echo()

    if not run["exists"]:
        click.secho("  Nothing run yet. Start with:  junior ingest", dim=True)
        return

    run_root = Path(run["run_root"])
    total = len(patients) or None
    # cohort_patients counts every subfolder of source_root, but ingest screens the
    # tree first: a directory of directories (an export bundle beside the patients)
    # never enters the run. Counting it in the denominator forever left a finished
    # cohort reading "1 of 2 · started". Once patients exist in the run, the run is
    # the denominator — for the later stages it is exactly who they can run over —
    # and the ingest row says how many folders never entered.
    patients_dir = run_root / "patients"
    folders_that_never_entered = 0
    if patients_dir.is_dir():
        in_the_run = sum(1 for child in patients_dir.iterdir() if child.is_dir())
        if in_the_run and total and in_the_run < total:
            folders_that_never_entered = total - in_the_run
            total = in_the_run
    # A run started by `extract --new-run` did not RUN ingest, embed or index — it
    # inherited their output. Progress is counted from state transitions, and an
    # inherited stage records none, so all three read "not started yet" while their
    # artifacts sit in the folder. Someone acting on that would rebuild a corpus they
    # already have.
    inherited_from = None
    inheritance_record = run_root / "corpus_inheritance.json"
    if inheritance_record.is_file():
        try:
            inherited_from = json.loads(
                inheritance_record.read_text(encoding="utf-8")
            ).get("corpus_source_run_id")
        except (OSError, json.JSONDecodeError):
            inherited_from = None
    corpus_stages = {"ingest", "embed", "index"}

    click.secho("  ✓ finished   · started   (blank) not started yet", dim=True)
    for position, stage in enumerate(PIPELINE_STAGE_DESCRIPTIONS, 1):
        progress = read_stage_progress(run_root, stage, total=total)
        done = progress.completed
        if inherited_from and stage in corpus_stages and not done:
            click.echo(f"  ✓ step {position}  {stage.ljust(9)} reused from {inherited_from}")
            continue
        if total:
            mark = "✓" if done >= total else ("·" if done else " ")
            # "1. ingest 1/2" read as a single fraction. Spell out both numbers so the
            # step's position and the patient count cannot be mistaken for each other.
            detail = f"{done} of {total} patients" if done else "not started yet"
        else:
            mark = "✓" if done else " "
            detail = f"{done} patients" if done else "not started yet"
        if stage == "ingest" and done and folders_that_never_entered:
            detail += (
                f" ({folders_that_never_entered} folder(s) screened out — "
                "not patient folders)"
            )
        click.echo(f"  {mark} step {position}  {stage.ljust(9)} {detail}")


def _where_this_variables_recipe_lives(recipes_root: Path, variable: str) -> Path | None:
    """The folder holding a variable's recipe, so a person can be sent straight to it.

    Recipes nest by collection (basic/, oncology/breast_oncology/, ...), which a reader
    has no reason to know. Nobody should have to go and find their own recipe."""
    matches = sorted(recipes_root.glob(f"**/{variable}/v*"))
    if not matches:
        return None
    found = matches[-1]
    try:
        # Relative to the checkout: shorter to read, and it is the same path for whoever
        # this gets pasted to, which an absolute one under a home directory is not.
        return found.relative_to(_resolve_repo_root())
    except ValueError:
        return found


def _what_to_do_next(run_root: Path) -> None:
    """Say what to do now the run is closed, given what this run actually produced.

    Junior exists to be iterated on: change a recipe, run it again, see whether it got
    better. That loop was undiscoverable. A run ended on a tick and a prompt, and
    everything after it -- which variable to look at, where its recipe lives, that a
    change needs a fresh run and why -- had to be known already or found by hitting an
    error. A tool whose whole design is iteration cannot go quiet at the moment someone
    is deciding what to change."""
    import yaml

    failed_by_variable: Counter = Counter()
    tried_by_variable: Counter = Counter()
    for result_file in sorted(run_root.glob("patients/*/extract/*/result.json")):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8")).get("payload") or {}
        except (OSError, json.JSONDecodeError):
            continue
        variable = result_file.parent.name
        tried_by_variable[variable] += 1
        if not payload.get("ok"):
            failed_by_variable[variable] += 1
    if not tried_by_variable:
        return  # nothing was extracted, so there is nothing to iterate on yet

    # WHERE THE ANSWERS ARE, before anything about what to change. A run ends on three
    # ticks and then explains how to iterate, which is the second question; the first is
    # "where is what I just waited for". Nothing printed it, and nothing else in either
    # interface names the folder — an operator had to know the layout, and the layout
    # moved. Asked of the same helpers that write them, so it cannot drift.
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        RUN_METADATA_TABLE_NAME,
        recipe_tables_in,
        run_output_dir,
    )

    tables = recipe_tables_in(run_root)
    if tables:
        click.echo()
        click.secho("✓ your answers", bold=True)
        click.secho(f"  {run_output_dir(run_root)}", fg="green")
        click.secho(f"  {len(tables)} recipe table(s), one row per patient, "
                    f"plus {RUN_METADATA_TABLE_NAME}", dim=True)

    recipes_root = None
    sealed_config = run_root / "code" / "config_resolved.yaml"
    if sealed_config.is_file():
        try:
            # The config that RAN, not whatever the working tree says now. The point of
            # naming a recipe here is that it is the one which produced these values.
            declared = yaml.safe_load(sealed_config.read_text(encoding="utf-8")) or {}
            if declared.get("recipes_root"):
                recipes_root = Path(str(declared["recipes_root"]))
        except (OSError, yaml.YAMLError):
            recipes_root = None

    # A pointer, not a second warning. The stage has already said what failed and how
    # much of it; repeating that here with a different denominator -- values there,
    # patients here -- reads like two problems.
    for variable, _failed in failed_by_variable.most_common(3):
        folder = _where_this_variables_recipe_lives(recipes_root, variable) if recipes_root else None
        if folder:
            click.secho(f"\n  {variable} recipe: {folder}", dim=True)
    # Three questions, because they are the three things somebody does after reading
    # the line above. The split between them is the real invalidation boundary: a
    # recipe change cannot alter passages or vectors, so it does not rebuild them.
    click.echo()
    click.secho("! Editing a recipe?", bold=True)
    click.echo("  Edit it, then re-extract in a fresh run. The existing embeddings will be re-used.")
    click.secho("      junior extract --new-run", dim=True)

    click.echo()
    click.secho("! Adding or removing a recipe/variable?", bold=True)
    click.echo("  Edit `recipes:` in your project settings.")
    click.secho("      added     junior extract --new-run", dim=True)
    click.secho("      removed   keep this run", dim=True)

    click.echo()
    click.secho("! Changing cohort or corpus settings, or rerunning everything from scratch?",
                bold=True)
    click.echo("  Rebuild from the charts, a stage at a time. The first one starts the run;")
    click.echo("  the rest follow it.")
    click.secho("      junior ingest --new-run\n"
                "      junior embed\n"
                "      junior index\n"
                "      junior extract", dim=True)

    click.secho("\n  junior run                 retry or continue this run\n"
                "  junior extract             retry or continue extraction only\n"
                "  junior extract --new-run   fresh extraction; reuse the embeddings\n"
                "  junior ingest --new-run    fresh run; rebuild from the charts", dim=True)


@main.command("finish", help="Close a run out: summarize it, then check it end to end.")
@click.option("--run-root", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.pass_context
def finish(ctx: click.Context, run_root: Path) -> None:
    """Summarize the run, validate its artifacts, and verify the code bundle.

    Run this once every patient is done. It is three things because they answer three
    different questions — is the run finished, is what it wrote well-formed, is it
    still the code that ran — but you want all three at the same moment, so they are
    one command. Each step delegates to the standalone command, which stays the single
    implementation.

    Summarizing happens even if the checks then fail: a run that finished badly still
    needs its status flipped off "running", or it looks like it is still going."""
    # The three commands underneath print their own JSON. That is the right output
    # for something being piped into a script, and the wrong output at the end of a
    # run someone just watched, so capture it and report the verdict instead. The
    # detail is on disk either way; -v on the individual commands still shows it.
    from contextlib import redirect_stdout
    from io import StringIO

    steps = (
        ("wrote the run summary", summarize_cmd),
        ("checked every file it wrote", validate),
        ("checked the code still matches what ran", verify),
    )
    failures: list[str] = []
    for description, command in steps:
        captured = StringIO()
        try:
            with redirect_stdout(captured):
                ctx.invoke(command, run_root=run_root)
        except SystemExit as e:
            # The standalone commands exit non-zero on failure. Keep going so one
            # report covers everything, then fail once at the end.
            if e.code:
                failures.append(description)
                click.secho(f"  ✗ {description}", fg="red")
                click.echo(captured.getvalue().rstrip())
                continue
        click.secho(f"  ✓ {description}", dim=True)

    if failures:
        raise click.ClickException(
            "The run finished, but these checks did not pass:\n  "
            + "\n  ".join(failures)
            + f"\nThe detail is above. Re-check with:  junior finish --run-root {run_root}"
        )
    click.secho("\n✓ run closed out", bold=True)
    _what_to_do_next(run_root)


@main.command("health", help="Print a run-level operational rollup from state.jsonl.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def health(run_root: Path) -> None:
    from collections import Counter

    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.run_output_reader import (
        iter_state,
        list_patients,
        manifest,
    )

    m = manifest(run_root)["payload"]
    transitions = list(iter_state(run_root))
    stage_states: dict[str, Counter] = {}
    for env in transitions:
        p = env["payload"]
        ent = p["entity"]
        if ent["kind"] != "step":
            continue
        bucket = stage_states.setdefault(ent["step"], Counter())
        bucket[p["to_state"]] += 1

    summary = {
        "run_id": m["run_id"],
        "status": m.get("status"),
        "patients": list_patients(run_root),
        "stage_states_terminal": {k: dict(v) for k, v in stage_states.items()},
        "n_state_events": len(transitions),
    }
    # Skipped stages (no torch / no embeddings / no allowlist) never record a state
    # transition, so the rollup above can't see them. Surface the runner's durable
    # cohort_result.json so a half-built cohort doesn't look clean here.
    cr_path = Path(run_root) / "cohort_result.json"
    if cr_path.is_file():
        try:
            cr = json.loads(cr_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cr = None
        if cr is not None:
            stage_status = {
                k: cr.get(k) or {} for k in ("embed_status", "index_status", "extract_status")
            }
            n_skipped = sum(
                1 for d in stage_status.values() for v in d.values() if str(v).startswith("skipped:")
            )
            n_failed = sum(
                1 for d in stage_status.values() for v in d.values() if str(v).startswith("failed:")
            )
            summary["runner_stage_status"] = stage_status
            summary["n_stage_skipped"] = n_skipped
            summary["n_stage_failed"] = n_failed
    click.echo(pretty(summary))


@main.group()
def code() -> None:
    """Operate on the code provenance stream (no PHI)."""


@code.command("inspect", help="Print code-bundle contents + scoped hashes.")
@click.option("--run-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def code_inspect(run_root: Path) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.run_output_reader import hashes

    code_dir = Path(run_root) / "code"
    click.echo(pretty({
        "code_lock_hash": hashes(run_root)["payload"]["code_lock_hash"],
        "files": sorted(p.relative_to(code_dir).as_posix() for p in code_dir.rglob("*") if p.is_file()),
        "hashes": hashes(run_root)["payload"],
    }))


@code.command("diff", help="Diff code/hashes.json between two runs.")
@click.option("--run-a", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--run-b", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def code_diff_cmd(run_a: Path, run_b: Path) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_run_outputs import code_diff

    click.echo(pretty(code_diff(run_a, run_b)))


@code.command("scoped-diff", help="Diff code hashes limited to one scope.")
@click.option("--run-a", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--run-b", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--scope", required=True, type=click.Choice(["recipe", "prompt", "schema", "python_helpers", "provider", "retrieval", "config", "env"]))
def code_scoped_diff(run_a: Path, run_b: Path, scope: str) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_run_outputs import code_diff

    click.echo(pretty(code_diff(run_a, run_b, scope=scope)))


@main.command("compare-models", help="Run a variable through two allowlist endpoints; compare outputs.")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
@click.option("--variable", required=True, type=str)
@click.option("--models", required=True, type=str, help="Comma-separated endpoint NAMES from the allowlist.")
def compare_models_cmd(config_path: Path, patient_id: str, variable: str, models: str) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_llm_models import compare_models

    cfg, _ = load_cohort_config(config_path, None)
    names = [s.strip() for s in models.split(",") if s.strip()]
    click.echo(pretty(compare_models(cfg=cfg, patient_id=patient_id, variable=variable, model_names=names)))


@main.command(
    "diff-with-code-context",
    help="Diff a patient/variable between two runs AND show the scoped code-hash differences that explain it.",
)
@click.option("--run-a", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--run-b", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--patient", "patient_id", required=True, type=str)
@click.option("--variable", required=True, type=str)
def diff_with_code_context_cmd(run_a: Path, run_b: Path, patient_id: str, variable: str) -> None:
    from jr_pipeline.evaluating_pipeline_performance.analysis_report import pretty
    from jr_pipeline.evaluating_pipeline_performance.compare_run_outputs import (
        compare_outputs_with_code_changes,
    )

    click.echo(pretty(compare_outputs_with_code_changes(run_a, run_b, patient_id, variable)))


# Commands that answer a question about a finished run rather than advance one.
# The app is the place for that work — it shows the evidence next to the value, which
# a terminal cannot. These stay registered and callable (the deployment docs and the
# workbench notes reference them, and a cluster shell may be all you have), but they
# are kept out of `junior --help` so the listed surface is only what a run needs.
# Move a name out of this set to advertise it again.
APP_SIDE_COMMANDS = frozenset({
    "run",  # whole-cohort execution; the app's Start tab is the equivalent
    # Sealing happens on the first stage command now. The standalone command stays
    # for a cluster login node, where sealing before `sbatch` fails a bad config
    # once instead of in every one of 200 array tasks.
    "seal",
    # Retrieval runs inside extract; this command probes that subsystem in isolation.
    "retrieve",
    # An engineer's post-task check ("tolerance for ||v|| ≈ 1"), not a step in a run.
    "validate-embed",
    # Extraction closes the run out itself, so none of these is something to choose.
    # They stay callable to re-check a run, or for a cluster that wants one alone.
    "finish", "summarize", "validate", "verify",
    "health", "inspect", "diff", "diff-with-code-context", "code",
    "eval", "eval-values", "compare-runs", "compare-models", "compare-retrievers",
    "collect-feedback",
})

for _name in APP_SIDE_COMMANDS:
    main.commands[_name].hidden = True


if __name__ == "__main__":
    main()
