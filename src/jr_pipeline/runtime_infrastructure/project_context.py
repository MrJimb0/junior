"""Which cohort the CLI is working on, when you did not say.

Typing ``--config`` on every command is the kind of repetition a CLI should absorb.
This module answers "which config" and "which run" from the surroundings, in a fixed
order of preference, so ``junior ingest`` on its own does the obvious thing while an
explicit flag still wins for a cluster job that must be unambiguous.

Config, most trusted first:

  1. ``--config`` on the command line
  2. ``$JUNIOR_CONFIG`` — set by the prompt when you pick a project, so a session works
     on the cohort you chose rather than on whichever folder the terminal opened in
  3. ``junior.yaml`` in the current directory, or any directory above it
  4. the bundled ``deployment/local/laptop.yaml``

Walking upward means a config at the top of a project applies to every directory
inside it, the way a git repo does.

Run id, most trusted first:

  1. ``--run-id`` on the command line
  2. ``$JR_RUN_ID`` — set by the SLURM scripts, so a cluster fan-out keeps every array
     task in one run
  3. ``run_id`` in the config file
  4. the newest run already under the data root, so a second command continues the run
     the first one started instead of silently beginning a new one
  5. a fresh timestamped id
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# A project's settings file is named after the project — `junior_breast2026.yaml` —
# so it is identifiable in an editor tab, a search result, or a folder listing, where
# a dozen files all called `junior.yaml` are indistinguishable. Discovery matches the
# pattern rather than one name, so the plain `junior.yaml` written by earlier versions
# keeps working.
PROJECT_CONFIG_GLOB = "junior*.yaml"
PROJECT_CONFIG_NAME = "junior.yaml"
CONFIG_ENVIRONMENT_VARIABLE = "JUNIOR_CONFIG"


def project_config_name(project: str) -> str:
    """The settings-file name for a project: `junior_<project>.yaml`."""
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in project.strip()
    ).strip("_")
    return f"junior_{safe}.yaml" if safe else PROJECT_CONFIG_NAME
BUNDLED_CONFIG_RELATIVE_PATH = Path("deployment") / "local" / "laptop.yaml"

# The shape _fresh_run_id produces. Runs that do not match are left alone when picking
# the newest run, so a scratch directory never gets mistaken for one.
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}_[0-9a-f]+")


@dataclass(frozen=True)
class ConfigChoice:
    """A config path and a short phrase saying why it was chosen, for `junior status`."""
    path: Path
    reason: str


def find_config(explicit: Path | None = None, *, start: Path | None = None) -> ConfigChoice:
    """Locate the config to use. Raises FileNotFoundError only if even the bundled one
    is missing, which means the checkout is broken rather than the caller wrong."""
    if explicit is not None:
        return ConfigChoice(Path(explicit).expanduser().resolve(), "given with --config")

    from_environment = os.environ.get(CONFIG_ENVIRONMENT_VARIABLE)
    if from_environment:
        return ConfigChoice(
            Path(from_environment).expanduser().resolve(), f"${CONFIG_ENVIRONMENT_VARIABLE}"
        )

    here = (start or Path.cwd()).resolve()
    for directory in [here, *here.parents]:
        found = sorted(path for path in directory.glob(PROJECT_CONFIG_GLOB) if path.is_file())
        if not found:
            continue
        if len(found) > 1:
            # A project owns its folder, so this means two projects were put in one.
            # Guessing between them would run the wrong cohort silently.
            listed = "\n  ".join(path.name for path in found)
            raise FileNotFoundError(
                f"More than one project settings file in {directory}:\n  {listed}\n"
                "Each project needs its own folder. Move them apart, or say which to "
                "use with --config."
            )
        where = "here" if directory == here else f"found in {directory}"
        return ConfigChoice(found[0], f"{found[0].name} {where}")

    bundled = _repo_root() / BUNDLED_CONFIG_RELATIVE_PATH
    if not bundled.is_file():
        raise FileNotFoundError(
            f"No project settings file ({PROJECT_CONFIG_GLOB}) here or above, and the "
            f"bundled config is missing from {bundled}. Pass --config, or run "
            f"`junior new-project <name>`."
        )
    return ConfigChoice(bundled, "bundled default")


def resolve_run_id(
    cfg: dict,
    explicit: str | None = None,
    *,
    data_root: Path | None = None,
    continue_newest: bool = True,
    start_fresh: bool = False,
) -> str:
    """Decide which run this command belongs to.

    ``continue_newest=False`` skips the "carry on with the newest run" step entirely.
    Passing ``data_root=None`` is NOT a way to do that — the lookup would fall back to
    the ambient data root, which is exactly the directory a caller outside a project
    must not append to.

    ``start_fresh`` is "start over", and it outranks every other source including a
    pinned ``run_id:`` in the config and ``JR_RUN_ID`` in the environment. Those exist
    to say WHICH run; this says a new one, and a request to start over that silently
    landed back in a pinned run would be the worst of both."""
    if start_fresh:
        return _fresh_run_id()
    if explicit:
        return explicit
    from_environment = os.environ.get("JR_RUN_ID")
    if from_environment:
        return from_environment
    from_config = cfg.get("run_id")
    if from_config:
        return str(from_config)
    if continue_newest and (latest := newest_run_id(data_root)) is not None:
        return latest
    return _fresh_run_id()


def newest_run_id(data_root: Path | None = None) -> str | None:
    """The most recently started run under the data root, if any.

    Continuing the newest run is what makes `junior ingest` then `junior embed` land in
    the same place without the caller naming it. Only canonically-named runs count, so
    a hand-made directory cannot capture the next command."""
    receipts = _receipts_root(data_root)
    if receipts is None or not receipts.is_dir():
        return None
    runs = [
        directory
        for directory in receipts.iterdir()
        if directory.is_dir() and RUN_ID_PATTERN.fullmatch(directory.name)
    ]
    if not runs:
        return None
    return max(runs, key=lambda directory: directory.name).name


def describe_run(run_id: str, data_root: Path | None = None) -> dict:
    """Enough about a run to print a status line without loading the pipeline."""
    receipts = _receipts_root(data_root)
    run_root = (receipts / run_id) if receipts else None
    patients_dir = (run_root / "patients") if run_root else None
    return {
        "run_id": run_id,
        "run_root": str(run_root) if run_root else None,
        "exists": bool(run_root and run_root.is_dir()),
        "sealed": bool(run_root and (run_root / "code" / "code.lock.json").is_file()),
        "patients": (
            sorted(child.name for child in patients_dir.iterdir() if child.is_dir())
            if patients_dir and patients_dir.is_dir()
            else []
        ),
    }


def _receipts_root(data_root: Path | None) -> Path | None:
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    # Ask the layout module for a known run's directory and step back up, so this stays
    # correct if the layout changes rather than re-spelling the path here.
    try:
        return phi_intermediate_run_dir("__probe__", data_root).parent
    except Exception:
        return None


def _fresh_run_id() -> str:
    import secrets

    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def _repo_root() -> Path:
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (  # noqa: E501
        resolve_repo_root,
    )

    return resolve_repo_root()
