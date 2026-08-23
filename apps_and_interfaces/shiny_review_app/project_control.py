"""Which project the app is on — shown, switchable, and creatable, the CLI's way.

The app opens on the project `junior workbench` resolved and pinned. This module is
how it stops being stuck there: the same registry the prompt's project menu reads,
a switch that re-pins the same environment the workbench command pins, and a New
project door that IS `junior new-project` — spawned as a subprocess with every
question answered by the form, so the app cannot grow a project-creation dialect
of its own.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class ProjectInfo:
    config_path: Path
    name: str
    source_root: str
    output_root: Path


def _load_settings(config_path: Path) -> dict:
    from jr_pipeline.runtime_infrastructure.config_loading import load_config

    return load_config(config_path)


def _resolved_output_root(config_path: Path, settings: dict) -> Path:
    """The project's data root, anchored the way the CLI anchors it: an absolute
    path as written, else beside the settings file that names it."""
    raw = str(settings.get("output_root") or "data")
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def project_info(config_path: Path) -> ProjectInfo:
    config_path = Path(config_path).expanduser().resolve()
    settings = _load_settings(config_path)
    return ProjectInfo(
        config_path=config_path,
        name=str(settings.get("project") or config_path.parent.name),
        source_root=str(settings.get("source_root") or ""),
        output_root=_resolved_output_root(config_path, settings),
    )


def current_project() -> ProjectInfo | None:
    """The project the app is pinned to, resolved exactly as every command resolves
    it — the pinned JUNIOR_CONFIG first, else discovery. None when nothing resolves
    (a bare checkout with no project anywhere)."""
    from jr_pipeline.runtime_infrastructure.project_context import find_config

    try:
        return project_info(find_config().path)
    except Exception:
        return None


def known_projects() -> list[Path]:
    """The registry's projects, most recently used first — the same list the
    prompt's project menu shows."""
    from apps_and_interfaces.project_registry import remembered_projects

    return remembered_projects()


def switch_to(config_path: Path) -> ProjectInfo:
    """Pin the app (and every child it spawns) to another project.

    The same two variables `junior workbench` pins at launch: the config, so the
    Run button's `junior run` child resolves this project; and the data root, so
    everything the app reads and writes moves with it. The registry remembers the
    switch, which is what keeps the project menu's most-recent-first order true."""
    from apps_and_interfaces.project_registry import remember

    info = project_info(config_path)
    from jr_pipeline.runtime_infrastructure.project_context import (
        CONFIG_ENVIRONMENT_VARIABLE,
    )

    os.environ[CONFIG_ENVIRONMENT_VARIABLE] = str(info.config_path)
    os.environ["JR_DATA_ROOT"] = str(info.output_root)
    remember(info.config_path)
    return info


def command_for_new_project(name: str, charts_folder: str, into_folder: str) -> list[str]:
    """The exact `junior new-project` invocation the Create button performs.

    Every question the command would ask is answered by a flag, so the subprocess
    never prompts — and the app adds nothing but a way to click it."""
    return [
        sys.executable, "-m", "apps_and_interfaces.command_line_interface",
        "new-project", name,
        "--input", str(charts_folder),
        "--into", str(into_folder),
    ]


def create_project(name: str, charts_folder: str, into_folder: str) -> tuple[str, Path | None]:
    """Run `junior new-project` and return (its output, the new config path or None).

    The current project's pin is withheld from the child: creating a project must
    not read as happening inside whichever project the app was open on."""
    from jr_pipeline.runtime_infrastructure.project_context import (
        CONFIG_ENVIRONMENT_VARIABLE,
    )

    # The CLI treats an existing project as a polite no-op ("left alone", exit 0),
    # which is right at a terminal and wrong behind a Create button: finding the
    # OLD config and switching to it would report a creation that never happened.
    # Refused here, before anything is spawned.
    project_dir = (Path(into_folder).expanduser() / name).resolve()
    if project_dir.exists():
        return (
            f"A project named {name} already lives in {project_dir.parent} — "
            "pick another name, or switch to the one that is there.",
            None,
        )

    environment = dict(os.environ)
    environment.pop(CONFIG_ENVIRONMENT_VARIABLE, None)
    finished = subprocess.run(
        command_for_new_project(name, charts_folder, into_folder),
        cwd=str(REPO), env=environment, capture_output=True, text=True, timeout=120,
    )
    output = (finished.stdout + finished.stderr).strip()
    if finished.returncode != 0:
        return output, None
    # The settings filename is the command's decision (junior*.yaml; the exact
    # spelling has changed across versions), so find what it wrote rather than
    # guessing its name.
    project_dir = (Path(into_folder).expanduser() / name).resolve()
    written = sorted(project_dir.glob("junior*.yaml")) if project_dir.is_dir() else []
    return output, written[0] if written else None
