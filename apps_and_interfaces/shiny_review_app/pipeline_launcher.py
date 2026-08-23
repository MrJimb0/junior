"""Drive a full pipeline run (ingest -> embed -> ... -> extract) from the Start tab.

The Run button spawns ``junior run`` itself — the same command, the same config
resolution, the same engine path an operator gets at a shell. The app adds nothing
but a way to click it, which is what keeps the two interfaces from growing
dialects: anything `junior run` learns, the button learns for free.

It runs as a SUBPROCESS, not in the app process: it loads the encoder and the
local LLM, takes minutes, and must not freeze the Shiny event loop or leave a
half-loaded model behind when the operator navigates away. A reader thread tails
the child's stdout into a bounded buffer the UI polls.

The child's stdout carries stage progress and per-variable ok/failed marks only —
extracted values are redacted there by the pipeline itself — so the log panel is
safe to show on screen.

Also answers the two questions the Start tab asks before offering the button:
which variables have recipes on disk (``available_variables``) and whether this
checkout can actually run a full spine (``readiness_problems``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPES_ROOT = REPO / "var_extraction_recipes"
MAX_LOG_LINES = 400

# The four stages, in the order the CLI runs them. Each is offered as its own button
# so a stage can be re-run alone — an embed that died on one patient does not mean
# ingesting the cohort again — and each button spawns the very command a shell would.
STAGES = ("ingest", "embed", "index", "extract")


# What the public release ships, in the order junior_v1_public_release_plan.md lists
# it: dates, staging, receptor status, clinical, genetics. The recipe tree still holds
# the thirteen that plan drops, and the Start tab must not offer them — a tick there
# RUNS the recipe, and a run that included `treatment_lines` produced a table the
# release is not going to publish.
RELEASE_VARIABLES = (
    "date_of_birth",
    "date_of_death",
    "date_of_diagnosis",
    "stage",
    "breast_receptors",
    "second_opinion_or_not",
    "genetics_germline",
    "genetics_somatics",
)


def recipes_on_disk() -> list[str]:
    """Every variable name with a recipe under the recipe tree, at any depth.

    A variable folder is one holding version subdirs (``v1``, ``v2``, ...) — the same
    shape the extract step resolves by name, so anything listed here is runnable.
    """
    if not RECIPES_ROOT.is_dir():
        return []
    names = set()
    for version_dir in RECIPES_ROOT.glob("**/v[0-9]*"):
        if version_dir.is_dir() and any(version_dir.glob("*_recipe.yaml")):
            names.add(version_dir.parent.name)
    return sorted(names)


def available_variables() -> list[str]:
    """The variables the Start tab offers: the release set, minus any whose recipe is
    not on disk. Intersected rather than asserted, so this keeps working while the cut
    is being made and a dropped folder simply stops appearing."""
    present = set(recipes_on_disk())
    return [name for name in RELEASE_VARIABLES if name in present]


def readiness_problems() -> list[str]:
    """Everything missing that would make a full run fail, in plain English. Empty
    list means the spine can run here."""
    problems = []
    try:
        import jr_pipeline  # noqa: F401
    except Exception as e:
        problems.append(f"jr_pipeline is not importable ({type(e).__name__}) — install the repo with pip install -e .")
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        problems.append('torch/transformers missing — embed and extract need them: pip install -e ".[torch]"')
    if not (REPO / "models" / "embedding").is_dir():
        problems.append("no encoder weights under models/embedding — embed cannot run")
    if not (REPO / "models" / "extraction").is_dir():
        problems.append("no LLM weights under models/extraction — extract cannot run")
    if not available_variables():
        problems.append(f"no recipes found under {RECIPES_ROOT}")
    return problems


@dataclass
class PipelineRun:
    """One in-flight (or finished) full-spine run and its captured output."""

    patients: list[str]
    variables: list[str]
    process: subprocess.Popen
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    lock: threading.Lock = field(default_factory=threading.Lock)
    run_id: str = ""

    def _tail(self) -> None:
        for raw in self.process.stdout:  # type: ignore[union-attr]
            line = raw.rstrip("\n")
            # Weight-loading progress bars redraw with \r, which universal newlines
            # turns into one "line" per repaint — hundreds of them would push the real
            # stage output out of the buffer.
            if "%|" in line:
                continue
            # The cohort runner announces its run id near the top ("Run ID:  <id>");
            # capture it so the app can jump straight to the new run.
            run_id = line.split(":", 1)[1].strip() if line.startswith("Run ID:") else ""
            with self.lock:
                self.lines.append(line)
                if run_id:
                    self.run_id = run_id
        self.process.wait()

    def log(self) -> list[str]:
        with self.lock:
            return list(self.lines)

    @property
    def is_running(self) -> bool:
        return self.process.poll() is None

    @property
    def succeeded(self) -> bool:
        return self.process.poll() == 0

    def status_line(self) -> str:
        who = f"{len(self.patients)} patient(s) × {len(self.variables)} variable(s)"
        if self.is_running:
            return f"Running — {who}. Ingest → embed → index → retrieve → rerank → extract."
        if self.succeeded:
            return f"Finished — {who}. Run {self.run_id or '(id not captured)'} is ready to review."
        return f"Stopped with exit code {self.process.poll()} — see the log below."

    @property
    def produces_values(self) -> bool:
        """A full spine ends in extract, so the review picker can jump to it."""
        return True

    def stop(self) -> None:
        if self.is_running:
            self.process.terminate()


@dataclass
class StageRun:
    """One CLI stage, run over the ticked patients, and its captured output.

    The same shape as PipelineRun on purpose — ``log()``, ``is_running``,
    ``succeeded``, ``status_line()``, ``stop()`` — so the Start tab's existing status
    line, log panel and Stop button drive a stage run with no second panel to show it.

    One subprocess per invocation, walked in order, because ``junior <stage>`` takes a
    single ``--patient``. Ticking every patient omits it instead, which is the same
    whole-cohort command rather than N of them."""

    stage: str
    patients: list[str]
    commands: list[list[str]]
    environment: dict[str, str]
    lines: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    lock: threading.Lock = field(default_factory=threading.Lock)
    run_id: str = ""
    process: subprocess.Popen | None = None
    failed_code: int | None = None
    finished: bool = False
    stopping: bool = False

    def _walk(self) -> None:
        for command in self.commands:
            if self.stopping:
                break
            with self.lock:
                self.lines.append(f"$ {' '.join(command)}")
            process = subprocess.Popen(
                command, cwd=str(REPO), env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self.process = process
            for raw in process.stdout:  # type: ignore[union-attr]
                line = raw.rstrip("\n")
                if "%|" in line:
                    continue
                # Every stage announces its run as "  run     <id>"; capture it so a
                # finished extract can hand the review picker somewhere to go.
                fields = line.split()
                with self.lock:
                    self.lines.append(line)
                    if len(fields) == 2 and fields[0] == "run":
                        self.run_id = fields[1]
            process.wait()
            if process.returncode != 0:
                self.failed_code = process.returncode
                break
        self.finished = True

    def log(self) -> list[str]:
        with self.lock:
            return list(self.lines)

    @property
    def is_running(self) -> bool:
        return not self.finished

    @property
    def succeeded(self) -> bool:
        return self.finished and self.failed_code is None and not self.stopping

    @property
    def produces_values(self) -> bool:
        """Only extract writes values. Adopting an ingest into the review picker
        would send the reviewer to a run with nothing in it yet."""
        return self.stage == "extract"

    def status_line(self) -> str:
        who = f"{len(self.patients)} patient(s)"
        if self.is_running:
            return f"Running {self.stage} — {who}, as `junior {self.stage}` does at a shell."
        if self.stopping:
            return f"{self.stage} stopped on request — see the log below."
        if self.failed_code is not None:
            return f"{self.stage} stopped with exit code {self.failed_code} — see the log below."
        return f"{self.stage} finished — {who}. Run {self.run_id or '(id not captured)'}."

    def stop(self) -> None:
        self.stopping = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def stage_command_for(stage: str, patient: str | None = None,
                      run_id: str = "", variables: list[str] | None = None,
                      new_run: bool = False) -> list[str]:
    """The exact `junior <stage>` invocation a step button performs.

    A stage command takes its cohort from the project the app is pinned to — it has no
    folder argument and no --output to be given one. That is not the button being
    lossy: it is the same command with the same reach a shell has, which is the whole
    point of the button being a button and not a second implementation.

    ``run_id`` is why the buttons chain. On the bundled example settings — no project —
    every stage command deliberately starts a FRESH run, because continuing whatever is
    newest on disk from an unknown directory could append to someone else's audit
    trail. Naming the run is how the CLI already lets a caller who DOES know say so,
    and it is what a SLURM array task passes. Without it, clicking Ingest then Extract
    extracted from an empty run it had just minted, and reported success."""
    if stage not in STAGES:
        raise ValueError(f"{stage!r} is not a pipeline stage; expected one of {STAGES}")
    if new_run and run_id:
        # The CLI refuses this pair outright — one names a run that does not exist yet,
        # the other one that does — so building it would only fail later, at the child.
        raise ValueError("a new run and a named run are alternatives; pass one")
    command = [
        sys.executable, "-u", "-m", "apps_and_interfaces.command_line_interface", stage,
    ]
    if patient is not None:
        command += ["--patient", patient]
    if run_id:
        command += ["--run-id", run_id]
    if new_run:
        command += ["--new-run"]
    # Extract is the only stage with variables, and the CLI refuses --variable on the
    # others rather than ignoring it — so the guard belongs where the command is built,
    # not only in the caller that happens to know better.
    if stage == "extract":
        for variable in (variables or []):
            command += ["--variable", variable]
    return command


def command_for(input_folder: Path | str, patients: list[str], variables: list[str],
                data_root: Path | str) -> list[str]:
    """The exact `junior run` invocation the Run button performs.

    --output spells the config's own destination out loud — the app's promise that
    results land in the tree it reads — and the CLI treats a non-move --output as
    no change, so this continues the project's newest run exactly like a bare
    `junior run` typed at a shell."""
    command = [
        sys.executable, "-u", "-m", "apps_and_interfaces.command_line_interface", "run",
        str(input_folder),
        "--output", str(data_root),
    ]
    for patient in patients:
        command += ["--patient", patient]
    for variable in variables:
        command += ["--variable", variable]
    return command


def child_environment(data_root: Path | str) -> dict[str, str]:
    """The environment the `junior run` child gets.

    The child writes where the app reads, whatever the launch cwd was. When the
    app was opened by `junior workbench`, that command also pinned the project's
    config file in the environment (JUNIOR_CONFIG), so the child resolves the
    same project — inherited here through the environment copy."""
    environment = dict(os.environ)
    environment["JR_DATA_ROOT"] = str(data_root)
    # The app's promise is that its log panel never shows extracted values. That
    # promise must not depend on what happens to be exported in the server's own
    # shell — a JR_SHOW_STDOUT_VALUES=1 inherited from the operator's terminal
    # would put values into the tailed log, so the child never receives it.
    environment.pop("JR_SHOW_STDOUT_VALUES", None)
    # A source checkout that was never pip-installed still finds both the engine
    # (src/jr_pipeline) and the interfaces (apps_and_interfaces).
    roots = [str(REPO / "src"), str(REPO)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def stage_commands_for(stage: str, patients: list[str], every_patient: list[str],
                       run_id: str = "", variables: list[str] | None = None,
                       new_run: bool = False) -> list[list[str]]:
    """What a stage button actually runs.

    Every patient ticked means the plain whole-cohort command, not one invocation each:
    N subprocesses where one would do is N interpreter start-ups and N passes over the
    same cohort for the same work.

    Only extract is given the ticked variables — it is the only stage that has any,
    and stage_command_for is what enforces that."""
    chosen = list(variables or [])
    if set(patients) == set(every_patient):
        return [stage_command_for(stage, None, run_id, chosen, new_run)]
    # Only the first invocation opens the run; the rest must land in the one it made,
    # or "new run" would mean one run per patient.
    first = stage_command_for(stage, patients[0], run_id, chosen, new_run)
    rest = [stage_command_for(stage, patient, run_id, chosen) for patient in patients[1:]]
    return [first, *rest]


def start_stage(stage: str, patients: list[str], every_patient: list[str],
                data_root: Path | str, run_id: str = "",
                variables: list[str] | None = None, new_run: bool = False) -> StageRun:
    """Spawn one stage over the ticked patients and start tailing it."""
    commands = stage_commands_for(stage, patients, every_patient, run_id, variables, new_run)
    run = StageRun(stage=stage, patients=list(patients), commands=commands,
                   environment=child_environment(data_root))
    threading.Thread(target=run._walk, daemon=True).start()
    return run


def start_run(input_folder: Path | str, patients: list[str], variables: list[str],
              data_root: Path | str) -> PipelineRun:
    """Spawn `junior run` as a subprocess and start tailing it. Returns immediately."""
    command = command_for(input_folder, patients, variables, data_root)
    environment = child_environment(data_root)

    process = subprocess.Popen(
        command, cwd=str(REPO), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    run = PipelineRun(patients=list(patients), variables=list(variables), process=process)
    run.lines.append(f"$ {' '.join(command)}")
    threading.Thread(target=run._tail, daemon=True).start()
    return run
