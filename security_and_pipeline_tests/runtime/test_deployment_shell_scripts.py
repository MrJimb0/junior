"""Guardrails for the SLURM deployment-example shell scripts.

Every shipped `.sh` must pass `bash -n` so a syntax error never reaches an operator.
The stage scripts must trim surrounding whitespace from a patient id without using the
`${PATIENT##[[:space:]]*}` idiom, which deletes the WHOLE value for any indented patient
id (e.g. "  P123" -> "") and aborts the task.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SLURM_DIR = REPO / "deployment" / "Local_SLURM_Cluster" / "slurm"
STAGE_SCRIPTS = ["ingest.sh", "embed.sh", "index.sh"]
ALL_SCRIPTS = sorted(SLURM_DIR.glob("*.sh"))

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def test_every_deployment_script_passes_bash_syntax_check():
    """`bash -n` on every .sh — catches any syntax error in a shipped operator script."""
    assert ALL_SCRIPTS, "no deployment shell scripts found"
    for script in ALL_SCRIPTS:
        proc = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, f"{script.name} failed bash -n: {proc.stderr}"


def test_stage_scripts_do_not_use_the_value_eating_trim():
    """The `##[[:space:]]*` idiom removes the longest whitespace-then-anything
    prefix -> it eats the entire value of any indented id, so it must not be used."""
    for name in STAGE_SCRIPTS:
        text = (SLURM_DIR / name).read_text(encoding="utf-8")
        assert "##[[:space:]]*" not in text, f"{name} still uses the value-eating trim idiom"


@pytest.mark.parametrize(
    "raw, expected",
    [("  P123", "P123"), ("\tP9", "P9"), ("P123", "P123"), ("P9 ", "P9"), ("   ", "")],
)
def test_stage_scripts_trim_surrounding_whitespace_without_eating_the_id(raw, expected):
    """Run each stage script's actual PATIENT-trim line on edge inputs: an indented
    id must keep its value; a whitespace-only line must become blank."""
    for name in STAGE_SCRIPTS:
        trim_line = next(
            (ln for ln in (SLURM_DIR / name).read_text(encoding="utf-8").splitlines()
             if "sed -E" in ln and "[[:space:]]" in ln and "PATIENT=" in ln),
            None,
        )
        assert trim_line is not None, f"{name} has no whitespace-trim line"
        snippet = f'PATIENT={shlex.quote(raw)}\n{trim_line}\nprintf "%s" "$PATIENT"'
        proc = subprocess.run([bash, "-c", snippet], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == expected, f"{name}: {raw!r} -> {proc.stdout!r}, expected {expected!r}"


def test_usage_examples_reference_the_real_repo_relative_path():
    """Every ``bash``/``sbatch`` example must name the script at its real repo path
    (``deployment/Local_SLURM_Cluster/slurm/``), not the stale ``scripts/`` prefix
    an operator would copy-paste into a command that fails."""
    for script in ALL_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "scripts/" not in text, f"{script.name} still uses the stale 'scripts/' path prefix"


def test_embed_does_not_export_a_device_env_the_pipeline_ignores():
    """The encoder selects its device from the embed config's ``device`` key, not from
    any env var. A ``JR_DEVICE`` export would read as control it does not have."""
    assert "JR_DEVICE" not in (SLURM_DIR / "embed.sh").read_text(encoding="utf-8")


def test_rclone_push_reads_from_the_runs_output_root():
    """LOCAL_OUT must inherit JR_OUT_ROOT (the root the pipeline actually wrote to)
    so ``push`` never reads a stale default when the operator relocated the output."""
    text = (SLURM_DIR / "rclone_sync.sh").read_text(encoding="utf-8")
    assert "${LOCAL_OUT:-${JR_OUT_ROOT:-" in text
