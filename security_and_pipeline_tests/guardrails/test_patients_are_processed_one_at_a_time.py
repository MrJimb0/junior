"""Guardrail: a cohort is walked one patient at a time, in one process.

Every stage holds a model on the device while it works — the encoder for embed, the
encoder AND the extraction LLM for extract. One extraction was measured at 11.8 GiB of
live tensors, and torch's allocator held 47 GiB more in transient buffers it had not yet
returned. Two patients in flight at once on a laptop is not twice the work, it is the MPS
high-watermark and then every allocation in the process failing, including the small one
the query encoder needs. That failure does not present as "out of memory" — it presents
as retrieval quietly returning nothing for charts that are perfectly fine.

Fanning out is the right thing to do where there is hardware for it, and Junior already
supports it the only way that is safe: SEPARATE PROCESSES, one per patient, each with
`--patient`, which is what the SLURM array scripts under deployment/ submit. Each has its
own allocator and its own ceiling, and the scheduler — not Junior — decides how many run
at once against the GPUs actually available.

So in-process concurrency over patients is not an optimisation waiting to be made. This
guardrail exists because it will look like one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Where a cohort is walked. Adding a fourth is fine; leaving it out of this list is not.
COHORT_WALKERS = [
    REPO / "apps_and_interfaces/command_line_interface.py",
    REPO / "src/jr_pipeline/runtime_infrastructure/cohort_runner.py",
    REPO / "apps_and_interfaces/shiny_review_app/cohort_ingest.py",
]

# Names that put more than one patient in flight inside one process.
CONCURRENCY = {
    "ThreadPoolExecutor", "ProcessPoolExecutor", "ThreadPool", "Pool",
    "multiprocessing", "concurrent", "joblib", "Parallel", "dask", "ray",
    "asyncio", "gather", "as_completed",
}


@pytest.mark.parametrize("path", COHORT_WALKERS, ids=lambda p: p.name)
def test_no_in_process_fan_out_over_patients(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offending: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in CONCURRENCY:
                    offending.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in CONCURRENCY:
                offending.append(f"line {node.lineno}: from {node.module} import ...")
            for alias in node.names:
                if alias.name in CONCURRENCY:
                    offending.append(f"line {node.lineno}: imports {alias.name}")

    assert not offending, (
        f"{path.relative_to(REPO)} walks a cohort, and now reaches for concurrency:\n  "
        + "\n  ".join(offending)
        + "\nTwo patients at once share one allocator and one device ceiling. Fan out "
          "with separate processes (`--patient`), the way the SLURM array scripts do."
    )


def test_the_cluster_scripts_fan_out_by_process_not_by_thread():
    """The supported fan-out, and the proof that in-process concurrency is unnecessary:
    each array task is its own process, handed one patient."""
    slurm = REPO / "deployment/Local_SLURM_Cluster/slurm"
    if not slurm.is_dir():
        pytest.skip("no cluster scripts in this checkout")

    def narrows_to_one_patient(body: str) -> bool:
        # `--patients-file` contains `--patient`. seal.sh takes the whole roster once on
        # the login node and is not an array job, which is correct and must not be
        # mistaken for a fan-out.
        return "--patient" in body.replace("--patients-file", "")

    fanned = [
        script for script in sorted(slurm.glob("*.sh"))
        if narrows_to_one_patient(script.read_text(encoding="utf-8"))
    ]
    assert fanned, "no cluster script passes --patient, so nothing fans out per patient"
    for script in fanned:
        body = script.read_text(encoding="utf-8")
        assert "--array" in body or "SLURM_ARRAY_TASK_ID" in body, (
            f"{script.name} narrows to one patient but is not an array job, so nothing "
            "schedules the others"
        )
