"""`junior run` and `junior extract` must extract under the same settings.

Extraction is reached two ways: the per-stage command hands it the loaded config, and
`junior run` hands it a config rebuilt from CohortSettings out of a hand-written set of
keys. Anything missing from that set is dropped in silence.

That is not hypothetical. One real run extracted the same variables for the
same patient twice in the same run — once at 7 chunks per prompt, once at 15 and 16 —
because max_chunks_per_prompt reached one path and not the other. It cost a 20 minute
re-extraction: attention is quadratic in prompt length, so the uncapped pass was far
slower, and every prompt differed from the capped one, so not a single response came
back from the model cache. The run's own sealed config named a ceiling that half of it
had never applied.
"""
from __future__ import annotations

import pytest

from apps_and_interfaces.command_line_interface import _RUN_INVARIANT_CFG_KEYS
from jr_pipeline.runtime_infrastructure.cohort_runner import (
    EXTRACT_EXECUTION_CFG_KEYS,
    _build_step_configs,
    cohort_settings_from_config,
)

_A_COHORT = {
    "project": "t", "run_id": "20990101_000000_aa",
    "source_root": "/charts", "output_root": "/out",
    "recipes": ["stage"], "recipes_root": "/recipes",
    "encoder": {"model_id": "m"}, "chunker": {"kind": "token_window"},
    "allowlist_path": "/allow.yaml", "llm_mode": "api",
}


@pytest.mark.parametrize("key", EXTRACT_EXECUTION_CFG_KEYS)
def test_each_execution_setting_survives_the_run_path(key):
    cfg = {**_A_COHORT, key: 7}

    built = _build_step_configs(cohort_settings_from_config(cfg), cfg["run_id"])

    assert built["extract"].get(key) == 7, (
        f"{key} is set in the config and never reaches extract via `junior run`, so "
        "that command extracts under different settings than `junior extract` does"
    )


def test_a_setting_that_is_absent_stays_absent():
    """Carrying it as None would override a default with 'unset' rather than leaving
    the default alone."""
    built = _build_step_configs(cohort_settings_from_config(dict(_A_COHORT)), "20990101_000000_aa")

    for key in EXTRACT_EXECUTION_CFG_KEYS:
        assert key not in built["extract"]


def test_every_run_invariant_execution_setting_is_carried():
    """A run is FIXED to its run-invariant settings — that is the promise the drift gate
    enforces and the sealed config records. A run-invariant setting that reaches only one
    of the two commands makes that promise false for the other, which is the exact shape
    of the bug this file exists for.

    Keys naming paths, recipes or the run itself are excluded: those are rebuilt from
    CohortSettings by name and are not execution settings."""
    structural = {"run_id", "output_root", "allowlist_path", "recipes_root"}

    should_be_carried = set(_RUN_INVARIANT_CFG_KEYS) - structural

    assert should_be_carried <= set(EXTRACT_EXECUTION_CFG_KEYS), (
        f"run-invariant but dropped on the `junior run` path: "
        f"{sorted(should_be_carried - set(EXTRACT_EXECUTION_CFG_KEYS))}"
    )
