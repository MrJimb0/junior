"""[PHI]: the sealed entry_point.json redacts the --patient argv.

A raw patient id passed as ``--patient <id>`` must not enter the shareable-post-scrub
code bundle, so the seal redacts it.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    _redact_patient_argv,
)


def test_patient_value_is_redacted_from_argv():
    ep = {"run_id": "r", "argv": ["jr-pipeline", "extract", "--patient", "STSS0123abc", "--config", "x.yaml"]}
    out = _redact_patient_argv(ep)
    assert "STSS0123abc" not in out["argv"]
    assert out["argv"][out["argv"].index("--patient") + 1] == "<redacted-patient>"
    assert "--config" in out["argv"] and "x.yaml" in out["argv"]  # other args untouched
    assert ep["argv"][3] == "STSS0123abc"  # the original is not mutated


def test_patient_equals_form_is_redacted():
    out = _redact_patient_argv({"argv": ["x", "--patient=STSS999"]})
    assert "STSS999" not in str(out["argv"])
    assert "--patient=<redacted-patient>" in out["argv"]


def test_no_argv_is_a_noop():
    assert _redact_patient_argv({"run_id": "r"}) == {"run_id": "r"}
