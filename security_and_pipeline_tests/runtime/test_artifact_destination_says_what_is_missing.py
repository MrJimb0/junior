"""An artifact with nowhere to go says which argument was left out.

Every base and identity argument to ``artifact_path`` is optional, because different
callers hold different ones — a step that has the run directory, a step that only has
the ids. A caller that supplies neither used to join a path onto ``None``, so the run
died on "unsupported operand type(s) for /: 'NoneType' and 'str'": a message that names
neither the artifact being written nor the argument that would have fixed it, raised
partway through writing a patient's results.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path, write_artifact


@pytest.mark.parametrize(
    ("artifact_type", "kwargs"),
    [
        ("manifest", {}),
        ("run_roster", {}),
        ("run_metrics_shard", {"name": "coverage"}),
        ("source_snapshot", {}),
        ("retrieval", {"query_id": "q1"}),
        ("variable_result", {"variable": "clinical_stage"}),
        ("invariants", {"variable": "clinical_stage"}),
        ("step_receipt", {"variable": "clinical_stage", "step_id": "retrieve_and_prompt"}),
    ],
)
def test_no_destination_names_the_artifact_and_what_to_pass(artifact_type, kwargs):
    with pytest.raises(ValueError) as raised:
        artifact_path(artifact_type, **kwargs)
    message = str(raised.value)
    assert artifact_type in message
    assert "pass" in message  # tells the caller what to supply, not just that it failed


def test_a_result_with_no_variable_name_says_so(tmp_path: Path):
    """The other half: the base is there, but the identity that names the folder is not.
    This one arrives through the front door — an envelope whose produced_by block is
    missing a field — so the message has to name the field, not the path fragment."""
    envelope = {
        "artifact_type": "variable_result",
        "produced_by": {"run_id": "20260101_run", "patient_id": "Patient_1"},
    }
    with pytest.raises(ValueError, match="variable"):
        write_artifact(envelope, patient_root=tmp_path)


def test_a_step_receipt_with_no_step_id_says_so(tmp_path: Path):
    with pytest.raises(ValueError, match="step_id"):
        artifact_path("step_receipt", patient_root=tmp_path, variable="clinical_stage")


def test_an_unregistered_artifact_type_still_raises_key_error(tmp_path: Path):
    """The pre-existing contract for a type with no registry entry stays a KeyError, so
    callers that distinguish "no such artifact type" from "nowhere to put it" still can."""
    with pytest.raises(KeyError):
        artifact_path("not_an_artifact", run_root=tmp_path)
