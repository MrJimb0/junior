"""Shared path and write helper for pipeline artifacts.

Pipeline steps write JSON "envelopes" for things like retrieval results, variable
outputs, manifests, and step receipts. This module keeps those paths in one place:
callers say what kind of artifact they are writing, and ``artifact_path`` returns
the standard file path for it.

Most callers already have a run directory or patient directory, so they pass
``run_root`` or ``patient_root``. If they only have IDs, the helper can derive the
same directories from ``run_id``, ``patient_id``, and ``dr``. A few files are
sidecars that must live next to a data file; those callers pass an explicit
``path`` to ``write_artifact``.

``write_artifact`` does the common write-time checks: compute the content hash,
validate the envelope against its schema, and write the JSON atomically.
"""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    validate_artifact,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    extract_output_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
    retrieval_output_dir,
)


def artifact_path(
    artifact_type: str,
    *,
    run_id: str | None = None,
    dr: Path | None = None,
    run_root: Path | None = None,
    patient_root: Path | None = None,
    patient_id: str | None = None,
    variable: str | None = None,
    step_id: str | None = None,
    query_id: str | None = None,
    name: str | None = None,
) -> Path:
    """Where an artifact of this type belongs, from whichever identity the caller holds.

    Every argument is optional because callers hold different ones — one has the run
    directory already, another only the ids. Supplying neither joined a path onto None,
    so the run died on "unsupported operand type(s) for /: 'NoneType' and 'str'": a
    message that names neither the artifact being written nor the argument that would
    have fixed it, raised partway through writing a patient's results.

    The three lookups below are deliberately lazy, so an artifact type that needs only
    the run directory is never refused for want of a patient."""
    def _run_directory() -> Path:
        if run_root is not None:
            return run_root
        if run_id is not None:
            return phi_intermediate_run_dir(run_id, dr)
        raise ValueError(
            f"Nowhere to put this {artifact_type}: pass run_root, or pass run_id so "
            f"the run's folder can be worked out."
        )

    def _patient_directory() -> Path:
        if patient_root is not None:
            return patient_root
        if run_id is not None and patient_id is not None:
            return phi_patient_run_dir(run_id, patient_id, dr)
        raise ValueError(
            f"Nowhere to put this {artifact_type}: pass patient_root, or pass both "
            f"run_id and patient_id so the patient's folder can be worked out."
        )

    def _required(value: str | None, argument: str) -> str:
        if not value:
            raise ValueError(
                f"Nowhere to put this {artifact_type}: pass {argument}, which names "
                f"the folder it goes in."
            )
        return value

    if artifact_type == "manifest":
        return _run_directory() / "manifest.json"
    if artifact_type == "run_roster":
        return _run_directory() / "run_roster.json"
    if artifact_type == "run_metrics_shard":
        return _run_directory() / "data" / "eval" / f"{_required(name, 'name')}.json"
    if artifact_type == "source_snapshot":
        return _patient_directory() / "source_snapshot.json"
    if artifact_type == "retrieval":
        query = _required(query_id, "query_id").replace("/", "_")
        return retrieval_output_dir(_patient_directory()) / f"{query}.json"
    if artifact_type in ("variable_result", "invariants", "step_receipt"):
        variable_dir = (
            extract_output_dir(_patient_directory()) / _required(variable, "variable")
        )
        if artifact_type == "variable_result":
            return variable_dir / "result.json"
        if artifact_type == "invariants":
            return variable_dir / "invariants.json"
        return variable_dir / "steps" / _required(step_id, "step_id") / "receipt.json"

    raise KeyError(f"no registry path for artifact_type {artifact_type!r}")


def write_artifact(
    envelope: dict,
    *,
    dr: Path | None = None,
    path: Path | None = None,
    run_root: Path | None = None,
    patient_root: Path | None = None,
    schema_name: str | None = None,
    query_id: str | None = None,
    name: str | None = None,
) -> Path:
    """Hash, validate, and atomically write an envelope; return the path.

    The destination is resolved from the registry by ``artifact_type`` + the supplied
    base (``run_root``/``patient_root``) or the envelope's ``produced_by`` identity,
    unless an explicit ``path`` is given (sidecars). ``schema_name`` overrides the
    validation schema (e.g. ``run_metrics_shard`` validates against ``artifact_envelope``)."""
    artifact_type = envelope["artifact_type"]
    if path is None:
        pb = envelope.get("produced_by", {})
        path = artifact_path(
            artifact_type,
            run_id=pb.get("run_id"),
            dr=dr,
            run_root=run_root,
            patient_root=patient_root,
            patient_id=pb.get("patient_id"),
            variable=pb.get("variable"),
            step_id=pb.get("step_id"),
            query_id=query_id,
            name=name,
        )
    envelope["content_hash"] = hash_artifact_payload(envelope)
    validate_artifact(envelope, schema_name or artifact_type)
    return atomic_write_json(path, envelope)
