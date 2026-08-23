"""Schema registry and validation.

Loads JSON Schemas from ``schemas/json/`` and validates every artifact the
pipeline writes. Schema-dir existence is checked lazily so `import jr_pipeline`
and `--help` work even if package data is missing.
"""
from __future__ import annotations

import json
from datetime import UTC
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

SCHEMAS_DIR = Path(__file__).parent / "json"

# The project format checker. Without it jsonschema ``format`` constraints (date,
# date-time, email, ...) are decorative and never enforced; it is passed to
# every validator so a bad-format value fails validation.
FORMAT_CHECKER = FormatChecker()


@cache
def load_schema(name: str) -> dict:
    """Load ``schemas/json/<name>.schema.json`` (name without extension)."""
    if not SCHEMAS_DIR.is_dir():
        raise FileNotFoundError(
            f"Schemas directory missing from package install: {SCHEMAS_DIR}. "
            "Check pyproject.toml [tool.setuptools.package-data] includes "
            "jr_pipeline/runtime_enforcing_safety_and_reproducibility/schemas/json/*.schema.json."
        )
    path = SCHEMAS_DIR / f"{name}.schema.json"
    if not path.is_file():
        raise FileNotFoundError(f"Schema not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@cache
def _registry() -> Registry:
    """Registry of every shipped schema, keyed by file:// and canonical https:// URIs so $refs resolve either way."""
    reg = Registry()
    for path in SCHEMAS_DIR.glob("*.schema.json"):
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        # The suppression below is there because the checker is wrong, not because the
        # call is. `referencing.Resource` keeps the specification in a private attribute
        # and its generated constructor takes it under the public name; the checker
        # reads the attribute name and reports both keywords as unknown.
        # inspect.signature(Resource.__init__) is (self, contents, specification), and
        # this call builds every schema in the registry the pipeline validates against.
        resource = Resource(contents=doc, specification=DRAFT202012)  # type: ignore[call-arg]
        reg = reg.with_resource(uri=f"file://{path.as_posix()}", resource=resource)
        reg = reg.with_resource(
            uri=f"https://jr-pipeline/schemas/{path.name}", resource=resource
        )
    return reg


@cache
def _validator_for(schema_name: str) -> Draft202012Validator:
    schema = load_schema(schema_name)
    return Draft202012Validator(schema, registry=_registry(), format_checker=FORMAT_CHECKER)


def validate_artifact(artifact: dict, schema_name: str) -> None:
    """Validate ``artifact`` against ``<schema_name>.schema.json``; raises ``jsonschema.ValidationError`` on failure."""
    _validator_for(schema_name).validate(artifact)


def iter_validation_errors(artifact: dict, schema_name: str) -> list[ValidationError]:
    """Return all validation errors without raising."""
    return sorted(
        _validator_for(schema_name).iter_errors(artifact),
        # Stringify path segments -- mixed str-key/int-index paths raise
        # TypeError when compared element-wise at a str-vs-int position.
        key=lambda e: ([str(p) for p in e.absolute_path], e.message),
    )


def known_schemas() -> list[str]:
    """Sorted list of every schema name in SCHEMAS_DIR."""
    return sorted(p.stem.replace(".schema", "") for p in SCHEMAS_DIR.glob("*.schema.json"))


def envelope_for(
    *,
    artifact_type: str,
    sensitivity: str,
    stream: str,
    run_id: str,
    step: str,
    payload: Any,
    patient_id: str | None = None,
    variable: str | None = None,
    step_id: str | None = None,
    parent_artifacts: list[dict] | None = None,
    code_lock_hash: str | None = None,
    code_version: str | None = None,
    created_at: str | None = None,
    **extra_produced_by: Any,
) -> dict:
    """Build an artifact envelope with the canonical produced_by shape. Caller fills ``content_hash`` via ``content_hash.hash_artifact_payload``."""
    from datetime import datetime

    produced_by = {
        "run_id": run_id,
        "patient_id": patient_id,
        "step": step,
        "variable": variable,
        "step_id": step_id,
        "code_lock_hash": code_lock_hash,
        "code_version": code_version,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "recipe_hash": None,
        "schema_hash": None,
        "prompt_hash": None,
        "python_helpers_hash": None,
        "provider_config_hash": None,
        "retrieval_config_hash": None,
    }
    for k, v in extra_produced_by.items():
        produced_by[k] = v

    # Compute the real content_hash here so a built-and-validated envelope always
    # carries a true hash, never a schema-valid all-zeros placeholder that would still
    # pass validation -- a silent integrity hazard. hash_artifact_payload excludes
    # content_hash + created_at, so it is deterministic; a caller that re-stamps it after
    # mutating the payload (e.g. mark_completed) recomputes the same way.
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
        hash_artifact_payload,
    )

    envelope = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "sensitivity": sensitivity,
        "stream": stream,
        "produced_by": produced_by,
        "parent_artifacts": parent_artifacts or [],
        "content_hash": "sha256:" + ("0" * 64),  # replaced immediately below
        "payload": payload,
    }
    envelope["content_hash"] = hash_artifact_payload(envelope)
    return envelope
