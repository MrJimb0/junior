"""JSON-schema validation for variable outputs.

After a recipe finishes extracting a variable, the extract runner calls
validate_output to confirm the assembled answer matches the recipe's declared
output_schema — for example that "stage" is a string and "date_of_death" looks
like a date.

A schema failure is non-fatal: the runner notes it under `validation` in the
result record so a reviewer can spot a misshapen answer, without aborting the
whole batch of patients.

Each schema's validator is cached by its absolute file path, so running the
same recipe across many patients pays the one-time cost of reading and parsing
that schema only once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    FORMAT_CHECKER,
)


@dataclass(frozen=True)
class SchemaValidationReport:
    """result of a schema check; each error names the offending field as a
    dotted path (e.g. "stage_at_diagnosis.overall_group: ...")."""

    ok: bool
    errors: list[str]


@lru_cache(maxsize=128)
def _validator(path: str) -> Draft202012Validator:
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    # Hand in the project's FORMAT_CHECKER so that when a schema says a field has
    # a "format" (date / date-time / email), that format is actually enforced
    # rather than being treated as a decorative annotation.
    return Draft202012Validator(schema, format_checker=FORMAT_CHECKER)


def validate_output(data: object, schema_path: Path) -> SchemaValidationReport:
    """validate data against the json schema at schema_path; never raises."""
    if data is None:
        return SchemaValidationReport(ok=False, errors=["no parsed data to validate"])
    v = _validator(str(Path(schema_path).resolve()))
    # Turn every path segment into a string before sorting. A field's location can
    # mix string keys with integer list positions, and sorting a list of such mixed
    # paths would crash when it tries to compare a string against an integer.
    # Stringifying first gives a stable, repeatable order without that crash.
    errors = sorted(v.iter_errors(data), key=lambda e: ([str(p) for p in e.absolute_path], e.message))
    return SchemaValidationReport(
        ok=not errors,
        errors=[f"{'.'.join(str(p) for p in e.absolute_path) or '$'}: {e.message}" for e in errors],
    )
