"""jsonschema format constraints are enforced, not decorative.

The project FORMAT_CHECKER is passed to every validator, so "format": "date" / "email"
rejects a bad value.
"""
from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    FORMAT_CHECKER,
    _validator_for,
)


def test_format_checker_rejects_a_bad_date():
    schema = {"type": "object", "properties": {"d": {"type": "string", "format": "date"}}}
    validator = Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    assert validator.is_valid({"d": "2024-01-15"})
    assert not validator.is_valid({"d": "not-a-date"})


def test_format_checker_rejects_a_bad_email():
    schema = {"type": "object", "properties": {"e": {"type": "string", "format": "email"}}}
    validator = Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    assert validator.is_valid({"e": "a@b.com"})
    assert not validator.is_valid({"e": "not-an-email"})


def test_format_checker_enforces_date_time():
    # jsonschema only registers the RFC3339 date-time checker when a validator package
    # is installed. jsonschema[format-nongpl] (ADR 0034) provides it, so the timestamp
    # fields' `format: date-time` constraints are enforced. Asserted unconditionally
    # because the extra is a declared core dependency.
    assert "date-time" in FORMAT_CHECKER.checkers, "date-time checker not registered"
    schema = {"type": "object", "properties": {"t": {"type": "string", "format": "date-time"}}}
    validator = Draft202012Validator(schema, format_checker=FORMAT_CHECKER)
    # the writer form (tz-aware UTC) and the log form (Z) are valid RFC3339
    assert validator.is_valid({"t": "2026-06-18T21:09:15.120536+00:00"})
    assert validator.is_valid({"t": "2026-06-18T21:09:15Z"})
    # a junk string and a BARE clinical date are NOT date-times -> rejected
    assert not validator.is_valid({"t": "not-a-datetime"})
    assert not validator.is_valid({"t": "2026-02-15"})


def test_project_validators_carry_the_format_checker():
    assert _validator_for("index_meta").format_checker is not None


def test_recipe_output_schema_enforces_format(tmp_path):
    # the SECOND validation call site: a recipe's output_schema. Drives the real
    # production entry point (validate_output), so it fails if the format_checker is
    # dropped there — unlike the two tests above, which build their own validators.
    from jr_pipeline.pipeline_steps.step_7_extract_variables.validate_recipe_output_schema import (
        validate_output,
    )

    schema_path = tmp_path / "out.schema.json"
    schema_path.write_text(
        json.dumps({"type": "object", "properties": {"d": {"type": "string", "format": "date"}}})
    )
    assert validate_output({"d": "2024-01-15"}, schema_path).ok
    assert not validate_output({"d": "not-a-date"}, schema_path).ok
