"""The column list a recipe's sealed output schema implies.

A run's answer table used to get its header from the answers: walk the rows, append
each key not seen before. That makes the header a property of what the model happened
to return, so the same recipe over a different cohort produces a different column set
in a different order, and two such files do not stack. It also puts a patient's extra
keys after the provenance block, because that patient sorted second.

Every recipe ships a JSON Schema for its output, sealed into the run's code bundle and
fingerprinted as ``schema_hash`` on every result envelope. That schema is the recipe's
promise about its shape, and it is the same promise for every patient and every run of
that recipe version. Deriving the header from it fixes the set, fixes the order, and
distinguishes "the recipe has no such field" from "this patient has no such value" --
the second is an empty cell, the first is no column at all.

WHAT AN UNDECLARED KEY GETS. Seventeen of the twenty-one recipes carry
``additionalProperties: true`` somewhere, so a model may legitimately return a key the
schema never named. Such a key does NOT get a column of its own: a column that appears
only when a model volunteers something is the instability this module exists to remove,
and it would break stacking again the first time it happened. It is not dropped either
-- silence is what makes a reader trust a table that is missing something. Every
undeclared key lands, path and value, in the single ``fields_not_in_the_recipe_schema``
column, which is always present and usually empty. Declared but shapeless objects
(``pretreatment_path_stage`` is declared ``{"type": "object"}`` with no properties) keep
one column of their own holding their JSON, because the recipe did name them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# A recipe that answers with a table of records -- one pathology report per row -- puts
# the records in a top-level array of objects. Those become the table's ROWS, so their
# fields are columns of a different grain from the patient-level ones and are tracked
# apart.
@dataclass(frozen=True)
class RecipeColumns:
    """The columns one recipe's output schema promises, in declaration order."""

    patient_level: tuple[str, ...]
    row_axis_field: str | None
    per_row: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.patient_level or self.per_row)


def _deref(node: Any, defs: dict[str, Any], seen: frozenset[str]) -> tuple[Any, frozenset[str]]:
    """Follow ``$ref`` into ``$defs``. A ref already followed on this path is left
    unresolved, so a self-referential schema ends as one cell instead of recursing."""
    while isinstance(node, dict) and "$ref" in node:
        ref = str(node["$ref"])
        if ref in seen or not ref.startswith("#/$defs/"):
            return {}, seen
        seen = seen | {ref}
        node = defs.get(ref[len("#/$defs/"):], {})
    return node, seen


def _merged_properties(node: dict, defs: dict[str, Any], seen: frozenset[str]) -> dict[str, Any]:
    """The properties of a node, with ``allOf`` branches merged in declaration order.

    ``allOf: [{$ref: component_base}, {properties: {value: <enum>}}]`` is how the
    recipes narrow a shared shape; the base names the fields and the second branch only
    constrains one of them, so a plain union in order gives the base's field order."""
    properties: dict[str, Any] = {}
    for branch in node.get("allOf") or []:
        resolved, branch_seen = _deref(branch, defs, seen)
        if isinstance(resolved, dict):
            properties.update(_merged_properties(resolved, defs, branch_seen))
    properties.update(node.get("properties") or {})
    return properties


def _without_null(node: dict, defs: dict[str, Any], seen: frozenset[str]) -> list[Any]:
    """The branches of a ``oneOf``/``anyOf`` that carry a shape.

    ``oneOf: [{object...}, {"type": "null"}]`` is how every recipe says "this may be
    absent". Taking the null branch as a shape of its own is what produced the phantom
    bare ``clinical_tnm`` column beside ``clinical_tnm.cT``: one patient returned the
    object and one returned null, and the header carried both readings."""
    branches: list[Any] = []
    for branch in (node.get("oneOf") or []) + (node.get("anyOf") or []):
        resolved, _ = _deref(branch, defs, seen)
        if isinstance(resolved, dict) and resolved.get("type") != "null":
            branches.append(branch)
    return branches


def _is_object(node: dict, properties: dict[str, Any]) -> bool:
    declared = node.get("type")
    types = declared if isinstance(declared, list) else [declared]
    return bool(properties) or "object" in types


def _is_array(node: dict) -> bool:
    declared = node.get("type")
    types = declared if isinstance(declared, list) else [declared]
    return "array" in types or "items" in node


def _leaf_columns(node: Any, defs: dict[str, Any], prefix: str, seen: frozenset[str]) -> list[str]:
    """The dotted column paths one schema node contributes, in declaration order."""
    node, seen = _deref(node, defs, seen)
    if not isinstance(node, dict):
        return [prefix] if prefix else []

    branches = _without_null(node, defs, seen)
    if branches:
        # More than one non-null branch is a genuine union of shapes; every branch's
        # columns are declared, so all of them get one, first spelling wins.
        columns: list[str] = []
        for branch in branches:
            for column in _leaf_columns(branch, defs, prefix, seen):
                if column not in columns:
                    columns.append(column)
        return columns

    properties = _merged_properties(node, defs, seen)
    if _is_object(node, properties):
        if not properties:
            return [prefix] if prefix else []  # declared, shape left open: one JSON cell
        columns = []
        for name, child in properties.items():
            columns += _leaf_columns(child, defs, f"{prefix}.{name}" if prefix else name, seen)
        return columns
    if _is_array(node):
        # A list of scalars is one cell. A list of records nested BELOW the row axis has
        # no row of its own to occupy, so it is one cell of JSON rather than dropped.
        return [prefix] if prefix else []
    return [prefix] if prefix else []


def _record_array_fields(properties: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    """Top-level fields declared as an array of objects -- the row-axis candidates."""
    found: list[str] = []
    for name, child in properties.items():
        node, seen = _deref(child, defs, frozenset())
        if not isinstance(node, dict) or not _is_array(node):
            continue
        items, _ = _deref(node.get("items") or {}, defs, seen)
        if isinstance(items, dict) and _is_object(items, _merged_properties(items, defs, seen)):
            found.append(name)
    return found


def columns_from_schema(schema: dict[str, Any]) -> RecipeColumns:
    """The ordered columns a recipe's output schema promises.

    Exactly one top-level array of objects is the row axis; its fields are per-row
    columns. Two of them is an ambiguity the table names rather than resolves, so
    neither is taken as the axis here and both stay patient-level JSON cells."""
    defs = schema.get("$defs") or {}
    properties = _merged_properties(schema, defs, frozenset())
    record_fields = _record_array_fields(properties, defs)
    row_axis = record_fields[0] if len(record_fields) == 1 else None

    patient_level: list[str] = []
    per_row: list[str] = []
    for name, child in properties.items():
        if name == row_axis:
            node, seen = _deref(child, defs, frozenset())
            per_row = _leaf_columns(node.get("items") or {}, defs, "", seen)
            continue
        patient_level += _leaf_columns(child, defs, name, frozenset())
    return RecipeColumns(tuple(patient_level), row_axis, tuple(per_row))


def sealed_schema_for(run_root: Path, variable: str, version: str) -> tuple[dict[str, Any] | None, Path | None]:
    """The output schema sealed into this run's code bundle for one recipe.

    Read from the run's own copy rather than from the working tree, so a table rebuilt
    months later gets the shape the run actually promised and not the shape the recipe
    has since grown. The collection folder is not recorded anywhere on the result, so
    the recipe is found by its variable/version leaf."""
    matches = sorted(
        Path(run_root).glob(f"code/recipes/**/{variable}/{version}/{variable}_{version}_output_schema.json")
    )
    if len(matches) != 1:
        # Two collections shipping the same variable name is what
        # test_recipe_collection_uniqueness forbids, so this is a damaged bundle.
        return None, None
    try:
        return json.loads(matches[0].read_text(encoding="utf-8")), matches[0]
    except (OSError, json.JSONDecodeError):
        return None, None


def sealed_schema_matches(schema_path: Path, expected_hash: str | None) -> bool:
    """Whether the sealed schema on disk is the one the result says it ran against.

    A run whose bundle was re-sealed after the fact still holds a schema, and it may be
    a different one; deriving a header from it and saying nothing would put the wrong
    promise on the file. The caller records which schema the header came from."""
    if not expected_hash:
        return False
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (  # noqa: E501
        schema_hash,
    )

    try:
        return schema_hash(schema_path) == expected_hash
    except (OSError, ValueError):
        return False
