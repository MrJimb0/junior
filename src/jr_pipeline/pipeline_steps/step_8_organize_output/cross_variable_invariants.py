"""Step 8: cross-variable clinical invariants -- consistency checks that span
several extracted variables at once (the only checker that needs no language model).

A "clinical invariant" is a rule that must hold across a patient's whole extracted
record -- e.g. a date of death cannot precede a date of diagnosis, or a recurrence
implies a prior primary. Step 8 owns the post-extraction organization. It runs the
shared clinical invariants over all of a patient's variable results, writes the
patient-identifiable ``clinical_invariants.json`` (CONTAINS_PHI side), and emits a
text-free, de-identified (NO_PHI) summary "weak label". This module is pure (no model
calls) and advisory: per architecture decision ADR 0025 it never fails the run, it
only records violations for review.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_execution_order import _recipe_path
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_log = get_logger("cross_variable_invariants")


class InvariantDefinitionError(RuntimeError):
    """A clinical-invariant definition points at a recipe/field that does not exist.

    This is a code/config bug (the rule would otherwise silently "pass" because it
    checks nothing -- the failure mode the ``_VAR`` name table exists to prevent), not
    patient data, so the runner fails loudly on it rather than swallowing it the way it
    would a rule-execution crash."""


def organize_clinical_invariants(
    *,
    run_id: str,
    patient_id: str,
    patient_out: Path,
    recipes_root: Path,
    results: dict[str, Any],
) -> int:
    """Run the cross-variable invariants over ``results``, write the
    patient-identifiable clinical_invariants.json, and emit the text-free,
    de-identified (NO_PHI) summary label. Returns the violation count. Never fails
    the run -- invariants are a safety-net check, not a gate that blocks output."""
    outcomes = run_cross_variable_invariants(recipes_root, results)
    n_violations = sum(1 for o in outcomes if not o.get("ok"))
    if outcomes:
        atomic_write_json(patient_out / "clinical_invariants.json", {
            "patient_id": patient_id,
            "run_id": run_id,
            "n_violations": n_violations,
            "outcomes": outcomes,
        })
        try:
            from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
                record_invariant_outcomes,
            )
            record_invariant_outcomes(run_id=run_id, patient_id=patient_id, outcomes=outcomes)
        except Exception as e:  # noqa: BLE001 — recording this telemetry is non-fatal
            _log.warning("invariant_exhaust_failed", extra_={"error": str(e)})
        if n_violations:
            _log.info("clinical_invariant_violations", extra_={"n": n_violations})
    return n_violations


def validate_invariant_targets(mod: Any, recipes_root: Path) -> list[str]:
    """Startup assertion: every ``(role, recipe_variable, output_field)`` the
    invariant module declares via ``invariant_target_specs()`` must point at a recipe
    that actually exists on disk and whose output schema declares the named field.
    Returns human-readable problems; ``[]`` means every active invariant points at a
    real field. Roles not yet ready are excluded by the module itself
    (``_PENDING_5B``)."""
    problems: list[str] = []
    for role, var, field_name in mod.invariant_target_specs():
        recipe_path = _recipe_path(recipes_root, var)
        if not recipe_path.is_file():
            problems.append(f"role {role!r}: recipe {var!r} has no recipe file on disk")
            continue
        try:
            spec = load_recipe(recipe_path)
            schema = json.loads(spec.output_schema_path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — unreadable schema is itself the problem
            problems.append(f"role {role!r}: recipe {var!r} output schema unreadable: {e}")
            continue
        props = schema.get("properties") or {}
        if field_name not in props:
            problems.append(
                f"role {role!r}: field {field_name!r} not declared in {var!r} output schema"
            )
    return problems


def run_cross_variable_invariants(recipes_root: Path, results: dict[str, Any]) -> list[dict[str, Any]]:
    """Load the shared clinical invariants (``_shared_validation_rules/clinical_invariants.py``)
    and run them over every variable result. Returns JSON outcomes, or [] if the
    module is absent/unloadable (it must never fail an extraction). A dangling invariant
    DEFINITION (a rule pointing at a field/recipe that does not exist), by contrast,
    raises loudly -- that is a code bug, not patient data."""
    recipes_root = Path(recipes_root)
    mod_path = recipes_root / "_shared_validation_rules" / "clinical_invariants.py"
    if not mod_path.is_file():
        return []
    try:
        name = "jr_clinical_invariants"
        spec = importlib.util.spec_from_file_location(name, mod_path)
        if spec is None or spec.loader is None:
            # Python declines to describe how to import this file (e.g. the name is not
            # one it recognizes as a module). Same outcome as an unloadable module: the
            # safety-net check is skipped, the extraction stands.
            return []
        mod = importlib.util.module_from_spec(spec)
        # Register before exec: the module defines a @dataclass, whose machinery looks
        # the module up in sys.modules during class creation.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 — an unloadable module just skips this safety-net check
        return []
    # Startup assertion: an invalid rule definition is a code/config error -> fail loud.
    problems = validate_invariant_targets(mod, recipes_root)
    if problems:
        raise InvariantDefinitionError(
            "clinical invariant definitions reference recipes/fields that do not exist:\n  "
            + "\n  ".join(problems)
        )
    try:
        return mod.to_json(mod.run_all(results))
    except Exception:  # noqa: BLE001 — rule execution is a safety net, never blocks the run
        return []
