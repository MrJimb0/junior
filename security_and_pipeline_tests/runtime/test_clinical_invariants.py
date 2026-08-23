"""The cross-variable clinical invariants (the only model-independent labeler)
are wired into the post-extract path, and their outcomes reach the exhaust as a
text-free, date-free weak label."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jr_pipeline.pipeline_steps.step_8_organize_output.cross_variable_invariants import (
    InvariantDefinitionError,
    run_cross_variable_invariants,
    validate_invariant_targets,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    record_invariant_outcomes,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "var_extraction_recipes"


def _load_real_invariant_module():
    mod_path = RECIPES / "_shared_validation_rules" / "clinical_invariants.py"
    spec = importlib.util.spec_from_file_location("jr_clinical_invariants_test", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["jr_clinical_invariants_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_death_before_diagnosis_trips_the_invariant():
    results = {
        "date_of_diagnosis": {"data": {"date_of_diagnosis": "2024-02-15"}},
        "date_of_death": {"data": {"date_of_death": "2020-01-01"}},  # impossible: before dx
    }
    outcomes = run_cross_variable_invariants(RECIPES, results)
    assert any(
        (not o["ok"]) and o["rule_id"] == "diagnosis_before_death" for o in outcomes
    ), "death-before-diagnosis was not flagged"


def test_consistent_dates_have_no_violation():
    results = {
        "date_of_diagnosis": {"data": {"date_of_diagnosis": "2020-01-01"}},
        "date_of_death": {"data": {"date_of_death": "2024-02-15"}},
    }
    outcomes = run_cross_variable_invariants(RECIPES, results)
    assert not [o for o in outcomes if not o["ok"]]


def test_real_invariant_targets_all_resolve():
    """Every active invariant role's (recipe, field) resolves against the real recipe
    tree. (The two tests above already exercise the runner's startup assertion over
    RECIPES; this asserts it directly.)"""
    mod = _load_real_invariant_module()
    assert validate_invariant_targets(mod, RECIPES) == []


def test_stage_role_now_resolved_in_5b():
    """The stage role is validated like every other role and resolves to the real
    ``stage`` recipe's ``stage_at_diagnosis`` object."""
    mod = _load_real_invariant_module()
    specs = {role: (var, fld) for role, var, fld in mod.invariant_target_specs()}
    assert specs.get("stage") == ("stage", "stage_at_diagnosis")
    assert mod._PENDING_5B == frozenset({"lines"})  # no shipped recipe reconstructs treatment lines


def test_validate_invariant_targets_flags_a_dangling_definition():
    """Non-tautology guard: a bogus (recipe, field) triple is reported as a problem."""
    class _FakeMod:
        @staticmethod
        def invariant_target_specs():
            return [("bogus", "no_such_recipe", "no_such_field")]

    problems = validate_invariant_targets(_FakeMod(), RECIPES)
    assert problems and "no_such_recipe" in problems[0]


def test_runner_raises_loud_on_dangling_definition(tmp_path):
    """A dangling invariant definition fails loud, rather than being swallowed like a
    rule-execution crash."""
    shared = tmp_path / "_shared_validation_rules"
    shared.mkdir(parents=True)
    (shared / "clinical_invariants.py").write_text(
        "def invariant_target_specs():\n"
        "    return [('bogus', 'no_such_recipe', 'no_such_field')]\n"
        "def run_all(results):\n    return []\n"
        "def to_json(rs):\n    return []\n",
        encoding="utf-8",
    )
    with pytest.raises(InvariantDefinitionError):
        run_cross_variable_invariants(tmp_path, {})


def test_invariant_exhaust_is_text_free_and_date_free(tmp_path):
    outcomes = [{
        "rule_id": "diagnosis_before_death", "ok": False, "severity": "error",
        "message": "Date of death precedes date of diagnosis.",
        "context": {"dx": [2024, 2, 15], "dod": [2020, 1, 1]},  # dates — must NOT leak
    }]
    out = record_invariant_outcomes(
        run_id="20260101_000000_aa", patient_id="Test_Patient",
        outcomes=outcomes, data_root=tmp_path,
    )
    raw = out.read_text(encoding="utf-8")
    assert "Test_Patient" not in raw            # patient surrogate, not the id
    assert "2024" not in raw and "2020" not in raw  # the context dates never travel
    data = json.loads(raw)
    assert data["record_type"] == "clinical_invariant"
    assert data["labeler_id"] == "clinical_invariants/cancer_staging"
    assert data["n_violations"] == 1
    o0 = data["outcomes"][0]
    assert o0["rule_id"] == "diagnosis_before_death" and o0["ok"] is False
    assert "context" not in o0 and "message" not in o0  # only rule_id/ok/severity travel
