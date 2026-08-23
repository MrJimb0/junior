"""ValidationResult — invariant check outcomes for one extraction."""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.base import ArtifactSchema


class RuleResult(ArtifactSchema):
    rule_name: str
    passed: bool
    # A rule that could not be loaded or blew up never looked at the value at all, which
    # is not the same as a rule that ran and found a problem. Both carry passed=False,
    # so this is what tells the two apart.
    did_not_run: bool = False
    message: str = ""


class ValidationResult(ArtifactSchema):
    patient_id: str
    variable: str
    run_id: str
    # All four count RULES, not entries in ``results``: one rule that reports three
    # problems contributes three RuleResults but one to rules_failed. rules_checked is
    # every rule the recipe declared and equals rules_passed + rules_failed +
    # rules_did_not_run; rules_failed counts only rules that RAN.
    rules_checked: int = 0
    rules_passed: int = 0
    rules_failed: int = 0
    rules_did_not_run: int = 0
    results: list[RuleResult] = []
