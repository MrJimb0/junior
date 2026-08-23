"""load and run the validation rules a recipe asks for.

A validation rule is a sanity check on one extracted variable (for example,
"a stage value must be one of the allowed AJCC groups"). Rules live in
var_extraction_recipes/_shared_validation_rules/ as plain
python files. Each file exports a check(results: dict) -> list[dict] function
returning a list of problems (empty list = clean). The recipe names which
rules to run; we load each one on demand so adding a rule is just dropping a
file into that folder.

Rules that fail to load or raise are reported as rule_results rather than crashing the
run — one broken rule should not take down extraction. Such a rule is marked
``did_not_run`` and counted separately, because it checked nothing: without that mark, a
project whose recipes_root has no _shared_validation_rules/ folder would record every
declared rule for every patient exactly like a rule that ran and found a problem, while
still reporting every variable clean.

No recipe shipped in var_extraction_recipes/ declares any validation.rules, so this
runs over an empty list today; it is here for the recipes that will declare them.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.validation_result import (
    RuleResult,
    ValidationResult,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger

_log = get_logger("validation_rules")


def run_recipe_validation_rules(
    *,
    recipes_root: Path,
    rule_names: list[str],
    extraction_results: dict[str, Any],
    patient_id: str,
    variable: str,
    run_id: str,
) -> ValidationResult:
    """run the validation rules listed in recipe.validation.rules.

    ``recipes_root`` is the real recipe-tree root the caller already holds. It
    must NOT be re-derived from a single recipe's yaml path: recipes nest at
    varying depths (``basic/<var>/v1`` vs ``oncology/<sub>/<var>/v1``), so a fixed
    ``.parent`` count lands inside a collection and never finds
    ``_shared_validation_rules``, making every rule report "not found"."""
    rules_dir = Path(recipes_root) / "_shared_validation_rules"

    rule_results: list[RuleResult] = []
    # Tallied per RULE, not per entry in rule_results: one rule that reports three
    # problems appends three of those, so counting the entries would tell the operator a
    # recipe declared more rules than it did — and the counts are what reaches the screen
    # and result.json.
    rules_that_passed: list[str] = []
    rules_that_failed: list[str] = []
    rules_that_did_not_run: list[str] = []

    for rule_name in rule_names:
        rule_file = rules_dir / f"{rule_name}.py"
        if not rule_file.is_file():
            rules_that_did_not_run.append(rule_name)
            rule_results.append(RuleResult(
                rule_name=rule_name,
                passed=False,
                did_not_run=True,
                message=(
                    f"Did not run — no rule file at {rule_file}, so this value was not "
                    f"checked. Put {rule_name}.py in that folder, or drop {rule_name} "
                    "from the recipe's validation.rules."
                ),
            ))
            continue

        try:
            mod = _load_rule_module(rule_file)
            check_fn = getattr(mod, "check", None)
            if check_fn is None:
                rules_that_did_not_run.append(rule_name)
                rule_results.append(RuleResult(
                    rule_name=rule_name,
                    passed=False,
                    did_not_run=True,
                    message=(
                        f"Did not run — {rule_file} defines no check(results) function, "
                        "so this value was not checked."
                    ),
                ))
                continue

            # Materialized before anything is decided from it. A rule that yields its
            # problems rather than returning a list hands back a generator, which is
            # always truthy — so a clean rule was recorded as failed with no message,
            # and a rule that raised partway through yielding landed in both the failed
            # and the did-not-run tallies, breaking the count this file promises.
            issues = list(check_fn(extraction_results) or [])
            if issues:
                rules_that_failed.append(rule_name)
                for issue in issues:
                    rule_results.append(RuleResult(
                        rule_name=rule_name,
                        passed=False,
                        message=str(issue),
                    ))
            else:
                rules_that_passed.append(rule_name)
                rule_results.append(RuleResult(
                    rule_name=rule_name,
                    passed=True,
                ))
        except Exception as e:
            rules_that_did_not_run.append(rule_name)
            # The exception's type separates a rule file that would not import from a
            # rule that blew up on this patient's answer, and a type name carries no
            # patient data, so it belongs here rather than in the message below.
            _log.warning("validation_rule_raised", extra_={
                "variable": variable,
                "rule": rule_name,
                "error_type": type(e).__name__,
            })
            rule_results.append(RuleResult(
                rule_name=rule_name,
                passed=False,
                did_not_run=True,
                message=(
                    f"Did not run — the rule {rule_name} stopped with an error while "
                    "loading or checking, so this value was not checked. Repair the rule "
                    f"in {rule_file}, or drop {rule_name} from the recipe's "
                    f"validation.rules, then re-run. The rule reported: {e}"
                ),
            ))

    if rules_that_did_not_run:
        # A rule that never ran leaves the value unchecked, and invariants.json is not
        # something anyone opens unprompted. The rule names and the folder are safe to
        # log; the rule's own message can quote patient data, so it stays out of here.
        _log.warning("validation_rules_did_not_run", extra_={
            "variable": variable,
            "rules": rules_that_did_not_run,
            "hint": f"{variable}: {len(rules_that_did_not_run)} of {len(rule_names)} "
                    "validation rule(s) never ran, so this value was NOT checked — which "
                    f"is not the same as passing. The rule files belong in {rules_dir}.",
        })

    return ValidationResult(
        patient_id=patient_id,
        variable=variable,
        run_id=run_id,
        rules_checked=len(rule_names),
        rules_passed=len(rules_that_passed),
        rules_failed=len(rules_that_failed),
        rules_did_not_run=len(rules_that_did_not_run),
        results=rule_results,
    )


def _load_rule_module(path: Path):
    """import a rule file by its path, then remember it under a unique name so
    the same file is loaded from disk only once per process."""
    mod_name = f"jr_validation_rule__{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rule: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module
