"""Whole-patient sanity checks that span more than one extracted variable.

These are clinical "does this make sense together?" checks that no single
variable can catch on its own — for example: death recorded before diagnosis, a
recurrence dated before the original diagnosis, treatment lines numbered out of
order, a metastatic treatment line in an early-stage patient, or a reported stage
that cites no source text it was read from.

Per-variable schema validation already catches structural errors (wrong type,
missing field). These rules instead catch combinations that are individually
well-formed but clinically contradictory. Each rule is a self-contained function
that reads the assembled extraction results
(``{variable_name: {"data": {...}}}``) and returns zero or more
``InvariantResult`` records. Severity is ``info`` | ``warning`` | ``error``;
none of them abort a run — they are a backstop / second opinion, surfaced for
review, and never on their own discard an extraction.

The runner loads this module by a HARDCODED filename
(``extract.py:_run_cross_variable_invariants``). If you rename this file,
update that path too.

VARIABLE NAMES LIVE IN ONE PLACE — the ``_VAR`` table below; rules never spell out
recipe or field names inline. If a ``_VAR`` entry points at a recipe or field that
does not exist, the rule just reads an empty value and quietly passes (it never
sees the data it was meant to check) — exactly the silent failure this table
guards against. A startup check (``invariant_target_specs`` + the runner's
``validate_invariant_targets``) fails loudly if any in-use entry names a recipe or
field that is missing, so the problem can't go unnoticed.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jr_pipeline.runtime_infrastructure.recipe_shared_rules import load_shared_validation_rule

# Partial-date parsing lives in one place. _date_key returns None for a non-date
# (the rules test `is None` to skip); date_sort_key is the sort-last variant.
_pd = load_shared_validation_rule("partial_date", __file__)
_date_key = _pd.date_key
_strictly_before = _pd.strictly_before
_month_ordinal = _pd.month_ordinal


# ---------------------------------------------------------------------------
# Variable name table — the single source of truth for which recipe/field a
# rule reads. Update HERE when a recipe is renamed; rules never hardcode names.
# ---------------------------------------------------------------------------

# logical role -> (variable_key, field within that variable's data payload).
_VAR: dict[str, tuple[str, str]] = {
    "original_dx":  ("date_of_diagnosis", "date_original_diagnosis"),
    "locoregional": ("date_of_diagnosis", "date_locoregional_recurrence_diagnosis"),
    "distant":      ("date_of_diagnosis", "date_metastatic_diagnosis"),
    "death":        ("date_of_death", "date_of_death"),
    "stage":        ("stage", "stage_at_diagnosis"),  # site-agnostic dx-anchored stage
    "lines":        ("treatment_lines", "lines"),
}

# Fallback field, checked only when the primary field above is empty. The
# date_of_diagnosis recipe exposes both ``date_original_diagnosis`` and an older
# ``date_of_diagnosis`` field name, so a chart that filled only the older name still
# satisfies the original-diagnosis role.
_ALIAS: dict[str, tuple[str, str]] = {
    "original_dx": ("date_of_diagnosis", "date_of_diagnosis"),
}

# Roles whose (recipe, field) target is intentionally NOT yet resolved, so
# validate_invariant_targets skips them. "lines" stays pending: no shipped recipe
# reconstructs treatment lines, and the line rules no-op until a site adds one.
_PENDING_5B: frozenset[str] = frozenset({"lines"})


def invariant_target_specs() -> list[tuple[str, str, str]]:
    """``(role, recipe_variable, output_field)`` for every invariant role a rule
    actively reads — excluding ``_PENDING_5B``. Also includes the ``_ALIAS`` fallback
    targets (older field names ``_field`` falls back to). Before running the rules the
    runner checks each of these points at a real recipe and a field its output schema
    actually declares; a target that points at nothing is a code bug — the silent
    "reads nothing and passes" failure this table exists to prevent — so it is
    reported loudly rather than ignored."""
    specs: list[tuple[str, str, str]] = []
    for role, (var, field_name) in _VAR.items():
        if role in _PENDING_5B:
            continue
        specs.append((role, var, field_name))
        if role in _ALIAS:
            avar, afield = _ALIAS[role]
            specs.append((role, avar, afield))
    return specs


@dataclass(frozen=True)
class InvariantResult:
    """One rule's verdict on a patient: pass/fail + severity + human message + context."""

    rule_id: str
    ok: bool
    severity: str                      # "info" | "warning" | "error"
    message: str
    context: dict[str, Any] = field(default_factory=dict)


def _get(results: dict[str, Any], var: str, *path: str, default: Any = None) -> Any:
    node = (results.get(var) or {}).get("data")
    for p in path:
        if not isinstance(node, dict):
            return default
        node = node.get(p)
    return node if node is not None else default


def _field(results: dict[str, Any], role: str, default: Any = None) -> Any:
    """Read a role's value by looking up its recipe/field in the ``_VAR`` table,
    falling back to the older field name in ``_ALIAS`` if the primary one is empty."""
    var, key = _VAR[role]
    val = _get(results, var, key)
    if val is None and role in _ALIAS:
        avar, akey = _ALIAS[role]
        val = _get(results, avar, akey)
    return default if val is None else val


def _stage_data(results: dict[str, Any]) -> dict[str, Any]:
    """The stage variable's full data payload (for provenance checks)."""
    var, _ = _VAR["stage"]
    return (results.get(var) or {}).get("data") or {}


def _stage_group(results: dict[str, Any]) -> str | None:
    """The overall AJCC stage group at diagnosis (AJCC = the standard cancer staging
    system), reduced to the bare stage FAMILY with the substage letter dropped
    ('IIA' -> 'II', 'IVB' -> 'IV'), or None. The early-stage check only cares
    whether the patient is stage 0/I/II/III vs IV, not about the substage."""
    sad = _stage_data(results).get("stage_at_diagnosis")
    group = sad.get("overall_group") if isinstance(sad, dict) else None
    if not isinstance(group, str):
        return None
    base = group.rstrip("ABC")
    return base or None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def rule_diagnosis_before_death(results: dict[str, Any]) -> list[InvariantResult]:
    """date_of_death must not precede the original diagnosis. Empty if either is missing."""
    dx = _date_key(_field(results, "original_dx"))
    dod = _date_key(_field(results, "death"))
    if dx is None or dod is None:
        return []
    if _strictly_before(dod, dx):
        return [InvariantResult(
            rule_id="diagnosis_before_death",
            ok=False, severity="error",
            message="Date of death precedes date of original diagnosis.",
            context={"dx": dx, "dod": dod},
        )]
    return [InvariantResult(rule_id="diagnosis_before_death", ok=True, severity="info", message="dx <= dod")]


def rule_original_before_locoregional(results: dict[str, Any]) -> list[InvariantResult]:
    """A locoregional recurrence must not predate the original diagnosis."""
    dx = _date_key(_field(results, "original_dx"))
    lrr = _date_key(_field(results, "locoregional"))
    if dx is None or lrr is None:
        return []
    if _strictly_before(lrr, dx):
        return [InvariantResult(
            rule_id="original_before_locoregional",
            ok=False, severity="error",
            message="Locoregional recurrence date precedes the original diagnosis.",
            context={"dx": dx, "locoregional": lrr},
        )]
    return [InvariantResult(rule_id="original_before_locoregional", ok=True, severity="info", message="dx <= locoregional")]


def rule_original_before_distant(results: dict[str, Any]) -> list[InvariantResult]:
    """A distant metastasis must not predate the original diagnosis. (De novo
    stage IV legitimately has distant == original, which is allowed; only a
    distant date strictly before the original is an error.)"""
    dx = _date_key(_field(results, "original_dx"))
    distant = _date_key(_field(results, "distant"))
    if dx is None or distant is None:
        return []
    if _strictly_before(distant, dx):
        return [InvariantResult(
            rule_id="original_before_distant",
            ok=False, severity="error",
            message="Distant metastasis date precedes the original diagnosis.",
            context={"dx": dx, "distant": distant},
        )]
    return [InvariantResult(rule_id="original_before_distant", ok=True, severity="info", message="dx <= distant")]


def rule_lrr_metastatic_mutual_exclusion(results: dict[str, Any]) -> list[InvariantResult]:
    """If BOTH a locoregional and a distant date are present and fall in the same
    or an adjacent calendar month, they may describe the SAME physical event (e.g.,
    supraclavicular nodal disease with concurrent distant imaging) and should be
    reconciled. Warning, not error — date_recurrence_reconciliation is the
    authoritative adjudicator; this just flags the case if that recipe was skipped
    or absent."""
    lrr = _date_key(_field(results, "locoregional"))
    distant = _date_key(_field(results, "distant"))
    if lrr is None or distant is None:
        return []
    a, b = _month_ordinal(lrr), _month_ordinal(distant)
    if a is None or b is None:
        return []   # month masked on one side -> insufficient precision to judge closeness
    if abs(a - b) <= 1:
        return [InvariantResult(
            rule_id="lrr_metastatic_mutual_exclusion",
            ok=False, severity="warning",
            message="Locoregional and distant recurrence dates are within one month; "
                    "they may be the same event and should be reconciled.",
            context={"locoregional": lrr, "distant": distant},
        )]
    return [InvariantResult(rule_id="lrr_metastatic_mutual_exclusion", ok=True, severity="info",
                            message="Locoregional and distant dates are separated in time.")]


def rule_therapy_line_chronological(results: dict[str, Any]) -> list[InvariantResult]:
    """Within each treatment strategy (adjuvant / metastatic / ...), line_number
    order must match start_date order. Reads the reconstructed treatment lines."""
    lines = _field(results, "lines", default=[])
    if not isinstance(lines, list) or not lines:
        return []
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for line in lines:
        if not isinstance(line, dict):
            continue
        by_strategy.setdefault(line.get("strategy") or "unknown", []).append(line)

    out: list[InvariantResult] = []
    for strat, items in by_strategy.items():
        items_sorted = sorted(items, key=lambda x: _date_key(x.get("start_date")) or (9999, 99, 99))
        nums = [int(i.get("line_number") or 0) for i in items_sorted]
        if nums != sorted(nums):
            out.append(InvariantResult(
                rule_id="therapy_line_chronological",
                ok=False, severity="error",
                message=f"Line numbers out of order within strategy={strat}.",
                context={"strategy": strat, "line_numbers": nums},
            ))
    if not out:
        out.append(InvariantResult(
            rule_id="therapy_line_chronological",
            ok=True, severity="info",
            message="Line numbers monotonic within each strategy.",
        ))
    return out


def rule_drug_date_plausibility(results: dict[str, Any]) -> list[InvariantResult]:
    """Treatment line start dates should fall between the original diagnosis and
    death. Pre-dx start is an error; post-dod start is a warning."""
    lines = _field(results, "lines", default=[])
    dx = _date_key(_field(results, "original_dx"))
    dod = _date_key(_field(results, "death"))
    if not isinstance(lines, list) or not lines:
        return []
    out: list[InvariantResult] = []
    for i, line in enumerate(lines):
        sd = _date_key(line.get("start_date")) if isinstance(line, dict) else None
        if sd is None:
            continue
        if dx is not None and _strictly_before(sd, dx):
            out.append(InvariantResult(
                rule_id="drug_date_plausibility",
                ok=False, severity="error",
                message=f"Treatment line {i} starts before diagnosis.",
                context={"line_index": i, "start_date": sd, "dx": dx},
            ))
        if dod is not None and _strictly_before(dod, sd):
            out.append(InvariantResult(
                rule_id="drug_date_plausibility",
                ok=False, severity="warning",
                message=f"Treatment line {i} starts after recorded date of death.",
                context={"line_index": i, "start_date": sd, "dod": dod},
            ))
    if not out:
        out.append(InvariantResult(
            rule_id="drug_date_plausibility",
            ok=True, severity="info",
            message="All treatment line start dates fall between dx and dod (or anchors missing).",
        ))
    return out


def rule_line_strategy_consistency(results: dict[str, Any]) -> list[InvariantResult]:
    """An early-stage patient (0/I/II/III) shouldn't have a metastatic treatment
    line before any curative-intent line."""
    lines = _field(results, "lines", default=[])
    stage = _stage_group(results)
    if not isinstance(lines, list) or not lines:
        return []
    out: list[InvariantResult] = []

    adjuvant_seen = False
    for line in lines:
        if not isinstance(line, dict):
            continue
        strat = line.get("strategy")
        if strat in {"adjuvant", "neoadjuvant"}:
            adjuvant_seen = True
        elif strat == "metastatic" and not adjuvant_seen and stage in {"0", "I", "II", "III"}:
            out.append(InvariantResult(
                rule_id="line_strategy_consistency",
                ok=False, severity="warning",
                message="Metastatic line appears before any curative-intent line for an early-stage patient.",
                context={"stage": stage, "line_strategy": strat},
            ))
    if not out:
        out.append(InvariantResult(
            rule_id="line_strategy_consistency",
            ok=True, severity="info",
            message="Line strategy order consistent with clinical stage.",
        ))
    return out


def rule_stage_provenance_consistent(results: dict[str, Any]) -> list[InvariantResult]:
    """If a stage at diagnosis is reported, it must cite the source text span (the
    "chunk") it was read from — i.e. show its provenance / where the value came from.
    The pipeline's general provenance validator already enforces this for every
    field; this rule is the stage-specific clinical backstop."""
    sad = _stage_data(results).get("stage_at_diagnosis")
    group = sad.get("overall_group") if isinstance(sad, dict) else None
    if not group:
        return [InvariantResult(
            rule_id="stage_provenance_consistent",
            ok=True, severity="info",
            message="No stage at diagnosis reported; no provenance required.",
        )]
    if not sad.get("evidence_chunk_id"):
        return [InvariantResult(
            rule_id="stage_provenance_consistent",
            ok=False, severity="warning",
            message="Stage at diagnosis is populated but cites no evidence_chunk_id.",
        )]
    return [InvariantResult(
        rule_id="stage_provenance_consistent",
        ok=True, severity="info",
        message="Stage at diagnosis cites a chunk.",
    )]


RULES: list[Callable[[dict[str, Any]], list[InvariantResult]]] = [
    rule_diagnosis_before_death,
    rule_original_before_locoregional,
    rule_original_before_distant,
    rule_lrr_metastatic_mutual_exclusion,
    rule_therapy_line_chronological,
    rule_drug_date_plausibility,
    rule_line_strategy_consistency,
    rule_stage_provenance_consistent,
]


def run_all(results: dict[str, Any]) -> list[InvariantResult]:
    """Run every registered invariant; a rule crash is recorded as an error, not raised."""
    out: list[InvariantResult] = []
    for rule in RULES:
        try:
            out.extend(rule(results))
        except Exception as e:
            out.append(InvariantResult(
                rule_id=rule.__name__,
                ok=False, severity="error",
                message=f"invariant crashed: {type(e).__name__}: {e}",
            ))
    return out


def to_json(results: list[InvariantResult]) -> list[dict[str, Any]]:
    """Convert a list of :class:`InvariantResult` to JSON-serializable dicts."""
    return [
        {
            "rule_id": r.rule_id,
            "ok": r.ok,
            "severity": r.severity,
            "message": r.message,
            "context": r.context,
        }
        for r in results
    ]
