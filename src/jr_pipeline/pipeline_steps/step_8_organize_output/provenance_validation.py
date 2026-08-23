"""Step 8: check that every extracted clinical value can be traced back to its source.

"Provenance" here means: which span of source text backed this value. A *value-bearing*
field is a non-null scalar under ``data_final`` that holds an extracted clinical value
-- as opposed to a STRUCTURAL field (confidence, rationale, counts, status, or the
source pointer itself) or a null/"unknown" placeholder. The rule, enforced on the
patient-identifiable (PHI) side: every object that holds an extracted value must carry
a non-empty pointer back to its source text span (``*evidence_chunk_id`` /
``evidence_chunk_ids``) at its OWN level, and a list of extracted objects must carry
the pointer on each item. A value with no chunk to trace it back to is an
unsupported claim.

``find_unprovenanced_value_paths`` returns the dotted paths of objects that hold an
extracted value but no source pointer; an empty list means every value is traceable
(what the toy / end-to-end result.json must satisfy).
"""
from __future__ import annotations

from typing import Any

# Structural keys never count as value-bearing -- they don't need their own evidence.
# ``setting`` is a fixed recipe-scope label (e.g. "original"/"metastatic"/
# "locoregional_recurrence") naming which episode the recipe covers, not an extracted
# finding, so like the n_/derived_ aggregates it is not a claim and carries no evidence.
_STRUCTURAL_KEYS = frozenset({
    "ok", "warnings", "errors", "error", "confidence", "rationale",
    "notes", "note", "reason", "method", "status", "setting",
})
# ``_status`` is deliberately NOT a structural suffix. A field like ``vital_status``
# ("dead"/"alive") or ``er_status`` ("positive"/"negative") is a real extracted clinical
# claim and must carry its own evidence pointer. (The bare ``status`` key above stays
# structural -- that names an ok/state marker, not an extracted finding.)
_STRUCTURAL_SUFFIXES = ("_certainty", "_confidence", "_method", "_evidence", "_count", "_rationale")
# Derived aggregates over an already-grounded sibling collection are not independent
# claims -- they restate what the grounded items already say, so they carry no evidence
# of their own. ``n_`` is the derived-COUNT prefix (n_primaries, n_lines); ``derived_``
# is the general opt-in marker a recipe puts on a derived flag/value (e.g.
# derived_multiple_primaries restates len(primaries) > 1, grounded per item in primaries[]).
_STRUCTURAL_PREFIXES = ("n_", "derived_")

# Sentinel non-answers: the model could not determine a value. These are placeholders,
# not extracted claims -- there is nothing to ground a non-determination on -- so they
# never count as value-bearing (this is the "null/unknown placeholder" the module
# docstring promises). Only the literal non-answer is exempt; a real categorical value
# like "dead"/"alive"/"positive" is still a claim that must carry its own evidence.
_NON_ANSWER_VALUES = frozenset({"unknown"})


def _is_evidence_pointer(key: str) -> bool:
    """A field that points back at the source text span(s): dob_evidence_chunk_id
    (a single id) or evidence_chunk_ids (a per-item list of ids)."""
    return "evidence_chunk_id" in key.lower()


def _is_structural(key: str) -> bool:
    k = key.lower()
    return (
        k in _STRUCTURAL_KEYS
        or k.endswith(_STRUCTURAL_SUFFIXES)
        or k.startswith(_STRUCTURAL_PREFIXES)
    )


def _is_nonempty(value: Any) -> bool:
    return value not in (None, "", [], {})


def _is_value_bearing_leaf(key: str, value: Any) -> bool:
    """A non-null scalar holding extracted clinical content. Nested dicts/lists are
    walked separately (their own objects get checked), so they are not leaves here.
    A null or "unknown" non-answer is a placeholder, not a claim."""
    if value is None or isinstance(value, (dict, list)):
        return False
    if isinstance(value, str) and value.strip().lower() in _NON_ANSWER_VALUES:
        return False
    return not _is_evidence_pointer(key) and not _is_structural(key)


def find_unprovenanced_value_paths(data: Any, path: str = "data") -> list[str]:
    """Walk ``data`` recursively; return the dotted paths of objects that hold an
    extracted value but carry no non-empty source pointer at their level.
    Empty => every extracted value is traceable to a source chunk id."""
    missing: list[str] = []
    _walk(data, path, missing)
    return missing


def _walk(node: Any, path: str, missing: list[str]) -> None:
    if isinstance(node, dict):
        has_value = any(_is_value_bearing_leaf(k, v) for k, v in node.items())
        has_evidence = any(_is_evidence_pointer(k) and _is_nonempty(v) for k, v in node.items())
        if has_value and not has_evidence:
            missing.append(path)
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                _walk(v, f"{path}.{k}", missing)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", missing)
