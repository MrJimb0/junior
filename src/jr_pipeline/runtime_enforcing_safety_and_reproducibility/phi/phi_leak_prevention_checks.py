"""Scrubbing invariant for the code stream (ADRs 0019 + 0026).

``code/config_resolved.yaml`` and ``code/entry_point.json`` MUST be stripped
of patient-specific selectors, local-filesystem paths referring to PHI, and
ad-hoc operator notes before the code bundle is sealed. This module
enforces that invariant mechanically. Failure to scrub is a hard error.

``check_scrub_file`` additionally runs the regex set against the
*serialized* YAML/JSON bytes written to disk so encoding differences
between the in-memory dict and the text representation cannot smuggle
values past the scanner.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Deliberately broad: false positives here are cheap (replace with a placeholder); misses are not.
_PHI_PATH_PATTERNS = [
    re.compile(r"raw/patients?/", re.IGNORECASE),
    re.compile(r"box[_\- ]?medicine", re.IGNORECASE),
    re.compile(r"/projects/[^/\s]+/[^/\s]+/raw/", re.IGNORECASE),
    re.compile(r"/share/pi/", re.IGNORECASE),
    re.compile(r"/oak/stanford", re.IGNORECASE),
    re.compile(r"/scratch/[^/]+/raw/", re.IGNORECASE),
    re.compile(r"/data/clinical/", re.IGNORECASE),
    re.compile(r"/ehr/", re.IGNORECASE),
    re.compile(r"/mrn(?:s)?/", re.IGNORECASE),
    re.compile(r"clarity[_\-/]", re.IGNORECASE),
    re.compile(r"ssn[:=]", re.IGNORECASE),
]
_PATIENT_ID_PATTERNS = [
    # literal identifier patterns that look like real patient IDs (gs123, mrn456, etc.)
    re.compile(r"\b(gs\d+|id\d+|mrn\d+|pt[_-]?\d+)\b", re.IGNORECASE),
    # quoted string value assigned to a patient-id-like key in config/YAML/JSON.
    # quote is REQUIRED — without it this matches Python keyword args like
    # patient_id=patient_id, which are code, not data.
    re.compile(r"\b(mrn|patient[_-]?id)\b\s*[:=]\s*[\"']\w{4,}", re.IGNORECASE),
]
_SYMBOLIC_PLACEHOLDERS = {
    "$PATIENT_LIST",
    "${PATIENT_LIST}",
    "$PATIENT_ID",
    "${PATIENT_ID}",
    "$RAW_PATIENTS",
    "${RAW_PATIENTS}",
}

# Content patterns: PHI that must never appear in NO_PHI exports/exhaust or logs.
# Distinct from _PHI_PATH_PATTERNS above (which target code-bundle config PATHS).
# The whole codebase scans PHI *content* against this ONE vocabulary. An absolute
# date (YYYY-MM-DD) is forbidden here; the administrative month stamp (YYYY-MM) is
# deliberately not matched.
PHI_CONTENT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "absolute date (HIPAA identifier)"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN-shaped"),
    (re.compile(r"\bMRN\s*[:=]?\s*\d+", re.IGNORECASE), "MRN reference"),
    (re.compile(r"\bDOB\s*[:=]?\s*\d", re.IGNORECASE), "DOB reference"),
    (re.compile(r"date\s+of\s+birth\s*:", re.IGNORECASE), "date-of-birth field"),
    (re.compile(r"patient_name\s*[:=]", re.IGNORECASE), "patient name field"),
    (re.compile(r"chief\s+complaint", re.IGNORECASE), "clinical note text"),
    (re.compile(r"history\s+of\s+present\s+illness", re.IGNORECASE), "clinical note text"),
    (re.compile(r"assessment\s+and\s+plan", re.IGNORECASE), "clinical note text"),
    (re.compile(r"patient\s+present", re.IGNORECASE), "clinical note text"),
    (re.compile(r"physical\s+exam", re.IGNORECASE), "clinical note text"),
]


def scan_text_for_phi_content(text: str) -> list[str]:
    """Return the label of every PHI content pattern found in ``text`` (empty = clean)."""
    return [label for pattern, label in PHI_CONTENT_PATTERNS if pattern.search(text)]


@dataclass
class ScrubReport:
    """Result of scanning a scrub target."""

    target_name: str
    ok: bool
    violations: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> None:
        if not self.ok:
            joined = "\n  - ".join(self.violations)
            raise ScrubViolation(
                f"Scrub failed for {self.target_name}:\n  - {joined}"
            )


class ScrubViolation(RuntimeError):
    """Raised when a code-bundle artifact fails the scrubbing invariant."""


def _walk_strings(obj: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    """Yield ``(jsonpath, string-value)`` for every string, including dict keys."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                yield f"{path}.<key>", k
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")


def _check_string(value: str, path: str) -> list[str]:
    findings: list[str] = []
    if value in _SYMBOLIC_PLACEHOLDERS:
        return findings
    for pat in _PHI_PATH_PATTERNS:
        if pat.search(value):
            findings.append(
                f"{path}: path pattern {pat.pattern!r} found in {value!r} — "
                "replace with a symbolic placeholder like $RAW_PATIENTS"
            )

    for pat in _PATIENT_ID_PATTERNS:
        if pat.search(value):
            findings.append(
                f"{path}: patient-id-like token matched by {pat.pattern!r} in {value!r} — "
                "move enumerated IDs out of config and pass via --patient-list"
            )

    return findings


def check_scrub(target_name: str, obj: Any) -> ScrubReport:
    """Check a config/entry-point object for PHI leakage."""
    violations: list[str] = []
    for jsonpath, s in _walk_strings(obj):
        violations.extend(_check_string(s, jsonpath))
    return ScrubReport(target_name=target_name, ok=len(violations) == 0, violations=violations)


def check_scrub_file(target_name: str, path: Path | str) -> ScrubReport:
    """Run the regex set against the raw serialized bytes of a file."""
    p = Path(path)
    if not p.is_file():
        return ScrubReport(target_name=target_name, ok=True)
    text = p.read_text(encoding="utf-8", errors="replace")
    violations: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if any(ph in line for ph in _SYMBOLIC_PLACEHOLDERS):
            continue
        for pat in _PHI_PATH_PATTERNS:
            if pat.search(line):
                violations.append(
                    f"{target_name}:L{line_no}: path pattern {pat.pattern!r} — "
                    f"serialized line contains: {line.strip()!r}"
                )
                break
        for pat in _PATIENT_ID_PATTERNS:
            if pat.search(line):
                violations.append(
                    f"{target_name}:L{line_no}: patient-id pattern {pat.pattern!r} — "
                    f"serialized line: {line.strip()!r}"
                )
                break
    return ScrubReport(target_name=target_name, ok=len(violations) == 0, violations=violations)


__all__ = [
    "ScrubReport",
    "ScrubViolation",
    "check_scrub",
    "check_scrub_file",
    "PHI_CONTENT_PATTERNS",
    "scan_text_for_phi_content",
]
