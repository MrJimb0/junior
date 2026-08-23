"""Allow-list forbidden-content scanner for NO_PHI exhaust records.

A deny-list (look for date, MRN, or SSN shapes) fundamentally cannot stop the largest PHI
classes -- a bare patient name ("Bassett, Kitty"), a non-standard id ("E1234567"), a
non-ISO date ("01/15/2026"), or arbitrary clinical prose carry none of the known-bad
tokens and would ship clean. So this scanner is **allow-list first**: an exhaust record
is built entirely from controlled values, so every string in it MUST be one of a small
set of safe forms -- a surrogate, a hash/fingerprint, the month stamp, a version, the
timestamped run id, or a short controlled token (recipe id, enum, code) with no date
shape and no long digit run. Anything else is rejected as possible raw PHI.

The shared deny-list (``check_scrub`` + ``scan_text_for_phi_content``) runs as a
secondary layer for richer labels, and a source-filename check catches a file name
that is otherwise token-shaped. A well-formed surrogate is the trusted id form.
``emitted_month`` (``YYYY-MM``) passes; any full date in any format does not.
"""
from __future__ import annotations

import re
from typing import Any

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
    check_scrub,
    scan_text_for_phi_content,
)
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import is_surrogate

# A controlled token: recipe id, enum value, step id, labeler id, site/study id, etc.
# Charset excludes whitespace, ':' (only surrogates/hashes use ':'), and quotes/commas,
# so names ("Bassett, Kitty"), prose, and raw chunk ids (colons) cannot match.
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-/]{1,80}$")
# Hash/fingerprint: sha256:<hex>, or a bare 16-hex config fingerprint, or a 64-hex
# digest. The bare branches REQUIRE at least one hex letter [a-f]: a pure-decimal 16-
# or 64-digit run (an MRN/account-shaped id) is NOT a hash, and the 5+-digit-run guard
# already rejects shorter numeric ids -- so without the letter requirement, exactly 16
# or 64 digits would be the one numeric length waved through as a "fingerprint".
_HASH_RE = re.compile(
    r"^(?:sha256:[0-9a-f]{3,64}"
    r"|(?=[0-9a-f]*[a-f])[0-9a-f]{16}"
    r"|(?=[0-9a-f]*[a-f])[0-9a-f]{64})$"
)
# A purely-alphabetic bare token. Every controlled value in an exhaust record is
# lowercase / snake / hex / dotted (enums, recipe & step & rule ids, archetypes,
# versions, the 'unknown' fingerprint placeholder) -- verified against the vocab enums
# and schemas -- so a purely-alphabetic token that carries an UPPERCASE letter is
# name-shaped ("BASSETT", "Bassett", "Smith") and is rejected everywhere except the
# operator-set site/study id fields (field-aware like run_id below).
_UPPERCASE_TOKEN_OK_FIELDS = frozenset({"site_id", "study_id"})
_PURELY_ALPHA_RE = re.compile(r"^[A-Za-z]+$")
_MONTH_RE = re.compile(r"^(?:\d{4}-\d{2}|unknown)$")
# A version must carry a 'v' prefix or a dotted form -- a bare integer is NOT a version
# (else a numeric MRN / compact date would be waved through as one).
_VERSION_RE = re.compile(r"^(?:v\d+(?:\.\d+)*|\d+\.\d+(?:\.\d+)*|[A-Za-z0-9_]+/v\d+)$")
# The timestamped run id (YYYYMMDD_<token>) -- digit-heavy but administrative, not PHI.
# Only ever trusted in the ``run_id`` field (field-aware): a compact datetime in any
# OTHER field could be a clinical timestamp, so it must NOT get this exemption.
_RUN_ID_RE = re.compile(r"^\d{8}_[A-Za-z0-9_]+$")
# Reject shapes that look like an identifier/date even inside a token.
_LONG_DIGIT_RUN_RE = re.compile(r"\d{5,}")
_DATE_SHAPE_RE = re.compile(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\b\d{8}\b")
# A source filename leaking into a value (records carry surrogates, not files). Broad.
_SOURCE_FILENAME_RE = re.compile(
    r"\.(?:csv|tsv|xlsx?|xlsm|jsonl?|parquet|txt|pdf|docx?|html?|xml|rtf|dat|dcm|out|log)\b",
    re.IGNORECASE,
)


def _is_safe_exhaust_value(
    value: str, *, allow_run_id_shape: bool = False, allow_uppercase_token: bool = False
) -> bool:
    """True iff ``value`` is one of the controlled forms an exhaust record may carry.
    The run-id timestamp shape is only allowed when ``allow_run_id_shape`` (the
    ``run_id`` field), so a compact clinical datetime elsewhere is not waved through.
    A name-shaped (purely-alphabetic, uppercase-bearing) bare token is allowed only
    when ``allow_uppercase_token`` (the operator-set site/study id fields)."""
    if value == "":
        return True
    if is_surrogate(value) or _HASH_RE.match(value) or _MONTH_RE.match(value):
        return True
    if _VERSION_RE.match(value):
        return True
    if allow_run_id_shape and _RUN_ID_RE.match(value):
        return True
    if not (
        _SAFE_TOKEN_RE.match(value)
        and not _DATE_SHAPE_RE.search(value)
        and not _LONG_DIGIT_RUN_RE.search(value)
    ):
        return False
    # Shape-clean controlled token. Reject a name-shaped one (purely alphabetic with
    # an uppercase letter) outside the operator-id fields -- every legitimate value is
    # lowercase/hex/snake/dotted, so this can only be raw PHI (a patient surname).
    if (
        not allow_uppercase_token
        and _PURELY_ALPHA_RE.match(value)
        and any(c.isupper() for c in value)
    ):
        return False
    return True


def _iter_strings(obj: Any, path: str = "$") -> list[tuple[str, str]]:
    """Yield ``(jsonpath, string)`` for every string value AND dict key."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, str):
        out.append((path, obj))
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(key, str):
                out.append((f"{path}.<key>", key))
            out.extend(_iter_strings(val, f"{path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, val in enumerate(obj):
            out.extend(_iter_strings(val, f"{path}[{i}]"))
    return out


def scan_record_for_forbidden_content(record: dict[str, Any]) -> list[str]:
    """Return violation labels (field-path + reason); empty means clean.

    Allow-list first: any string that is not a safe controlled form is rejected as
    possible raw PHI. Never includes the offending value in the message."""
    violations: list[str] = []

    # Secondary shared deny-list (paths + patient-id-like tokens) for richer labels.
    report = check_scrub("exhaust_record", record)
    if not report.ok:
        violations.extend(report.violations)

    for jsonpath, value in _iter_strings(record):
        leaf = jsonpath.rsplit(".", 1)[-1]
        is_run_id_field = leaf == "run_id"
        allow_upper = leaf in _UPPERCASE_TOKEN_OK_FIELDS
        if not _is_safe_exhaust_value(
            value, allow_run_id_shape=is_run_id_field, allow_uppercase_token=allow_upper
        ):
            violations.append(f"{jsonpath}: non-allowlisted value (possible raw PHI / id / text)")
            continue
        if _SOURCE_FILENAME_RE.search(value):
            violations.append(f"{jsonpath}: source filename")
        violations.extend(f"{jsonpath}: {label}" for label in scan_text_for_phi_content(value))
    return violations
