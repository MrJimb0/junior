"""Domain-separated HMAC surrogates for NO_PHI exhaust identifiers.

A raw chunk/patient/source id cannot leave the institution, but the *linkage* it
encodes (these rows belong to one patient; this chunk recurs across runs) is the
whole training signal. A surrogate is an irreversible, keyed stand-in: same raw id +
same secret -> same surrogate, so rows still join, but the raw id cannot be recovered
or brute-forced (the secret lives PHI-side, `secret_lifecycle`).

The HMAC message is **domain-separated** by
``surrogate_version | linkage_scope | record_type | output_field | raw_identifier``,
so the same raw value yields *different* surrogates in different fields, record types,
and scopes. That is deliberate: linkage is scoped, never a free
cross-record-type join of a patient. Output format is fixed and schema-declared:
``surr:v1:<scope>:<field>:<hex>``.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from enum import StrEnum

# Bump on any change to the message construction or output format; a new version is a
# new linkage generation (old + new surrogates never collide / never silently link).
SURROGATE_VERSION = "v1"

# 96 bits of HMAC-SHA256 — collision-safe across a multi-site study, compact on disk.
_DIGEST_HEX_LEN = 24

# Component charset for the domain-separation fields. Excludes the '|' delimiter and ':'
# so the HMAC message is unambiguous and the minted surrogate is always recognizable.
_COMPONENT_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# The closed set the egress/forbidden-content scanner trusts as a surrogate. The hex
# tail is pinned to the EXACT mint length so a hex-encoded raw id (a longer/shorter blob)
# in an id slot is not waved through as a "surrogate" and skips the PHI scan.
_SURROGATE_RE = re.compile(
    rf"^surr:v1:(run|study|site):[a-z][a-z0-9_]*:[0-9a-f]{{{_DIGEST_HEX_LEN}}}$"
)


class LinkageScope(StrEnum):
    """How far a surrogate links. The secret that mints it is scoped the same way:
    a RUN surrogate only joins rows within one run; STUDY/SITE join across runs."""

    RUN = "run"
    STUDY = "study"
    SITE = "site"


# A fixed, record-type-independent linkage domain for EVIDENCE-CHUNK surrogates.
# Normally the ``record_type`` component scopes a surrogate to one record type, so the
# same raw value yields different stand-ins across record types and cannot be joined
# between them -- the right default for PATIENT identity. But the retriever-vs-cross-
# encoder flow needs the opposite for a CHUNK: a reviewer's ``relevance_label`` (and a
# ``retriever_miss``) must agree with the ``selection_judgment`` row on a chunk's stand-in,
# so the human label joins back to that chunk's retriever/cross-encoder ranks. Minting
# chunk surrogates under this one shared domain (with a shared ``output_field``, which is
# also part of the message) makes that join work -- and ONLY for chunk identity. Patient
# and reviewer ids stay per-record-type, so no cross-type patient/reviewer join is created.
EVIDENCE_CHUNK_LINKAGE_DOMAIN = "evidence_chunk"


def make_surrogate(
    secret: bytes,
    *,
    scope: str,
    record_type: str,
    output_field: str,
    raw: str,
) -> str:
    """Mint the domain-separated surrogate for ``raw`` under (scope, record_type, field).

    ``secret`` is the scope's HMAC key (from ``secret_lifecycle``). The result is
    deterministic for a fixed (secret, scope, record_type, output_field, raw) and
    irreversible without the secret.
    """
    if not isinstance(secret, (bytes, bytearray)) or not secret:
        raise ValueError("surrogate secret must be non-empty bytes")
    # The domain-separation components must be unambiguous (no '|' delimiter collision)
    # and must match the recognizer's charset (else a minted surrogate would be rejected
    # by is_surrogate and re-scanned as raw). Only ``raw`` is free.
    if scope not in {s.value for s in LinkageScope}:
        raise ValueError(f"unknown surrogate scope: {scope!r}")
    if not _COMPONENT_RE.match(record_type) or not _COMPONENT_RE.match(output_field):
        raise ValueError(
            f"record_type/output_field must match {_COMPONENT_RE.pattern} "
            f"(got record_type={record_type!r}, output_field={output_field!r})"
        )
    message = "|".join(
        [SURROGATE_VERSION, scope, record_type, output_field, str(raw)]
    ).encode("utf-8")
    digest = hmac.new(bytes(secret), message, hashlib.sha256).hexdigest()[:_DIGEST_HEX_LEN]
    return f"surr:{SURROGATE_VERSION}:{scope}:{output_field}:{digest}"


def is_surrogate(value: object) -> bool:
    """True iff ``value`` is a well-formed surrogate string (the form the NO_PHI
    scanner trusts in an id position)."""
    return isinstance(value, str) and bool(_SURROGATE_RE.match(value))
