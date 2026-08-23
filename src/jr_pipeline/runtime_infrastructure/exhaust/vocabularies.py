"""Versioned controlled vocabularies for the NO_PHI exhaust layer.

Every value that leaves the institution in an exhaust record must come from a
*closed* set — never free EHR text. A controlled vocabulary does two jobs:

1. **PHI safety** (``doc_type`` especially): a raw EHR ``doc_type`` is free text and
   can carry a patient name; ``canonicalize_doc_type`` maps any raw value to a fixed
   member or ``"unknown"`` and *can never return the raw input*, so only this closed
   set reaches NO_PHI. Unmapped non-empty values are counted, never silently dropped.
2. **Poolability**: rows from different sites/runs only aggregate if their categorical
   fields share one enum. The ``StrEnum`` members ARE the wire values (a ``StrEnum``
   member serialises as its string), so the schema layer can type a field as one of
   these and pydantic rejects any out-of-vocab value loudly.

``VOCAB_VERSION`` stamps the whole set onto the run header. Bump it when a member is
added/removed/re-meant so pooled rows from different vocab generations stay
distinguishable.
"""
from __future__ import annotations

from enum import StrEnum

# The version of the *whole* vocabulary set, stamped onto RunHeader.vocab_version.
# Additive members -> minor; removed/re-meant members -> major.
VOCAB_VERSION = "exhaust/v1"


class DocType(StrEnum):
    """Document-kind of an evidence chunk (gated from raw EHR ``doc_type`` text)."""

    PATHOLOGY = "pathology"
    RADIOLOGY = "radiology"
    PROGRESS_NOTE = "progress_note"
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    OPERATIVE_NOTE = "operative_note"
    CONSULT = "consult"
    LABORATORY = "laboratory"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    DEMOGRAPHICS = "demographics"
    ENCOUNTER = "encounter"
    PROCEDURE = "procedure"
    ORDER = "order"
    FLOWSHEET = "flowsheet"
    IMMUNIZATION = "immunization"
    UNKNOWN = "unknown"


class Retriever(StrEnum):
    """Which retriever surfaced a candidate (matches ``RetrieverInfo.kind``)."""

    BM25 = "bm25"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"
    EXACT = "exact"
    DIRECT_PARQUET = "direct_parquet"
    UNKNOWN = "unknown"


class CorrectionType(StrEnum):
    """Kind of expert-feedback record emitted by the app/CLI."""

    EXTRACTION_CORRECTION = "extraction_correction"
    EXTRACTION_CONFIRMATION = "extraction_confirmation"
    CHUNK_RELEVANCE = "chunk_relevance"


class RelevanceLabel(StrEnum):
    """A reviewer's relevance verdict on one candidate chunk for one question.

    The human label the workbench captures (relevant / not relevant / unsure), carried
    by a ``relevance_label`` exhaust record as the cross-encoder fine-tuning target the
    chunk's retriever/cross-encoder ranks are paired against."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    UNSURE = "unsure"


class SignalType(StrEnum):
    """Kind of disagreement/weak-label signal (the disagreement_signal axis)."""

    PARSE_FAILURE = "parse_failure"
    VALUE_DISAGREEMENT = "value_disagreement"
    INVARIANT_VIOLATION = "invariant_violation"


class ValueShape(StrEnum):
    """Shape of an extracted value (lets the flywheel reason about output form
    without seeing the value itself)."""

    SCALAR = "scalar"
    LIST = "list"
    OBJECT = "object"
    NULL = "null"


class HumanTier(StrEnum):
    """Reviewer expertise tier (matches the Shiny reviewer-tier resolution)."""

    EXPERT = "expert"
    TRAINEE = "trainee"
    UNKNOWN = "unknown"


class ErrorKind(StrEnum):
    """Bucketed extraction failure reason (extraction_outcome.error_kind).

    Maps the raw error/exception strings -- which can echo prompt or clinical text --
    onto a closed set, so the flywheel sees *why* an extraction failed without the
    free text. The retry/quarantine members mirror the controlled ``kind`` the retry
    layer already records on the PHI side; ``parse_failure`` defers detail to
    ``ParseFailureKind``."""

    NONE = "none"
    PARSE_FAILURE = "parse_failure"
    EMPTY_EVIDENCE = "empty_evidence"
    VALIDATION_FAILURE = "validation_failure"
    DETERMINISTIC_FAILURE_NO_RETRY = "deterministic_failure_no_retry"
    TRANSPORT_RETRIES_EXHAUSTED = "transport_retries_exhausted"
    NON_RETRYABLE = "non_retryable"
    RETRIES_EXHAUSTED = "retries_exhausted"
    TRUNCATION = "truncation"
    UNKNOWN = "unknown"


class ParseFailureKind(StrEnum):
    """How the model response failed to parse (extraction_outcome.parse_failure_kind).

    A closed bucketing of the ``_best_effort_json`` failure modes -- the raw
    ``JSONDecodeError`` message (which can echo model output) never travels."""

    NONE = "none"
    EMPTY_CONTENT = "empty_content"
    JSON_DECODE_ERROR = "json_decode_error"
    NOT_A_JSON_OBJECT = "not_a_json_object"


class FinishReason(StrEnum):
    """Closed set of LLM finish reasons (extraction_outcome.finish_reason). A raw
    provider string is mapped onto this; an unrecognised one becomes ``unknown`` so a
    free value can never ride through."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"
    ERROR = "error"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    """Clinical-invariant outcome severity (the closed set the rule engine emits)."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# The closed set that may appear in a NO_PHI ``doc_type`` field. Derived from the
# enum so the two can never drift. Plain strings for unambiguous membership checks.
VALID_DOC_TYPES: frozenset[str] = frozenset(member.value for member in DocType)

# Lowercased substring -> canonical member. First match wins, so order from most to
# least specific. Quality only affects the "unknown" rate; the closed-set guarantee
# (below) holds regardless of the map.
_DOC_TYPE_PATTERNS: tuple[tuple[str, DocType], ...] = (
    ("patholog", DocType.PATHOLOGY),
    ("radiolog", DocType.RADIOLOGY),
    ("imaging", DocType.RADIOLOGY),
    ("discharge", DocType.DISCHARGE_SUMMARY),
    ("operative", DocType.OPERATIVE_NOTE),
    ("op note", DocType.OPERATIVE_NOTE),
    ("consult", DocType.CONSULT),
    ("immuniz", DocType.IMMUNIZATION),
    ("flowsheet", DocType.FLOWSHEET),
    ("med_admin", DocType.MEDICATION),
    ("med_order", DocType.MEDICATION),
    ("medication", DocType.MEDICATION),
    ("laborator", DocType.LABORATORY),
    ("diagnos", DocType.DIAGNOSIS),
    ("demographic", DocType.DEMOGRAPHICS),
    ("encounter", DocType.ENCOUNTER),
    ("procedure", DocType.PROCEDURE),
    ("progress", DocType.PROGRESS_NOTE),
    ("order", DocType.ORDER),
    ("clinical", DocType.CLINICAL_NOTE),
    ("note", DocType.CLINICAL_NOTE),
    ("lab", DocType.LABORATORY),
)


def canonicalize_doc_type(raw: object) -> str:
    """Map any raw doc_type to a VALID_DOC_TYPES member; never returns the raw value."""
    if not isinstance(raw, str):
        return DocType.UNKNOWN.value
    lowered = raw.strip().lower()
    if not lowered:
        return DocType.UNKNOWN.value
    if lowered in VALID_DOC_TYPES:
        return lowered
    for needle, canonical in _DOC_TYPE_PATTERNS:
        if needle in lowered:
            return canonical.value
    return DocType.UNKNOWN.value


def is_unmapped(raw: object) -> bool:
    """True iff raw was a non-empty string that fell through to "unknown"."""
    return (
        isinstance(raw, str)
        and bool(raw.strip())
        and canonicalize_doc_type(raw) == DocType.UNKNOWN.value
    )
