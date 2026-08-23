"""NO_PHI exhaust layer.

The exhaust is the shareable, text-free, id-free training/measurement signal the
pipeline mints alongside every run. Everything here is built to leave the
institution: controlled vocabularies (`vocabularies`), frozen record schemas
(`schema`), a writer-owned `emit()` with domain-separated surrogates, and a locked
shard->parquet finalize. The PHI seam is enforced *inside* this package — a record
that carries a raw id, a full clinical date, or raw text never reaches a shard.
"""
from __future__ import annotations

from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    VALID_DOC_TYPES,
    VOCAB_VERSION,
    CorrectionType,
    DocType,
    ErrorKind,
    FinishReason,
    HumanTier,
    ParseFailureKind,
    RelevanceLabel,
    Retriever,
    Severity,
    SignalType,
    ValueShape,
    canonicalize_doc_type,
    is_unmapped,
)

# NB: schema.py (pydantic RunHeader/RecipeHeader + record models) is intentionally
# NOT re-exported here -- it pulls pydantic, and this __init__ runs on the
# stdlib-only vocabularies import path. Import it directly from .schema. Its
# production consumer is the writer-owned emit().

__all__ = [
    "VOCAB_VERSION",
    "VALID_DOC_TYPES",
    "CorrectionType",
    "DocType",
    "ErrorKind",
    "FinishReason",
    "HumanTier",
    "ParseFailureKind",
    "RelevanceLabel",
    "Retriever",
    "Severity",
    "SignalType",
    "ValueShape",
    "canonicalize_doc_type",
    "is_unmapped",
]
