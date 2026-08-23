"""Frozen pydantic schemas for the NO_PHI exhaust records.

Every record that leaves the institution is one of the models registered here. Two
frozen headers carry the identity that makes a row poolable across runs/sites:

* ``RunHeader`` — schema/vocab version, site/study/run identity, the method
  fingerprints (encoder/reranker/extractor + chunking), the sealed ``code_lock_hash``,
  and the month stamp. Run-level records (``method_provenance``) use it directly.
* ``RecipeHeader`` — ``RunHeader`` plus the recipe identity (id/version/variable),
  archetype, and the sealed prompt/schema hashes. Recipe-level records
  (``extraction_outcome``, ``selection_judgment``) use it.

``extra='forbid'`` + typed enum fields are the schema half of the write gate: a stray
key or an out-of-vocab categorical fails validation loudly, before any row
reaches a shard. ``emitted_month`` is pattern-pinned to ``YYYY-MM`` (or ``unknown``)
so a full clinical date can never masquerade as the month stamp.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jr_pipeline.runtime_infrastructure.exhaust.vocabularies import (
    VOCAB_VERSION,
    DocType,
    ErrorKind,
    FinishReason,
    HumanTier,
    ParseFailureKind,
    RelevanceLabel,
    Retriever,
    Severity,
    ValueShape,
)

# A controlled error code: a hash of the raw error (never the raw text). error_kind
# carries the bucket, this the fine digest.
# (A pure-decimal run that slips this schema pattern is still rejected by the
# forbidden_content scanner's hex-letter _HASH_RE + 5+-digit-run guard at emit time;
# pydantic's Rust regex engine has no lookahead, so the letter requirement lives there.)
_HASH_PATTERN = r"^(?:sha256:[0-9a-f]{3,64}|[0-9a-f]{16,64})$"

# The exhaust schema version. Additive columns -> minor; a removed/retyped/
# re-meant column -> major. Independent of the PHI artifact-envelope schema version.
# 2.0.0 (MAJOR): SelectionJudgment.available_evidence_tokens (the input budget
# ceiling) was removed and re-meant as total_evidence_tokens (the actual assembled
# evidence size) when step 6 became a consolidator -- a re-meant column, which the
# policy above makes a major bump. This release also adds the human-review record
# types (relevance_label, retriever_miss) and routes the evidence-chunk surrogate
# through the shared EVIDENCE_CHUNK_LINKAGE_DOMAIN so a human label joins to its
# selection_judgment row (those additions are themselves additive).
SCHEMA_VERSION = "2.0.0"

_MONTH_OR_UNKNOWN = r"^(\d{4}-\d{2}|unknown)$"


class _ExhaustBase(BaseModel):
    """Frozen, closed base for every exhaust model (a stray key is a hard error)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class RunHeader(_ExhaustBase):
    """Run-level identity stamped on every exhaust record."""

    schema_version: str = SCHEMA_VERSION
    vocab_version: str = VOCAB_VERSION
    site_id: str
    study_id: str
    run_id: str
    emitted_month: str = Field(pattern=_MONTH_OR_UNKNOWN)
    encoder_fingerprint: str
    reranker_fingerprint: str
    extractor_fingerprint: str
    code_lock_hash: str
    chunking_config_id: str


class RecipeHeader(RunHeader):
    """Recipe-level identity: ``RunHeader`` plus the question/recipe the row answers."""

    recipe_id: str
    recipe_version: str
    variable_key: str  # the question id; equals recipe_id today (one recipe -> one var)
    archetype: str  # gated against VALID_ARCHETYPES upstream at recipe load
    prompt_hash: str
    schema_hash: str


class SelectionJudgmentRow(_ExhaustBase):
    """One ranked candidate in a selection_judgment record. No raw id/text/date."""

    chunk_surrogate: str
    doc_type: DocType
    rank: int | None = None
    prior_rank: int | None = None
    score: float | None = None
    token_count: int | None = None
    in_bundle: bool
    cited: bool
    # The reranker's per-candidate features -- combined-score contributions or
    # {"cross_encoder_logit": <4dp>} -- the reranker-optimization training signal. Pure
    # numbers (keys are controlled feature names); the cross-encoder rank is `rank`.
    features: dict[str, float] = Field(default_factory=dict)


class SelectionJudgment(RecipeHeader):
    """The shareable per-(patient, recipe, step) relevance signal."""

    record_type: Literal["selection_judgment"] = "selection_judgment"
    group: str
    step_id: str
    retriever: Retriever
    total_evidence_tokens: int | None = None
    n_candidates: int
    n_in_bundle: int
    n_cited: int
    n_doc_type_unmapped: int
    rows: list[SelectionJudgmentRow]


class ClinicalInvariantOutcome(_ExhaustBase):
    """One invariant rule outcome. Only the controlled id/ok/severity travel; the
    rule context (which carries dates) is never emitted."""

    rule_id: str
    ok: bool
    severity: Severity | None = None


class ClinicalInvariant(RunHeader):
    """An independent, model-free weak labeler over a patient."""

    record_type: Literal["clinical_invariant"] = "clinical_invariant"
    labeler_id: str
    labeler_version: str
    patient_surrogate: str
    outcomes: list[ClinicalInvariantOutcome]
    n_violations: int


class MethodProvenance(RunHeader):
    """One row per run recording the method identity (all of it lives in RunHeader).
    The flywheel's join key: it ties every other record's fingerprints to a run."""

    record_type: Literal["method_provenance"] = "method_provenance"


class ExtractionOutcome(RecipeHeader):
    """Per-(patient, recipe) extraction result signal (the flywheel ignition).

    Carries *why* an extraction succeeded/failed and what it cost -- never the value,
    the prompt, the response, or any exception text. ``value_shape``/``n_value_fields``
    describe the output form without the content; ``error_kind``/``parse_failure_kind``
    bucket the failure; the token counts are economics, not PHI."""

    record_type: Literal["extraction_outcome"] = "extraction_outcome"
    patient_surrogate: str
    ok: bool
    value_shape: ValueShape
    n_value_fields: int = 0
    error_kind: ErrorKind = ErrorKind.NONE
    error_code: str | None = Field(default=None, pattern=_HASH_PATTERN)  # a hash, never raw text
    parse_failure_kind: ParseFailureKind = ParseFailureKind.NONE
    n_passes: int = 1
    retry_attempts: int = 0
    finish_reason: FinishReason | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached: bool = False


class ReviewHeader(RunHeader):
    """``RunHeader`` plus the question a human reviewed (recipe + step).

    Lighter than ``RecipeHeader``: a relevance judgment must locate the question
    (recipe_id/version, step_id) and record which retriever/reranker version was in play
    -- both already in ``RunHeader``'s method fingerprints -- but it does not depend on
    the extraction prompt/schema/archetype, so those are omitted."""

    recipe_id: str
    recipe_version: str
    step_id: str


class EvidenceRelevanceLabel(ReviewHeader):
    """A reviewer's relevance verdict on one candidate chunk -- the cross-encoder
    fine-tuning target.

    Joins to a ``SelectionJudgmentRow`` on (run_id, recipe_id, step_id,
    ``chunk_surrogate``): the chunk surrogate is minted in the shared
    ``EVIDENCE_CHUNK_LINKAGE_DOMAIN`` so it equals the selection_judgment row's, which is
    how a label is tied back to that chunk's retriever_rank/cross_encoder_rank. The
    patient/reviewer surrogates are per-record-type (no cross-type patient join); only the
    controlled ``label`` + ``reviewer_tier`` carry the human signal."""

    record_type: Literal["relevance_label"] = "relevance_label"
    patient_surrogate: str
    reviewer_surrogate: str
    reviewer_tier: HumanTier = HumanTier.UNKNOWN
    chunk_surrogate: str
    label: RelevanceLabel


class RetrieverMiss(ReviewHeader):
    """A reviewer found relevant evidence the retriever missed or buried -- the
    retriever-improvement signal.

    Only the structural outcome travels: the manual search *query* is free EHR-adjacent
    text and stays PHI-side, never here. ``chunk_surrogate`` is the reviewer-selected
    chunk, minted in the shared ``EVIDENCE_CHUNK_LINKAGE_DOMAIN``; when
    ``was_in_retriever_pool`` it joins to that chunk's ``selection_judgment`` row and the
    recorded ranks attribute the miss to the cross-encoder, otherwise retrieval/chunking
    is the bottleneck."""

    record_type: Literal["retriever_miss"] = "retriever_miss"
    patient_surrogate: str
    reviewer_surrogate: str
    reviewer_tier: HumanTier = HumanTier.UNKNOWN
    chunk_surrogate: str
    was_in_retriever_pool: bool
    retriever_rank_if_present: int | None = None
    cross_encoder_rank_if_present: int | None = None
    manual_search_used: bool = True


# record_type -> model. The single source of truth emit() validates against.
RECORD_SCHEMAS: dict[str, type[_ExhaustBase]] = {
    "selection_judgment": SelectionJudgment,
    "clinical_invariant": ClinicalInvariant,
    "method_provenance": MethodProvenance,
    "extraction_outcome": ExtractionOutcome,
    "relevance_label": EvidenceRelevanceLabel,
    "retriever_miss": RetrieverMiss,
}


def validate_record(record_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Validate a record dict against its registered schema; return the canonical
    JSON-able dict. Raises ``KeyError`` for an unknown record_type and
    ``pydantic.ValidationError`` for a stray key / out-of-vocab / missing field."""
    try:
        model = RECORD_SCHEMAS[record_type]
    except KeyError:
        raise KeyError(f"unknown exhaust record_type: {record_type!r}") from None
    return model(**data).model_dump(mode="json")
