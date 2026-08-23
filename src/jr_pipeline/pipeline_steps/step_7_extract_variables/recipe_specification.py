"""recipe spec loader + validator.

A "recipe" is the complete, self-contained plan for extracting one clinical
variable (say, cancer stage or date of death) from a patient's chart. It spells
out the whole path end to end:
  - retrieval: pull the chart passages most likely to mention the variable
    (a mix of vector search over text meaning and classic keyword search),
  - reranking: re-order those passages so the best evidence is on top,
  - evidence packing: fit the chosen passages into the model's prompt,
  - llm config: which model to call and with what settings,
  - post-processing: tidy/normalize the model's answer,
  - validation: sanity-check the final value.
Keeping all of this in one folder means everything needed to extract a variable
ships and versions together, so a recipe behaves the same wherever it runs.

All paths inside a recipe yaml resolve relative to the yaml file itself. That
keeps recipe folders portable — drop one into a new repo and its prompt +
schema references still point at the right files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined

from jr_pipeline.pipeline_steps.step_6_prepare_evidence_for_extraction.prepare_evidence import (
    DEFAULT_MAX_CONTEXT_TOKENS,
)

# The four kinds of step a recipe may use: ask the model after retrieving chart
# evidence; ask the model with no retrieval; read a value straight from a
# structured table; or run a plain python post-processing helper. Anything else
# in a recipe yaml is a typo or an unmerged feature — refuse it loudly.
_VALID_KINDS = {
    "retrieve_and_prompt",
    "map_table_rows_and_prompt",
    "llm_only",
    "direct_parquet",
    "python",
}

# "archetype" labels the broad shape of an extraction task (a single fact, a
# date, a sequence of events, ...). It is written verbatim into the run's
# permanent, append-only metadata log as the main grouping key, so letting
# authors invent free-form labels would permanently splinter the pooled records.
# Restrict it at authoring time, the same way step kinds are restricted. A
# genuinely new pattern declares `archetype_proposed: true` (the channel for
# proposing a new label) instead of inventing one on the spot.
VALID_ARCHETYPES = frozenset({
    "point_fact",
    "temporal_anchor",
    "event_sequence",
    "classification_adjudication",
    "enumeration",
    "causal",
})
_STEP_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
# A step may carry a `stop_if` condition that ends the recipe early (e.g. "stop
# if no cancer was found"). We use the same template engine the extract step
# uses to evaluate it, and compile the expression here at load time so a typo
# fails loudly instead of silently always evaluating to False forever.
_STOP_IF_ENV = Environment(undefined=StrictUndefined)

# The set of infrastructure blocks a recipe may carry is closed. `retrieval`,
# `model_policy`, `feedback`, and `evidence.formatting_style` are no longer read
# anywhere. A recipe still carrying one is an out-of-date copy, so reject it
# loudly instead of silently filing it under front_matter (the catch-all bucket
# for recipe metadata). The `evidence` sub-block is locked down separately against
# EvidenceSpec's own fields; these top-level blocks sit among open-ended
# front_matter keys, so they need an explicit deny list.
_REMOVED_TOP_LEVEL_KEYS = ("retrieval", "model_policy", "feedback")

# The keys a recipe may set under `llm:`. Anything else is a typo (or an unmerged
# feature) and is refused loudly — a misspelled optional knob would otherwise be
# silently ignored and the recipe would run without the behavior it asked for.
_RECIPE_LLM_KEYS = frozenset(
    {"model", "max_tokens", "seed", "expected_fingerprint", "retry_cut_off_answers"}
)

# `temperature` and `response_format` are fixed by the engine, not knobs a
# recipe may set. A clinical extractor pins greedy decoding (temperature=0.0 —
# the model always picks its single most likely next token, so the same input
# gives the same output every time, which is reproducible and lets the response
# cache work) and JSON output (response_format="json_object"). Letting a recipe
# override either would re-introduce random sampling or a non-JSON response, so
# reject both loudly under `llm:`; the engine supplies them (LLMSpec defaults
# below). `model` + `max_tokens` stay recipe-owned; `seed` /
# `expected_fingerprint` remain optional knobs for reproducibility and for
# pinning a specific model endpoint.
_ENGINE_DEFAULT_LLM_KEYS = ("temperature", "response_format")


@dataclass(frozen=True)
class LLMSpec:
    """settings for calling the language model; one per recipe, shared by every step."""

    model: str
    # Fixed by the engine, not overridable per recipe:
    # temperature 0.0 = the model always takes its single most likely next token
    # (same input -> same output); response_format = the model must return JSON.
    temperature: float = 0.0
    max_tokens: int = 1024
    response_format: str | None = "json_object"
    seed: int | None = None
    # When true, an answer that comes back cut off (the model hit its token limit
    # mid-JSON) is retried with a bigger max_tokens budget instead of being shipped
    # truncated. Off by default because the retry re-spends tokens on the same
    # question; recipes whose answers are long (e.g. a whole treatment history)
    # opt in here. The recipe declares this — the pipeline holds no list of which
    # variables qualify.
    retry_cut_off_answers: bool = False
    # When set, a cached response whose model fingerprint (a short value
    # identifying the exact model/endpoint that produced it) no longer matches is
    # treated as a cache miss — this catches the gateway silently swapping the
    # endpoint to a different model behind the same name (ADR 0028).
    expected_fingerprint: str | None = None


@dataclass(frozen=True)
class StepSpec:
    """one step of a recipe; ``kind`` picks the handler in the step registry."""

    id: str
    kind: str
    config: dict[str, Any]
    prompt: Path | None = None
    module: str | None = None
    stop_if: str | None = None


@dataclass(frozen=True)
class FilterSpec:
    """one metadata condition the reranker uses to keep or drop a candidate chunk.

    ``value`` may still contain a ``{{ vars.* }}`` reference here (e.g. comparing a
    document date against an upstream diagnosis date); it is resolved per patient
    just before reranking. A list value is allowed for the ``in`` operator."""

    field: str
    op: str
    value: Any
    keep_if_missing: bool = False


@dataclass(frozen=True)
class RerankingSpec:
    """how step 5 re-orders retrieved evidence.

    The cross-encoder is a toggle; everything else (metadata/date filtering,
    duplicate-text removal, the rule-based combined score, date-ordered selection,
    the optional re-sort into reading order) is the candidate ranker.
    ``model_id`` names an entry in models_registry.yaml (a lookup key, not a path)
    and is required only when ``cross_encoder`` is true. ``rank_by`` chooses which
    scorer selects the top_n and defaults to the cross-encoder when one is on, the
    combined score otherwise. ``dedup_identical_text`` drops candidates that repeat
    text already kept — on by default, because two copies of one note are not two
    pieces of evidence. ``scorer_config`` carries the combined-score knobs (weights,
    source_priority, k0) and any cross-encoder runtime knobs."""

    cross_encoder: bool = False
    model_id: str | None = None
    rank_by: str | None = None
    dedup_identical_text: bool = True
    top_n: int = 10
    resort_by_date: bool = False
    chronological_order: str | None = None
    filters_fallback_to_unfiltered: bool = False
    filters: tuple[FilterSpec, ...] = ()
    scorer_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSpec:
    """how the selected chart passages are assembled for the prompt.
    ``max_context_tokens`` is a loud sanity ceiling, not a packing budget: step 6
    includes every passage step 5 selected and fails loudly if the assembled total
    (estimated, chars/4) exceeds this, rather than silently truncating. It defaults
    high; set it to a specific model's real context window for a tighter guard."""

    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS


@dataclass(frozen=True)
class ValidationSpec:
    """post-extraction validation rules for this variable."""

    rules: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RecipeSpec:
    """fully-validated recipe — the complete extraction path for a variable."""

    name: str
    version: str
    path: Path
    output_schema_path: Path
    llm: LLMSpec
    depends_on: list[str]
    steps: list[StepSpec]
    reranking: RerankingSpec = field(default_factory=RerankingSpec)
    evidence: EvidenceSpec = field(default_factory=EvidenceSpec)
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    front_matter: dict[str, Any] = field(default_factory=dict)


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"recipe {ctx}: missing required key {key!r}")
    return d[key]


# Top-level reranking keys the loader handles directly; anything else is a
# combined-score / cross-encoder runtime knob and goes into scorer_config.
_RERANKING_TOP_KEYS = {
    "cross_encoder",
    "model_id",
    "rank_by",
    "dedup_identical_text",
    "top_n",
    "resort_by_date",
    "chronological_order",
    "filters_fallback_to_unfiltered",
    "filters",
}


def _parse_filter(raw: Any, path: Path) -> FilterSpec:
    """Validate one reranking filter clause and return a FilterSpec."""
    # Import the canonical field/operator vocabulary from the executor so there is
    # one source of truth (kept here as a function-level import to keep this
    # module's top-level imports light).
    from jr_pipeline.pipeline_steps.step_5_rerank_chunks.filter_candidates import (
        ALL_OPS,
        ORDERED_OPS,
    )
    from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
        METADATA_FIELD_KINDS,
        ORDERED_METADATA_KINDS,
    )
    from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
        STANDARD_METADATA_FIELDS as FILTERABLE_FIELDS,
    )

    if not isinstance(raw, dict):
        raise ValueError(f"recipe {path}: each reranking filter must be a mapping, got {raw!r}")
    field_name = raw.get("field")
    op = raw.get("op")
    if field_name not in FILTERABLE_FIELDS:
        raise ValueError(
            f"recipe {path}: reranking filter field {field_name!r} is not filterable; "
            f"choose one of {sorted(FILTERABLE_FIELDS)}"
        )
    if op not in ALL_OPS:
        raise ValueError(
            f"recipe {path}: reranking filter op {op!r} is unknown; "
            f"choose one of {sorted(ALL_OPS)}"
        )
    if op in ORDERED_OPS and METADATA_FIELD_KINDS[field_name] not in ORDERED_METADATA_KINDS:
        orderable = sorted(
            f for f, kind in METADATA_FIELD_KINDS.items() if kind in ORDERED_METADATA_KINDS
        )
        raise ValueError(
            f"recipe {path}: reranking filter op {op!r} orders its operands, and "
            f"{field_name!r} holds text. Orderable fields: {orderable}"
        )
    if "value" not in raw:
        raise ValueError(f"recipe {path}: reranking filter on {field_name!r} needs a 'value'")
    return FilterSpec(
        field=field_name,
        op=op,
        value=raw["value"],
        keep_if_missing=bool(raw.get("keep_if_missing", False)),
    )


def _parse_reranking(reranking_raw: dict[str, Any], path: Path) -> RerankingSpec:
    """Build a RerankingSpec, rejecting the removed ``kind`` field loudly."""
    if "kind" in reranking_raw:
        raise ValueError(
            f"recipe {path}: reranking.kind was removed. Use 'cross_encoder: true|false' "
            "to pick the scorer, 'resort_by_date: true' for newest-document-first order, "
            "and 'filters:' for metadata/date filtering."
        )
    filters = tuple(_parse_filter(f, path) for f in (reranking_raw.get("filters") or []))
    chronological_order = reranking_raw.get("chronological_order")
    if chronological_order not in (None, "newest_first", "oldest_first"):
        raise ValueError(
            f"recipe {path}: reranking.chronological_order must be "
            "'newest_first' or 'oldest_first'"
        )
    # Vocabulary check only: whether the chosen scorer is compatible with the
    # cross_encoder toggle is CandidateRanker's rule, enforced where it is defined.
    from jr_pipeline.pipeline_steps.step_5_rerank_chunks.rank_candidates import (
        COMBINED_SCORE_SIGNALS,
        RANK_BY_CHOICES,
    )

    rank_by = reranking_raw.get("rank_by")
    if rank_by is not None and rank_by not in RANK_BY_CHOICES:
        raise ValueError(
            f"recipe {path}: reranking.rank_by must be one of {list(RANK_BY_CHOICES)}"
        )
    # Scoring looks up the weight of each signal it computes, so a weight naming anything
    # else is silently ignored — an author retunes the ranking, sees no change, and gets
    # no error. Caught here, before the run, rather than never.
    weights = reranking_raw.get("weights") or {}
    if not isinstance(weights, dict):
        raise ValueError(
            f"recipe {path}: reranking.weights must map a signal name to a number"
        )
    unknown_signals = sorted(set(weights) - set(COMBINED_SCORE_SIGNALS))
    if unknown_signals:
        raise ValueError(
            f"recipe {path}: reranking.weights names no such signal: {unknown_signals}; "
            f"choose from {list(COMBINED_SCORE_SIGNALS)}"
        )
    return RerankingSpec(
        cross_encoder=bool(reranking_raw.get("cross_encoder", False)),
        model_id=reranking_raw.get("model_id"),
        rank_by=rank_by,
        dedup_identical_text=bool(reranking_raw.get("dedup_identical_text", True)),
        top_n=int(reranking_raw.get("top_n", 10)),
        resort_by_date=bool(reranking_raw.get("resort_by_date", False)),
        chronological_order=chronological_order,
        filters_fallback_to_unfiltered=bool(
            reranking_raw.get("filters_fallback_to_unfiltered", False)
        ),
        filters=filters,
        scorer_config={
            k: v for k, v in reranking_raw.items() if k not in _RERANKING_TOP_KEYS
        },
    )


def _reject_removed_keys(raw: dict[str, Any], path: Path) -> None:
    """Refuse a recipe that still carries blocks the engine no longer reads."""
    present = [k for k in _REMOVED_TOP_LEVEL_KEYS if k in raw]
    if present:
        raise ValueError(
            f"recipe {path}: top-level key(s) {present} were removed and are no longer "
            "honored — delete them. Per-step retrieval config belongs under each step "
            "in 'steps:', not at recipe top level."
        )
    evidence_raw = raw.get("evidence")
    if isinstance(evidence_raw, dict):
        # Only the keys EvidenceSpec actually parses are allowed; anything else (e.g. the
        # removed formatting_style, or a typo) is a stale recipe. Compare against the
        # dataclass's own fields so the two can't drift apart.
        valid_evidence_keys = {spec_field.name for spec_field in fields(EvidenceSpec)}
        unsupported = sorted(set(evidence_raw) - valid_evidence_keys)
        if unsupported:
            raise ValueError(
                f"recipe {path}: unsupported key(s) {unsupported} under 'evidence:' — only "
                f"{sorted(valid_evidence_keys)} valid."
            )
    llm_raw = raw.get("llm")
    if isinstance(llm_raw, dict):
        present_llm = [k for k in _ENGINE_DEFAULT_LLM_KEYS if k in llm_raw]
        if present_llm:
            raise ValueError(
                f"recipe {path}: llm key(s) {present_llm} are engine defaults and may "
                "not be set per-recipe — remove them from 'llm:'. The engine fixes "
                "temperature=0.0 (greedy/deterministic) and response_format='json_object'."
            )
        unknown_llm = sorted(set(llm_raw) - _RECIPE_LLM_KEYS)
        if unknown_llm:
            raise ValueError(
                f"recipe {path}: unknown llm key(s) {unknown_llm} — valid keys: "
                f"{sorted(_RECIPE_LLM_KEYS)}. A misspelled key would be silently "
                "ignored, so it is refused instead."
            )


def load_recipe(path: Path) -> RecipeSpec:
    """read a recipe yaml file and check it is well-formed, returning a fully
    validated RecipeSpec; raises ValueError on any structural problem."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"recipe {path}: top-level must be a mapping")
    _reject_removed_keys(raw, path)

    name = _require(raw, "name", str(path))
    version = str(_require(raw, "version", str(path)))
    output_schema_rel = _require(raw, "output_schema", str(path))
    output_schema_path = (path.parent / output_schema_rel).resolve()
    # Fail at load, not at extract: a typo'd schema path would otherwise render an
    # empty {{ OUTPUT_SCHEMA }} into every prompt and only surface at step 8.
    if not output_schema_path.is_file():
        raise ValueError(
            f"recipe {path}: output_schema {output_schema_rel!r} not found at "
            f"{output_schema_path}"
        )

    llm_raw = raw.get("llm") or {}
    # A recipe that only reads values from structured tables (direct_parquet) and
    # tidies them with python steps legitimately never calls a model, so `llm.model`
    # is required only when at least one step actually calls one.
    _llm_step_kinds = {str(s.get("kind")) for s in (raw.get("steps") or []) if isinstance(s, dict)}
    _needs_llm = bool(
        _llm_step_kinds
        & {"retrieve_and_prompt", "map_table_rows_and_prompt", "llm_only"}
    )
    # temperature / response_format are engine defaults — _reject_removed_keys has
    # already refused any recipe that tries to set them, so we never read them here;
    # LLMSpec supplies the fixed values.
    llm = LLMSpec(
        model=(str(_require(llm_raw, "model", f"{path}:llm")) if _needs_llm
               else str(llm_raw.get("model", "none"))),
        max_tokens=int(llm_raw.get("max_tokens", 1024)),
        seed=llm_raw.get("seed"),
        retry_cut_off_answers=bool(llm_raw.get("retry_cut_off_answers", False)),
        expected_fingerprint=llm_raw.get("expected_fingerprint"),
    )

    depends_on = list(raw.get("depends_on") or [])

    stages_raw = raw.get("steps") or []
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ValueError(f"recipe {path}: 'steps' must be a non-empty list")

    steps: list[StepSpec] = []
    for i, s in enumerate(stages_raw):
        if not isinstance(s, dict):
            raise ValueError(f"recipe {path}: steps[{i}] must be a mapping")
        sid = _require(s, "id", f"{path}:steps[{i}]")
        kind = _require(s, "kind", f"{path}:steps[{i}] id={sid}")
        if kind not in _VALID_KINDS:
            raise ValueError(f"recipe {path}: unknown step kind {kind!r}; supported: {sorted(_VALID_KINDS)}")
        if not _STEP_ID_RE.fullmatch(str(sid)):
            raise ValueError(
                f"recipe {path}: step id {sid!r} must match [A-Za-z0-9_-]+ "
                "(it is used as a path segment and a step_outputs key)"
            )
        retrieval_raw = s.get("retrieval")
        if kind == "retrieve_and_prompt" and isinstance(retrieval_raw, dict):
            include_recent = retrieval_raw.get("include_recent")
            if (
                include_recent is not None
                and (
                    isinstance(include_recent, bool)
                    or not isinstance(include_recent, int)
                    or include_recent < 1
                )
            ):
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} retrieval.include_recent "
                    "must be a positive integer"
                )
        step_reranking = s.get("reranking")
        if step_reranking is not None:
            if kind != "retrieve_and_prompt" or not isinstance(step_reranking, dict):
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} reranking must be a mapping "
                    "on a retrieve_and_prompt step"
                )
            unsupported = set(step_reranking) - {"top_n", "chronological_order"}
            if unsupported:
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} unsupported per-step "
                    f"reranking keys {sorted(unsupported)}"
                )
            order = step_reranking.get("chronological_order")
            if order not in (None, "newest_first", "oldest_first"):
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} reranking.chronological_order "
                    "must be 'newest_first' or 'oldest_first'"
                )
            if "top_n" in step_reranking and int(step_reranking["top_n"]) < 1:
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} reranking.top_n "
                    "must be a positive integer"
                )
        prompt_rel = s.get("prompt")
        prompt_path = (path.parent / prompt_rel).resolve() if prompt_rel else None
        stop_if = s.get("stop_if")
        if stop_if:
            try:
                _STOP_IF_ENV.compile_expression(str(stop_if))
            except Exception as e:
                raise ValueError(
                    f"recipe {path}: steps[{i}] id={sid} has an invalid stop_if "
                    f"expression {stop_if!r}: {e}"
                ) from None
        steps.append(
            StepSpec(
                id=sid,
                kind=kind,
                config={k: v for k, v in s.items() if k not in {"id", "kind", "prompt", "stop_if", "module"}},
                prompt=prompt_path,
                module=s.get("module"),
                stop_if=stop_if,
            )
        )

    step_ids = [s.id for s in steps]
    if len(step_ids) != len(set(step_ids)):
        dups = sorted({sid for sid in step_ids if step_ids.count(sid) > 1})
        raise ValueError(
            f"recipe {path}: duplicate step ids {dups} — each step id is used both as a "
            "lookup key for that step's output and as its steps/<id>/receipt.json audit "
            "path, so two steps sharing an id would silently overwrite each other."
        )
    # Folder convention: <recipes_root>/<name>/v<N>/<name>_v<N>_recipe.yaml. If the
    # recipe's declared name disagrees with its folder, its results would be filed
    # under the wrong variable.
    folder = path.parent.parent.name
    if name != folder:
        raise ValueError(f"recipe {path}: name {name!r} does not match its folder {folder!r}")
    # The declared version must match the version folder for the same reason: the
    # sealed per-recipe index is keyed by FOLDER names while extract looks entries up
    # by the DECLARED version, so "v1/" holding `version: 1` silently gets no sealed
    # entry — receipts then stamp null hashes and resume cannot trust the result.
    version_folder = path.parent.name
    if version != version_folder:
        raise ValueError(
            f"recipe {path}: version {version!r} does not match its folder "
            f"{version_folder!r}"
        )

    reranking_raw = raw.get("reranking") or {}
    reranking = _parse_reranking(reranking_raw, path)

    evidence_raw = raw.get("evidence") or {}
    evidence = EvidenceSpec(
        max_context_tokens=int(
            evidence_raw.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        ),
    )

    validation_raw = raw.get("validation") or {}
    validation = ValidationSpec(
        rules=list(validation_raw.get("rules") or []),
    )

    # Any keys not handled above are kept in front_matter, the catch-all bucket, so
    # recipe-level metadata (owner, notes) stays readable without having to formally
    # add every new field here.
    _known_keys = {
        "name", "version", "output_schema", "llm", "depends_on", "steps",
        "reranking", "evidence", "validation",
    }

    front_matter = {k: v for k, v in raw.items() if k not in _known_keys}
    _validate_archetype(front_matter, path)

    # A scaffolded recipe carries this line until its author removes it. Copying a
    # working recipe gets the WIRING right -- file names, schema and prompt paths, the
    # python module reference -- and cannot get the clinical content right: the prompts
    # still ask the question the original asked and the schema still names its fields.
    # A half-authored recipe that RAN would return confident, well-formed, fully
    # provenanced values for the wrong variable, which is the worst outcome available
    # here. Removing the line is the author saying the content is theirs now. Checked
    # LAST, after everything else, so draft-tolerant tooling (the app's recipe editor
    # validates every save) can catch exactly this error and know the rest is sound.
    if raw.get("needs_editing"):
        raise ValueError(
            f"recipe {path}: scaffolded from another variable and not finished yet. "
            "Its prompts and output schema still describe the variable it was copied "
            "from, so running it would produce confident answers to the wrong question. "
            "Edit them, then delete the `needs_editing: true` line from this file."
        )

    return RecipeSpec(
        name=name,
        version=version,
        path=path,
        output_schema_path=output_schema_path,
        llm=llm,
        depends_on=depends_on,
        steps=steps,
        reranking=reranking,
        evidence=evidence,
        validation=validation,
        front_matter=front_matter,
    )


def _validate_archetype(front_matter: dict[str, Any], path: Path) -> None:
    """Check the recipe's `archetype` against the allowed, fixed list of labels at
    load time (or accept an explicit proposal of a new one)."""
    archetype = front_matter.get("archetype")
    if archetype is None:
        raise ValueError(
            f"recipe {path}: missing required 'archetype' (one of "
            f"{sorted(VALID_ARCHETYPES)}, or set 'archetype_proposed: true')"
        )
    if archetype in VALID_ARCHETYPES:
        return
    if front_matter.get("archetype_proposed"):
        return  # a genuinely new pattern, flagged for review to maybe add it to the
                 # allowed list later — not treated as an error
    raise ValueError(
        f"recipe {path}: unknown archetype {archetype!r}; controlled vocabulary is "
        f"{sorted(VALID_ARCHETYPES)}. For a genuinely new pattern set "
        f"'archetype_proposed: true' to flag it for promotion."
    )
