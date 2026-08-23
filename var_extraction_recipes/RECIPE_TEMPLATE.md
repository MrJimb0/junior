# Junior recipe template — the one clinician-readable standard

Every variable-extraction recipe follows this fixed six-section shape, in this order.
The point is that a clinician can read a recipe top-to-bottom and understand exactly what
it extracts, what evidence it looks at, how it decides, and how every value is grounded —
without reading any engine code.

A recipe is a folder:

```
var_extraction_recipes/<collection>/<subdomain>/<variable>/v1/
    <variable>_v1_recipe.yaml          # the six sections below
    <variable>_v1_output_schema.json   # THE output contract (section 5)
    <variable>_v1_python_helper.py     # the finalize merge (usual section-3 authority)
    prompts/
        <variable>_v1_<pass>.md         # one per LLM pass
```

The variable's directory name **is** the recipe `name`; `collection:` **is** the
top-level directory (`basic` / `oncology`). Adopted in Phase 5b (S5b-03) and migrated one
recipe at a time (S5b-04); as of S5b-05 every recipe conforms and the conformance guard
(`security_and_pipeline_tests/guardrails/test_recipe_template_conformance.py`) enforces the
machine-checkable rule strictly (no recipe may inject the full schema into a partial pass).

---

## 1. INTENT

A comment block at the top of `recipe.yaml` that states, in clinician language, **what
clinical variable this recipe extracts and why**, plus the scope boundaries (what counts,
what is explicitly excluded, how it relates to neighbouring recipes). This is the first
thing a reviewer reads.

## 2. EVIDENCE POLICY

How evidence is gathered and ranked — the keys an author tunes:

- `roles:` — the source files this recipe reads (e.g. `pathology_reports: { source_file: pathology_report.csv }`).
- `steps[].retrieval.include_recent: N` — add up to N newest dated chunks to the
  relevance hits, deduplicated and restricted by the step's optional `source_file`.
  Use this when a late result or treatment change must not be missed.
- `reranking:` — step 5 selects which chunks reach the model and in what order:
  - `cross_encoder: true` (+ `model_id:`) scores each (query, chunk) pair with the learned
    reranker; omit it (default) to use the rule-based combined score (`weights`, `source_priority`, `k0`).
  - `rank_by:` — which scorer selects the `top_n`: `cross_encoder`, `combined_score`,
    `newest_documents` or `oldest_documents`. Omit it and the cross-encoder selects when
    it is on, the combined score otherwise. Use `newest_documents` when the question is
    "what happened most recently" rather than "what is most on topic" — it takes the N
    most recent documents that pass the filters, undated ones last. Asking for
    `cross_encoder` without turning it on (or turning it on and ranking by something
    else) is refused rather than silently ignored.
  - `dedup_identical_text: false` — keep chunks whose text exactly repeats another
    chunk's. On by default: a copy-forward note or a document ingested twice would
    otherwise spend two `top_n` slots, and repetition reads to the model as
    corroboration. Matching is exact after whitespace/case folding, never fuzzy.
  - `filters:` — drop candidates by metadata before scoring; a list of
    `{ field, op, value, keep_if_missing? }`. `field` is one of `document_date`, `age`,
    `doc_type`, `encounter_id`, `author`, `linked_author`, `title`, `specialty`
    (`STANDARD_METADATA_FIELDS`); `op` is `contains` / `not_contains` / `==` / `!=` / `in`
    (any field) or `>` / `>=` / `<` / `<=`, which order their operands and so need a field
    that can be ordered — a date (`document_date`) or a number (`age`). Each field's kind
    is declared once in `METADATA_FIELD_KINDS`, and it decides how the comparison is made:
    dates compare as ranges, numbers numerically (`age: ">=", 18` keeps 18 and 62.4 but not
    9, however the site typed the column), text as case-insensitive text. `value` may reference an
    upstream variable, e.g. `"{{ vars.date_of_diagnosis.data.date_of_diagnosis }}"` (declare it
    in `depends_on`). A missing field drops the chunk unless `keep_if_missing: true`.
    Filters apply to structured-table rows (`:TABLE` evidence from a `direct_parquet`
    step) as well as free text: a table row's metadata is read from its own columns,
    through the mapping ingest resolved for that file. So "the most recent result signed
    by medical oncology" is a `filters:` clause on `author` plus
    `rank_by: newest_documents`, whether the evidence is notes or table rows. A file that
    genuinely has no column for the filtered field still counts as missing.
  - `filters_fallback_to_unfiltered: true` — when valid filters remove every
    candidate, retry the same candidate pool without filters and record the fallback.
    Use only where the clinical contract explicitly prefers unfiltered evidence over
    an empty packet.
  - `resort_by_date: true` — after selecting `top_n`, re-order them newest-document-first.
  - `chronological_order: newest_first|oldest_first` — explicit evidence reading
    order. Prefer this when the extractor's temporal direction is clinically material.
    This is the READING order only and composes with any `rank_by`: e.g. let the
    cross-encoder trim to the 5 best, then read them oldest-first.
  - A `retrieve_and_prompt` step may override only `top_n` and
    `chronological_order` under its own `reranking:` block when different passes in
    one recipe require different temporal directions.
  - Filters only help if search hands over a rich pool, so set retrieval `k` well above `top_n`.
- `evidence.max_context_tokens:` — optional. Step 6 includes every passage step 5 selected (it does NOT pack to a budget); this is only a loud sanity ceiling — if the assembled evidence (estimated chars/4) exceeds it, the run fails rather than letting the model silently truncate. Defaults high; set it to a specific model's real context window for a tighter guard. Omit it for the default.

`temperature` and `response_format` are **engine defaults** — a recipe does not declare
them (target state; the S5b-04 migration removes them from `llm:` and leaves only `model`
+ `max_tokens`).

- `llm.retry_cut_off_answers: true` — optional. If the model's answer comes back cut
  off (it hit `max_tokens` mid-JSON), retry with a bigger token budget instead of
  shipping the truncated answer. Off by default because the retry re-spends tokens on
  the same question; declare it on variables whose answers are long lists (treatment
  histories, event sequences).

## 3. STEPS

The passes, numbered, top to bottom. Two rules:

- Use `map_table_rows_and_prompt` when the contract says **every ingested row**
  must be processed. It reads every row from every matched structured source table
  and makes one bounded model call per row. Do not use relevance retrieval for this
  shape: retrieval is allowed to omit rows by design. Configure exact
  `tables: [...]` names plus optional `table_name_contains: [...]` fallbacks.
  `doc_types: [...]` can additionally honor a semantic document type assigned at
  ingestion even when the source filename is site-specific. Each matched row
  receives a stable `:TABLE` evidence id. No shipped recipe currently uses this
  step kind; the engine and its tests cover it.

- Use `llm_only` for a pass that reasons over **already-extracted values** rather than
  chart text. It runs no retrieval: the prompt is filled from the outputs of earlier steps
  in the same recipe (read via `steps.<id>.data`), and the reply is parsed as best-effort
  JSON — an unparseable reply records the parse error and returns no data rather than
  failing the run. Use it to derive a value from fields another pass produced: a stage
  group from TNM fields, one adjudicated list from three candidate lists. Because it never
  sees the chart, it cannot introduce a fact no earlier pass found.

- **Each partial pass declares only the keys it owns.** A pass that extracts a subset of
  the output (a date, a status, one specimen's receptors) shows the model **only those
  keys** as an inline JSON shape — it must **not** inject the full `{{ OUTPUT_SCHEMA }}`.
  Showing a partial pass the whole output contract invites it to invent fields it does not own.
- **Exactly one authoritative final merge produces the schema-validated output.** This is
  either:
  - a `python` finalize step (preferred — deterministic, no prompt, no LLM drift), which
    merges the partial passes and is the recipe's last step; **or**
  - a final LLM pass that assembles the whole output — and only **that** pass may inject
    `{{ OUTPUT_SCHEMA }}`.

  The full output is validated against `output_schema.json` in step 8, once, on the
  authoritative merge's result.

## 4. PROVENANCE CONTRACT

Every value must be traceable to the chart.

- **Every PHI-side value-bearing field carries a raw `evidence_chunk_id`** at its own
  object level (arrays carry it item-by-item). The finalize grounds a determination on the
  chunk the deciding pass actually read — the model's cited chunk when it is one the pass
  was shown, otherwise the top chunk that pass read — and **drops an ungrounded claim to
  null rather than ship it** (so a real value is never lost, and a value never floats free).
- **NO_PHI exports never carry the raw id** — the selection trace and exhaust emit an HMAC
  `evidence_surrogate_id`; the raw chunk id stays PHI-side in `result.json` (sensitivity
  medium). The finalize may stamp a PHI-side `source_file` for human review.
- A `null` or `"unknown"` value is a **non-answer placeholder**, not a claim — it needs no
  grounding. A **derived aggregate** (a count or a flag restating a grounded sibling
  collection) uses the `n_` / `derived_` prefix and is likewise exempt.

The step-8 provenance validator (`find_unprovenanced_value_paths`) enforces this; a
conformant recipe yields `unprovenanced_value_paths == []`.

## 5. OUTPUT SHAPE

`output_schema.json` is the **single** contract for the final output. One field per fact —
**no alias duplication** (do not emit the same value under two field names, and do not keep
a second hand-written copy of the output shape inside a prompt; the schema is the one
source of truth, shown only to the authoritative merge per section 3).

## 6. INVARIANT SYNC

The recipe's fields stay in sync with the cross-variable clinical invariants
(`_shared_validation_rules/clinical_invariants.py`, the `_VAR` map). A recipe that an
invariant reads must expose the field the invariant names; a dangling invariant target
raises loudly at load (the S5a-02 assertion), so the invariant set and the recipe set
can never silently drift apart.

---

## Skeleton

```yaml
name: <variable>
version: v1
collection: <basic|oncology>                 # == the top-level directory

# 1. INTENT --------------------------------------------------------------------
# What this extracts and why, in clinician language. Scope: what counts, what is
# excluded, how it relates to neighbouring recipes.

archetype: <point_fact|temporal_anchor|event_sequence|classification_adjudication|enumeration>

# 2. EVIDENCE POLICY -----------------------------------------------------------
roles:
  <role>: { source_file: <file>.csv }
output_schema: <variable>_v1_output_schema.json
llm:
  model: local_qwen          # temperature / response_format are engine defaults
  max_tokens: <n>
# evidence:                   # optional; omit unless you need a tighter context ceiling
#   max_context_tokens: <n>
reranking:
  source_priority: { <file>.csv: 1.0 }

depends_on: []               # 6. INVARIANT SYNC: upstream recipes whose fields this reads

# 3. STEPS ---------------------------------------------------------------------
steps:
  - id: <partial_pass>       # declares ONLY the keys it owns (inline JSON shape);
    kind: retrieve_and_prompt #   NEVER injects the full {{ OUTPUT_SCHEMA }}
    retrieval: { kind: hybrid, query: "...", k: 10, source_file: <file>.csv }
    prompt: prompts/<variable>_v1_<partial_pass>.md

  - id: <adjudicate>         # reasons over earlier steps' output; no chart search
    kind: llm_only           #   reads steps.<id>.data; cannot see the chart
    prompt: prompts/<variable>_v1_<adjudicate>.md

  - id: finalize             # the one authoritative merge -> schema-validated output;
    kind: python             #   grounds every value (4. PROVENANCE CONTRACT)
    module: <variable>_v1_python_helper.finalize
```
