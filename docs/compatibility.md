# Compatibility Policy

`jr_pipeline` follows semver. Surfaces marked **stable** below are part of the public API — breaking changes require a major version bump and an ADR entry. Surfaces marked **internal** may change in any minor release without notice.

## Stable surfaces

- **Artifact schemas** — every file under `runtime_enforcing_safety_and_reproducibility/schemas/json/*.schema.json` and its associated parquet column schemas. A breaking schema change increments both the package major version AND the artifact's `schema_version` field.
- **Recipe DSL** — YAML layout, prompt front-matter, `depends_on`, cross-variable context interpolation.
- **CLI** — subcommand names, primary flags (`--config`, `--patient`, `--run-root`, `--force`). Addition of new flags or subcommands is non-breaking, as is giving an already-required flag a default: a caller that passes it explicitly is unaffected. `--config` and `--patient` are optional as of 0.0.1 — omitting `--config` discovers one, omitting `--patient` runs the whole cohort — and the SLURM scripts pass both, so their behavior is unchanged.
- **Protocol signatures** — `Encoder`, `Chunker`, `Retriever`, `AnnIndex`, `LLMProvider`, `StageHandler`, `OutputValidator`.
- **`code_lock_hash` semantics** — what the hash covers, which files are excluded from it, and the layout of `code/`.
- **Artifact envelope** — `schema_version`, `artifact_type`, `sensitivity`, `stream`, `produced_by` (including scoped sub-hashes), `parent_artifacts`, `content_hash`, `payload`.
- **Scrubbing invariant** — the set of patterns that trigger a scrub failure may only grow, never shrink.

## Internal surfaces

- Implementation details of built-in stage kinds, retrievers, providers.
- Internal registry structure (dict keys and dispatch shape).
- Log format beyond required fields (`ts`, `level`, `logger`, `message`). New optional fields may appear.
- Test fixtures and their layout.

## Deprecation policy

Removing a stable interface requires one minor-version deprecation cycle:

1. Minor release `X.Y`: interface is marked deprecated. Using it logs a warning. Behavior unchanged.
2. Minor release `X.(Y+1)` or later: interface may be removed in a major bump.

An ADR entry is required at both steps.

## Changing an artifact schema

Any change to `runtime_enforcing_safety_and_reproducibility/schemas/json/*.schema.json` (fields, enums, required fields) follows this process:

1. Write an ADR explaining the change, the migration path, and version bump.
2. Bump the artifact's `schema_version` integer.
3. Update readers to accept both old and new versions for at least one minor release.
4. Include a mapping/migration utility in `src/jr_pipeline/runtime_enforcing_safety_and_reproducibility/` if the change is not strictly additive.

Additive changes (new optional fields, new enum values in open-ended enums) may be made without a major bump, but still require a schema version increment and an ADR note.

## Artifact sensitivity classification

Artifact sensitivity (low / medium / high) is part of the stable surface. Downgrading an artifact's sensitivity — declaring something less private than it previously was — is a breaking change. Upgrading is not.
