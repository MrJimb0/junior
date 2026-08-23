# Two-Stream Provenance

Every run of `jr_pipeline` produces two logically separate provenance streams that sit side-by-side under the run output tree:

```
out/<run_id>/
├── code/          # CODE PROVENANCE — what produced this run
├── data/          # DATA PROVENANCE — what happened to each patient
├── logs/          # structured JSON logs
└── manifest.json  # minimal run descriptor (low sensitivity)
```

## Why two streams

They answer different questions for different audiences and have different sensitivity, retention, and storage profiles:

| Stream | Question it answers | Audience | Sensitivity | Retention | Storage target |
|---|---|---|---|---|---|
| `code/` | What code / config / recipe / environment produced this run? | Engineers, auditors, reviewers, downstream proprietary tooling | Low (zero PHI) | Indefinite | Anywhere — git, S3, local, code-provenance repo |
| `data/` | What happened to each patient? What did the model see? | Clinical reviewers, recipe developers, QA | Medium / High | 30d high / 1y medium | your secure Box tenancy only |

The split exists because conflating them is a real problem. Receipts that embed full recipe text duplicate prompt content across every patient-stage combination; engineers debugging a recipe change have to wade through HIPAA-scoped artifacts to compare code; and a downstream tool that wants to mine development history has to operate inside the PHI boundary unnecessarily.

## Cross-link: `code_lock_hash`

Every data-side artifact carries `produced_by.code_lock_hash` pointing into `code/code.lock.json`. To reconstruct "what code produced this receipt," look up the hash once and read the specific file in `code/recipes/` that ran.

Receipts are deliberately lighter than v2 drafts: they do NOT duplicate prompt template text. Instead, the stage receipt stores:

- `payload.prompt.template_name` — pointer into `code/recipes/prompts/`
- `payload.prompt.template_hash` — content hash of the template at run time
- `payload.prompt.context_vars` — interpolated values (e.g. `evidence_json`, upstream `vars.*` fields)
- `payload.messages_sent` — the authoritative, fully-rendered messages as sent to the LLM

Rendering logic can itself be buggy; `messages_sent` is the audit ground truth. The template pointer enables reconstruction from first principles when needed.

## Scoped sub-hashes

The code bundle also writes `code/hashes.json` — a structured index of finer hashes that the invalidation machinery consults:

- `code_lock_hash` — the whole code bundle (excludes bundle-metadata files that reference this hash)
- `config_hash` — resolved configuration
- `provider_config_hash` — LLM provider configuration subset (Gate 3+)
- `retrieval_config_hash` — chunker + retriever configuration
- `env_hash` — Python + key package versions
- `dependencies_hash` — `dependencies.lock`
- `per_recipe[<name>]` — recipe, schema, prompt, and python_helpers hashes

Every data-side artifact's `produced_by` carries the relevant sub-hashes. Invalidation is table-driven, not judgment-driven — see `src/jr_pipeline/invalidation.py` and v5 plan §6.

## Scrubbing invariant

`code/config_resolved.yaml` and `code/entry_point.json` are mechanically scanned at seal time (`src/jr_pipeline/provenance/scrub.py`). Violations — PHI paths, enumerated patient IDs — fail the seal. The code stream's "low sensitivity, shareable outside HIPAA" property is enforced, not assumed.

Operators who need to reference PHI paths in configs use symbolic placeholders (`$RAW_PATIENTS`, `$PATIENT_LIST`) that survive into `config_resolved.yaml` unexpanded, rather than enumerated values.

## Code-stream files

See [runtime_enforcing_safety_and_reproducibility/schemas/json/](schemas/) for the full envelope schema. Files in `code/`:

| File | Contents |
|---|---|
| `git.json` | sha, branch, tag, dirty flag, remote URL |
| `config_resolved.yaml` | self-contained config with `${VAR}` interpolation resolved, scrubbed |
| `recipes/` | snapshot of every recipe, prompt, schema, Python helper |
| `env.json` | Python + key package versions, platform |
| `dependencies.lock` | `pip freeze` output |
| `allowlist_names.json` | endpoint NAMES + attestations only — no URLs, no keys |
| `entry_point.json` | invocation that started the run, scrubbed |
| `hashes.json` | scoped sub-hash index (above) |
| `code.lock.json` | anchor document with `code_lock_hash` for external references |

## Verification

Any caller can recompute `code_lock_hash` from a sealed code directory and compare against the stored value:

```bash
jr-pipeline verify --run-root out/<run_id>/
```

`hashes.json` and `code.lock.json` are deliberately excluded from the hash they describe — updating them with the final hash value would otherwise invalidate that very hash. They're part of the bundle but not part of its substantive identity.
