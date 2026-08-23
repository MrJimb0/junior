# PHI Governance

`jr_pipeline` handles patient health information. This document enumerates the architectural controls that keep PHI where it belongs and makes misconfiguration hard.

## Architectural principle: only approved infrastructure

The pipeline refuses to connect to public consumer LLM APIs. That refusal is **hardcoded in core** and cannot be removed by configuration — removing the denylist is a code change that must pass review. See `src/jr_pipeline/pipeline_steps/step_7_extract_variables/providers/` (notably `llm_endpoint_denylist.py` and `llm_endpoint_allowlist.py`).

Approved deployment options:

- **Institutionally-hosted BAA endpoints** (e.g., an institution's API-gateway LLM service, Azure OpenAI under institutional BAA, Vertex AI in your institution's GCP project). Each is listed in a private allowlist file outside this repo with `attestation: baa`.
- **Self-hosted local models** running inside the institutional security perimeter (Ollama, vLLM, institution-fine-tuned models). Listed in the same allowlist with `attestation: self_hosted`.

The pipeline has no path for "scrub PHI and send to a public API." This is not a missing feature; it's a deliberate architectural refusal.

## Artifact sensitivity classification

| Stream | Sensitivity | Artifacts | Retention policy (enforced manually — see Retention) |
|---|---|---|---|
| `code/` | Low | git.json, config_resolved.yaml *(scrubbed)*, recipes/, env.json, deps.lock, allowlist_names.json *(names only)*, entry_point.json *(scrubbed)*, hashes.json, code.lock.json | Indefinite |
| `data/` | **High** | stage receipts (evidence, rendered prompts, response_raw) | 30 days after analysis complete |
| `data/` | **Medium** | state.jsonl, source_snapshot.json, structured/\*.parquet, chunk_index.parquet, embeddings.npy, hnsw.bin, cohort.parquet/.xlsx, result.json, invariants.json | 1 year |
| `data/` | Low | health.json *(scrubbed aggregate)*, manifest.json | Indefinite |
| `logs/` | Low | structured JSON (PHI-scrubbed at log site) | 1 year |

Note: `state.jsonl` and `source_snapshot.json` are medium, not low — their entity keys include `patient_id` and their presence reveals cohort membership.

## The PHI boundary on disk: `CONTAINS_PHI/` vs `NO_PHI__shareable/` (S4-10)

Under the data root (`JR_DATA_ROOT`, default `./data`) two top-level directories
make the classification a filesystem fact, not a convention:

- `CONTAINS_PHI/` — everything per-patient: `pipeline_run_receipts/<run>/patients/<pid>/`
  (structured parquet, embeddings, indexes, evidence, `result.json`, invariants),
  plus the **raw patient roster** at `pipeline_run_receipts/<run>/run_roster.json`
  and expert review under `expert_label_corrections/<run>/`.
- `NO_PHI__shareable/<run>/` — only run-level metadata: sealed code, run config,
  aggregate metrics, and the **exhaust manifest** at `manifest.json`.

The **NO_PHI exhaust manifest** is the shareable inventory of a run: record-type
tables (`exhaust/<record_type>.parquet`), file hashes, schema versions, and a
secret fingerprint. It deliberately carries **no patient roster** — the roster is
split out PHI-side into `run_roster.json`, so the shareable manifest can never
re-identify the cohort. `export_run_metadata` (`evaluating_pipeline_performance/export_shareable_metadata.py`)
scans the entire NO_PHI tree before bundling and raises on any raw id, clinical
date, or free text; a clean export is the containment proof. The layout helpers
that own these paths live in
`src/jr_pipeline/runtime_infrastructure/data_directory_layout_and_safe_writes.py`
(`no_phi_manifest_path`, `phi_intermediate_run_dir`, `clinician_feedback_dir`).

## Scrubbing invariant

Two files in `code/` are mechanically scrubbed before the code bundle is sealed:

- `config_resolved.yaml` — the run's self-contained YAML config with `${VAR}` interpolation resolved
- `entry_point.json` — invocation shape (argv, SLURM params)

Prohibited in either:

- Patient IDs matching known institutional conventions (`gs\d+`, `MRN\d+`, `ID\d+`, `PT\d+`)
- Paths containing `raw/patients/` or `box[_\- ]?medicine`
- Cluster raw-data path shapes — per-lab project trees ending in `raw/` (see `phi_leak_prevention_checks.py` for the exact patterns)

Operators use symbolic placeholders (`$RAW_PATIENTS`, `$PATIENT_LIST`) that survive into the sealed config unexpanded.

Scrub failures are hard errors. The seal step refuses to complete. Fixing is always cheap — replace the offending value with a symbolic placeholder.

## Retention and sharing

- **Retention is manual.** No command deletes run data; the table above is
  the policy, and the operator enforces it by deleting run trees per the
  compliance checklist's retention decisions. The code stream is never
  purged.
- **The shareable bundle is `jr-pipeline export-metadata --run-id <id>
  --output <zip>`.** It zips the run's `NO_PHI__shareable/` tree after
  scanning every file for forbidden content (one finding aborts). Nothing
  is redacted at export time, because nothing PHI ever enters that tree —
  the two-directory boundary above, plus the exhaust write gate, does the
  containment up front instead of scrubbing on the way out.
- There is no artifact read-audit log; access control is the filesystem's.

## Secrets

- Development: env var or `.env` file (gitignored). `.env.example` committed.
- Production on GCP: GCP Secret Manager. Service account scoped to the specific secret.
- Production on a shared cluster (interim, until a secret manager is available there): user's `.bashrc` with `chmod 600`.

Secrets are never committed to the repo, baked into container images, included in logs, or passed via SLURM `--export`.

## Logging and stdout rules

- Pipeline events go through the structured JSON logger
  (`src/jr_pipeline/runtime_infrastructure/json_event_logging.py`).
- No patient text, note content, evidence, or identifiers in log messages. Log structured fields instead (`run_id`, `patient_id`, `error_class`, `chunk_idx`).
- Operator-facing progress lines (cohort runner, CLI) print deliberately —
  and extracted values are redacted from stdout by default
  (`_stdout_values_allowed` in `cohort_runner.py`: `JR_REDACT_STDOUT`
  always wins; values appear only when `JR_SHOW_STDOUT_VALUES=1`). The
  per-step `extract` command does not redact; its stdout is debug output,
  never captured into shared logs.
