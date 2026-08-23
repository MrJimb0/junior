# Local dev / smoke setup

Install, fetch the encoder, then run the full pipeline against a synthetic
patient. The LLM (a small Qwen) downloads itself on first use; the encoder
(BioClinical-ModernBERT) does not, because embed reads its `model_id` as a
folder on this machine and never reaches the network. No external server, no
allowlist URL config, no ollama.

## One-time setup

```bash
pip install -e ".[app,dev,torch]"

hf download NeuML/bioclinical-modernbert-base-embeddings \
  --local-dir models/embedding/NeuML_bioclinical-modernbert-base-embeddings
```

The encoder folder is ~150 MB. The first time you run `extract`,
Qwen2.5-0.5B-Instruct (~1 GB) downloads to `~/.cache/huggingface/`: the dev
allowlist opts into that with `allow_download: true`, which is the line a
production allowlist leaves out so a missing model folder fails loudly instead
of fetching something.

All three extras are here on purpose. `torch` is the encoder and the local LLM,
`app` is the workbench, and the test suite covers both — with either one absent
the tests that need it skip and say which extra they wanted.

## Where the synthetic patient lives

The repo ships a realistic synthetic chart at `examples/Test_Patient/`
with the standard 7 CSVs:

```
examples/Test_Patient/
  clinical_note.csv
  demographics.csv
  diagnoses.csv
  med_admin.csv
  med_orders.csv
  pathology_report.csv
  radiology_report.csv
```

## Run the full pipeline against the synthetic patient

Two ways in. The **interface path** (preferred for local dev): open
`quickstart.py` as a notebook, set your project in Section 1 (input
folder, patients, recipes), run cells top to bottom — no YAML, no env
vars; step defaults live in code.

The **CLI path** (same flow production/SLURM uses) needs one small
config file per environment — they live under `deployment/`, never in
`src/`:

```bash
export JR_SOURCE_ROOT=$PWD/examples                 # folder ABOVE the patient folder
export JR_OUT_ROOT=./out                            # data root for this run
# Digit-free dev id: a run id with a 5+-digit run (a timestamp) that is not the
# canonical YYYYMMDD_* shape is rejected by the NO_PHI exhaust scanner, and the
# run's exhaust records silently fail to write.
export JR_RUN_ID=dev_smoke

# Seal the code bundle (records git sha + config + recipe snapshot)
jr-pipeline seal --config deployment/Local_SLURM_Cluster/slurm/ingest_override.yaml

# Per-stage. --patient is the folder name under JR_SOURCE_ROOT.
jr-pipeline ingest  --config deployment/Local_SLURM_Cluster/slurm/ingest_override.yaml --patient Test_Patient
jr-pipeline embed   --config deployment/local/embed_override.yaml                          --patient Test_Patient
jr-pipeline index   --config deployment/local/embed_override.yaml                          --patient Test_Patient
jr-pipeline extract --config deployment/local/embed_override.yaml                          --patient Test_Patient

# Roll up (also compacts the run's NO_PHI exhaust into parquet + manifest) and validate
jr-pipeline summarize --run-root $JR_OUT_ROOT/CONTAINS_PHI/pipeline_run_receipts/$JR_RUN_ID
jr-pipeline validate  --run-root $JR_OUT_ROOT/CONTAINS_PHI/pipeline_run_receipts/$JR_RUN_ID
```

(`index` and `extract` reuse the embed config — one config across stages keeps the
run-invariant check happy. `embed_override.yaml` carries the extra keys `extract`
needs (`recipes`, `recipes_root`, `allowlist_path`), which the embed/index steps
ignore. The 0.5B smoke model named in `llm_allowlist.yaml` auto-downloads on first
use.)

## What's actually running

* **Encoder**: `NeuML/bioclinical-modernbert-base-embeddings` via
  `deployment/local/embed_override.yaml`. Real production encoder; ~150 MB.
* **LLM**: `Qwen/Qwen2.5-0.5B-Instruct` via the `local_hf` provider in
  `deployment/local/llm_allowlist.yaml`. Loaded in-process; no HTTP. Greedy
  decoding (`temperature=0`) for deterministic output and clean LLM
  cache hits.

Expect the 0.5B to fail the extraction rather than answer it. At this size it
copies instruction text out of the prompt into the chunk id field, the
provenance check sees a citation pointing at no passage it showed, and the
value is refused as unsupported. That is the guardrail doing its job on a model
too small for the task, and it is deterministic, so re-running reproduces it.
This path is here to exercise the plumbing end to end; for a value that comes
back populated, point the allowlist at one of the larger models below.

## Swapping models

Both are configurable in one place:

* Change the encoder → edit `deployment/local/embed_override.yaml` `model_id`.
* Change the LLM → edit `deployment/local/llm_allowlist.yaml`'s entry's `url`
  field (which carries the HF Hub ID for `local_hf` provider kind).

For better LLM quality at the cost of size/speed:
- `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB)
- `Qwen/Qwen3-1.7B-Instruct` (~3.5 GB)
- `microsoft/Phi-3.5-mini-instruct` (~7 GB)

## Production differs how

Production runs use:
* `deployment/Local_SLURM_Cluster/slurm/embed_override.yaml` — pinned
  local model path with SHA-256 verification (`expected_file_sha256`).
* An institutional allowlist file OUTSIDE this repo, pointing at the
  HIPAA-attested institutional gateway endpoint with `provider: openai_compat`.

The local_hf provider exists only for dev / smoke. The denylist still
fires for any URL-based provider, so a misconfigured production
allowlist still gets caught before any PHI moves.
