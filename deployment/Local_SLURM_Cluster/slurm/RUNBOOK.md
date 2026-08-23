# the cluster corpus-build runbook (build on the cluster → consume on the laptop)

The plan: do the heavy GPU work — reading the charts and turning every text chunk into
an embedding (a numeric vector capturing its meaning) — on the compute cluster,
then copy the finished, searchable "corpus" back to the laptop where you actually
develop recipes and pull values out of the charts. "the cluster" is <YOUR_INSTITUTION>'s research
compute cluster; the laptop is your everyday machine.

Authoritative ordered steps. Supersedes the loose commands in `~/Desktop/junior_the cluster_readiness.md`.
Verified locally 2026-06-20 (the embedding model works the same on both machines, and a full
per-step seal → ingest → embed → index → extract round-trip ran with the cluster
`float16`/`safetensors` config). the cluster-side steps still require the <YOUR_INSTITUTION> VPN + a cluster account.

## Path facts (why the split works)
- `output_root` (cfg) binds `JR_DATA_ROOT`, so the cluster (`JR_OUT_ROOT`) and the laptop produce the
  **identical** folder subtree `CONTAINS_PHI/pipeline_run_receipts/<run_id>/patients/<pid>/`. Nothing
  has to be translated between the two machines.
- The two machines agree they are using the same embedding model by comparing **content hashes**
  (short values computed from the model's contents) — the weight-file hash (`model_sha256`) plus the
  tokenizer hash plus the dtype/pooling/normalize/max_tokens settings. The `model_id` PATH is
  deliberately NOT compared, so the model can sit at a different path on the cluster than on the laptop.
  The laptop `model.safetensors` already hashes `ea7388…`, matching the copy pinned on the cluster.

## One-time setup on the cluster
1. Deploy the repo to `JR_CODE_ROOT` (default `<YOUR_CODE_ROOT>`).
2. Build the Linux+CUDA conda env (NOT the laptop `.venv-arm64`): torch≥2.4 CUDA, transformers, polars,
   hnswlib, pydantic, `jsonschema[format-nongpl]`.
3. **Place the embedding model** at `embed_override.yaml`'s `model_id`
   (`<YOUR_SHARED_DIR>/models/BioClinical-ModernBERT-base`) so its `model.safetensors`
   hashes `ea7388682b12cb833491ca8a697e84e3e67122c65c9c658f4ad3355fdd891546` (== the laptop model →
   guarantees the two machines agree on the model). `HF_HOME` is HuggingFace's (the model hub/library)
   cache location; `HF_HUB_OFFLINE=1` forbids any download so the run stays offline.
4. `srun --pty -p dev --gres=gpu:1 bash; nvidia-smi -L` — grab an interactive GPU shell and confirm the GPU is visible.
5. Confirm the `<YOUR_BOX_REMOTE>:` rclone remote (`ml rclone`). (rclone is the file-copy tool that moves data to/from Box.)
6. `mkdir -p logs` in `JR_CODE_ROOT` (the `#SBATCH --output/--error` job-log paths are written relative to `logs/`).
7. **Decision (drives only paths):** reuse an existing project tree or give junior its own subdir → set
   `JR_CODE_ROOT/JR_OUT_ROOT/JR_SOURCE_ROOT/HF_HOME/CONDA_ENV` accordingly (all env-overridable).

## Per-run flow

### 1. Prep input
- Convert the <YOUR_EHR_EXPORT> export to per-patient folders:
  `deployment/<your_institution>/<YOUR_EHR_EXPORT>_download_to_junior_format/convert_<your_ehr_export>_download_to_patient_folders.py`.
- Push inputs to the cluster: `bash …/rclone_sync.sh pull` (Box→`LOCAL_RAW`) or `sftp <YOUR_TRANSFER_HOST>`.
- Build `patient_ids.txt` (one folder-name id per line; must match the converter's naming).
- **`JR_SOURCE_ROOT` must equal the rclone `LOCAL_RAW`** (default `<YOUR_CHART_SOURCE>`).

### 2. Build the corpus on the cluster (login node, then submit cluster jobs with sbatch)
<!-- The three stages run as SLURM "array" jobs: SLURM is the cluster's job scheduler,
     sbatch submits a job, and an array runs the same job once per patient (the `--array=1-$N`
     means tasks 1..N, and `%20` / `%10` caps how many run at the same time). -->

```bash
cd "$JR_CODE_ROOT"
export JR_RUN_ID=<batch_id>  JR_OUT_ROOT=<YOUR_OUTPUT_ROOT>  JR_SOURCE_ROOT=<YOUR_CHART_SOURCE>
SLURM=deployment/Local_SLURM_Cluster/slurm
PL=<YOUR_CODE_ROOT>/patient_ids.txt
N=$(wc -l < "$PL")

# (a) SEAL ONCE — "seal" snapshots the config so the run is reproducible and can't drift;
#     every array task requires it (omitting it fails task #1 of each stage):
CFG=$SLURM/ingest_override.yaml PATIENT_LIST=$PL bash $SLURM/seal.sh

# (b) ingest (read charts) -> embed (make embeddings, needs the GPU) -> index (build the
#     fast vector-search structure). Reuse the SAME JR_RUN_ID across all three so they
#     write into one corpus.
sbatch --export=ALL,CFG=$SLURM/ingest_override.yaml,PATIENT_LIST=$PL --array=1-$N%20 $SLURM/ingest.sh
sbatch --export=ALL,CFG=$SLURM/embed_override.yaml,PATIENT_LIST=$PL  --array=1-$N%10 $SLURM/embed.sh
sbatch --export=ALL,CFG=$SLURM/index_override.yaml,PATIENT_LIST=$PL  --array=1-$N%20 $SLURM/index.sh
# watch progress: squeue -u <YOUR_USERNAME> ; re-run only the patients that failed: --array="$(bash $SLURM/failed.sh <job_id>)"
```

### 3. Sync the corpus back
```bash
bash $SLURM/rclone_sync.sh push "$JR_RUN_ID"          # copies <JR_OUT_ROOT>/CONTAINS_PHI/pipeline_run_receipts/<RUN_ID>/ up to Box
```
On the laptop, pull that run so it lands at `data/CONTAINS_PHI/pipeline_run_receipts/<run_id>/` (Box Drive
or rclone). PHI: approved/encrypted storage only.

### 4. Develop recipes locally (laptop)
```bash
cd <YOUR_CODE_ROOT>
# (a) set run_id in the consume config:
#     deployment/local/extract_consume_override.yaml -> run_id: "<synced_run_id>"
# (b) GATE: confirm the laptop's embedding model matches the one that built the synced corpus,
#     checked on ONE patient BEFORE scaling up. This matters most for a "hybrid" recipe (one that
#     uses both keyword search and vector search); a recipe like date_of_birth that needs no vector
#     search never touches the embeddings, so a mismatch there would slip through unnoticed.
PYTHONPATH=src:. .venv-arm64/bin/python $PWD/deployment/Local_SLURM_Cluster/slurm/check_encoder_alignment.py \
  --config deployment/local/extract_consume_override.yaml --patient <pid>     # expect: alignment: OK
# (c) SEAL ONCE locally (snapshot the config; required), then EXTRACT — pull the values out of the
#     charts. The "variables" extracted are the recipes named in the cfg `recipes:` list.
JR_DATA_ROOT=$PWD/data POLARS_SKIP_CPU_CHECK=1 PYTHONPATH=src:. .venv-arm64/bin/python -m jr_pipeline \
  seal --config deployment/local/extract_consume_override.yaml
JR_DATA_ROOT=$PWD/data POLARS_SKIP_CPU_CHECK=1 PYTHONPATH=src:. .venv-arm64/bin/python -m jr_pipeline \
  extract --config deployment/local/extract_consume_override.yaml --patient <pid>
cat data/CONTAINS_PHI/pipeline_run_receipts/<run_id>/patients/<pid>/extract/<variable>/result.json
# iterate: edit var_extraction_recipes/…, RE-SEAL (the seal snapshots the recipes too), re-extract.
```

## Smoke gate before scaling
A "smoke test" is a quick end-to-end run to catch obvious breakage before committing to the full cohort.
Run steps 2–4 for ONE `Test_Patient` first, using a **hybrid** recipe — one that exercises both keyword
and vector search (e.g. `stage` / `date_of_diagnosis`). Green = `check_encoder_alignment.py` prints `OK`
(the laptop and corpus agree on the embedding model) and `extract` returns `ok=True`. Then scale to the cohort.

## Gotchas
- `seal` is mandatory before every per-step run (the cluster and laptop); editing a recipe needs a re-seal or a new run_id.
- `extract` has no `--recipes` flag (use the cfg `recipes:` list); `inspect` takes `--run-root`/`--step`, not `--variable`; standalone `retrieve` needs `query.text` and is for debugging only.
- The committed `deployment/local/llm_allowlist.yaml` points at the 0.5B model downloaded from the HuggingFace hub — use `llm_allowlist_local3b.yaml` for the fully-offline 3B model.
- Never `ls` in the repo (it crashes the libgit2 library); never add `--verbose` to rclone (filenames could leak hints about PHI).
