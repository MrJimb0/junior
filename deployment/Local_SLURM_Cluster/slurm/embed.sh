#!/usr/bin/env bash
# embed.sh — turn one patient's chart text into numeric vectors for search.
#
# This stage splits the chart text into chunks and runs each
# through a language model to produce an "embedding" (a numeric vector capturing
# its meaning). Outputs: embeddings.npy (the vectors) + chunk_index.parquet (a
# table mapping each vector back to its source chunk). It needs a GPU, so it
# runs as a SLURM job array (one array task = one patient) on a GPU partition.
#
# Canonical submission (one array per chunk-size config; submit from the code
# root with logs/ present — the #SBATCH --output=logs/ paths below are relative to the
# submit directory):
#
#   cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --export=ALL,CFG=deployment/<env>/embed_override.yaml \
#          --array=1-500%10 deployment/Local_SLURM_Cluster/slurm/embed.sh
#
# Required env:
#   JR_OUT_ROOT     - <YOUR_OUTPUT_ROOT>
#   JR_RUN_ID       - this batch's run identifier
#
# Pre-flight: verify that NO data needs to leave the machine (no "egress") — i.e.
# this stage must NOT call the LLM gateway or the HuggingFace model hub (HF Hub) over
# the network. It uses only the ModernBERT model weights already cached on disk
# under $HF_HOME. (HuggingFace = the model hub & library this project loads from.)

#SBATCH --job-name=embed
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-1
#SBATCH --output=logs/embed_%A_%a.out
#SBATCH --error=logs/embed_%A_%a.err

set -euo pipefail

CFG="${CFG:?must set CFG=deployment/<env>/embed_override.yaml}"
PATIENT_LIST="${PATIENT_LIST:?must set PATIENT_LIST=/path/to/patient_ids.txt}"
: "${JR_RUN_ID:?must export JR_RUN_ID for this batch}"
: "${JR_OUT_ROOT:?must export JR_OUT_ROOT (e.g. <YOUR_OUTPUT_ROOT>)}"

if [[ ! -r "$PATIENT_LIST" ]]; then
  echo "[embed] FATAL: PATIENT_LIST not readable: $PATIENT_LIST" >&2
  exit 2
fi

PATIENT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PATIENT_LIST")"
PATIENT="${PATIENT//$'\r'/}"
PATIENT="$(printf '%s' "$PATIENT" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"  # trim surrounding whitespace (indented ids keep their value)
if [[ -z "${PATIENT:-}" || "${PATIENT:0:1}" == "#" ]]; then
  echo "[embed] FATAL: no patient id at line ${SLURM_ARRAY_TASK_ID} of $PATIENT_LIST" >&2
  exit 2
fi

echo "[embed] task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} host=$(hostname)"

module load anaconda3 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-<YOUR_CONDA_ENV>}"
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# Force HuggingFace fully offline (no network model downloads) and make the
# tokenizer behave predictably on compute nodes.
export HF_HOME="${HF_HOME:-<YOUR_MODEL_CACHE>}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# The embed override pins `device: cuda` (CUDA = NVIDIA's GPU compute interface),
# so the encoder fails loudly rather than silently falling back to the much slower
# CPU. This check verifies a GPU is actually visible and stops immediately if not
# (nvidia-smi is the NVIDIA GPU query tool) — so we fail here instead of waiting in
# the queue only to discover the task landed on a CPU-only node.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[embed] FATAL: nvidia-smi not on PATH; this task needs a GPU node." >&2
  exit 3
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[embed] FATAL: nvidia-smi -L failed; no GPU visible on $(hostname)." >&2
  nvidia-smi || true
  exit 3
fi
nvidia-smi -L | head -1

cd "${JR_CODE_ROOT:-<YOUR_CODE_ROOT>}"

python -m jr_pipeline embed --config "$CFG" --patient "$PATIENT"

# Post-task sanity check on the vectors just written. Catches silent corruption
# — missing/not-a-number values (NaN), wrong array shape, or vectors that aren't
# length-normalized — before this patient's outputs feed the index stage.
# The exit code propagates: a failed check fails the whole SLURM task, so a
# human notices it in "sacct" (the cluster's job-accounting/status report).
python -m jr_pipeline validate-embed --config "$CFG" --patient "$PATIENT"
