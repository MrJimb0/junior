#!/usr/bin/env bash
# index.sh — build the fast search index over one patient's text embeddings.
#
# An "embedding" is a numeric vector capturing a chunk of chart text's meaning;
# this stage builds an HNSW index (hnsw.bin) over those vectors. HNSW (hnswlib)
# is an approximate-nearest-neighbor index that makes "find the most similar
# chunks" fast. One array task = one patient.
#
# Runs on the cluster's "dev" partition (a pool of compute nodes); CPU only, no
# GPU needed.
#
# Canonical submission (submit from the code root with logs/ present — the
# #SBATCH --output=logs/ paths below are relative to the submit directory):
#   cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --export=ALL,CFG=deployment/Local_SLURM_Cluster/slurm/index_override.yaml,PATIENT_LIST=<YOUR_CODE_ROOT>/patient_ids.txt \
#          --array=1-500%20 deployment/Local_SLURM_Cluster/slurm/index.sh

#SBATCH --job-name=index
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-1
#SBATCH --output=logs/index_%A_%a.out
#SBATCH --error=logs/index_%A_%a.err

set -euo pipefail

CFG="${CFG:?must set CFG=deployment/Local_SLURM_Cluster/slurm/index_override.yaml}"
PATIENT_LIST="${PATIENT_LIST:?must set PATIENT_LIST=/path/to/patient_ids.txt}"
: "${JR_RUN_ID:?must export JR_RUN_ID for this batch}"
: "${JR_OUT_ROOT:?must export JR_OUT_ROOT (e.g. <YOUR_OUTPUT_ROOT>)}"

if [[ ! -r "$PATIENT_LIST" ]]; then
  echo "[index] FATAL: PATIENT_LIST not readable: $PATIENT_LIST" >&2
  exit 2
fi

PATIENT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PATIENT_LIST")"
PATIENT="${PATIENT//$'\r'/}"
PATIENT="$(printf '%s' "$PATIENT" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"  # trim surrounding whitespace (indented ids keep their value)
if [[ -z "${PATIENT:-}" || "${PATIENT:0:1}" == "#" ]]; then
  echo "[index] FATAL: no patient id at line ${SLURM_ARRAY_TASK_ID} of $PATIENT_LIST" >&2
  exit 2
fi

echo "[index] task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} host=$(hostname)"

module load anaconda3 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-<YOUR_CONDA_ENV>}"
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

cd "${JR_CODE_ROOT:-<YOUR_CODE_ROOT>}"

python -m jr_pipeline index --config "$CFG" --patient "$PATIENT"
