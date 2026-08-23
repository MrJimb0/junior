#!/usr/bin/env bash
# ingest.sh — read one patient's raw chart files and write clean, typed tables.
#
# SLURM is the cluster's job scheduler; "sbatch" submits a job; an "array" runs
# this same script once per patient (one array task = one patient). The output
# is a typed parquet file per source (parquet = a compact columnar table format,
# like a strongly-typed CSV).
#
# Canonical submission (submit from the code root with logs/ present — the
# #SBATCH --output=logs/ paths below are relative to the submit directory):
#
#   cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --export=ALL,CFG=deployment/Local_SLURM_Cluster/slurm/ingest_override.yaml,PATIENT_LIST=<YOUR_CODE_ROOT>/patient_ids.txt \
#          --array=1-500%20 deployment/Local_SLURM_Cluster/slurm/ingest.sh
#
# Required env (usually set in the submitter's shell or a companion runs.env):
#   JR_OUT_ROOT     - <YOUR_OUTPUT_ROOT>
#   JR_RUN_ID       - this batch's run identifier
#   JR_SOURCE_ROOT  - <YOUR_CHART_SOURCE>

#SBATCH --job-name=ingest
#SBATCH --partition=dev
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-1
#SBATCH --output=logs/ingest_%A_%a.out
#SBATCH --error=logs/ingest_%A_%a.err

set -euo pipefail

CFG="${CFG:?must set CFG=deployment/Local_SLURM_Cluster/slurm/ingest_override.yaml}"
PATIENT_LIST="${PATIENT_LIST:?must set PATIENT_LIST=/path/to/patient_ids.txt}"
: "${JR_RUN_ID:?must export JR_RUN_ID for this batch (e.g. 20260429_modernbert_chunk512)}"
: "${JR_OUT_ROOT:?must export JR_OUT_ROOT (e.g. <YOUR_OUTPUT_ROOT>)}"

if [[ ! -r "$PATIENT_LIST" ]]; then
  echo "[ingest] FATAL: PATIENT_LIST not readable: $PATIENT_LIST" >&2
  exit 2
fi

# Read line N, treating blank/comment lines as a hard error so a poisoned
# PATIENT_LIST surfaces at task 1 rather than silently dropping patients.
PATIENT="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PATIENT_LIST")"
PATIENT="${PATIENT//$'\r'/}"  # strip stray CR (Windows-edited list)
PATIENT="$(printf '%s' "$PATIENT" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"  # trim surrounding whitespace (indented ids keep their value)
if [[ -z "${PATIENT:-}" || "${PATIENT:0:1}" == "#" ]]; then
  echo "[ingest] FATAL: no patient id at line ${SLURM_ARRAY_TASK_ID} of $PATIENT_LIST" >&2
  exit 2
fi

echo "[ingest] task=${SLURM_ARRAY_TASK_ID} patient=${PATIENT} host=$(hostname)"

module load anaconda3 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-<YOUR_CONDA_ENV>}"
# shellcheck disable=SC1091
source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

cd "${JR_CODE_ROOT:-<YOUR_CODE_ROOT>}"

python -m jr_pipeline ingest --config "$CFG" --patient "$PATIENT"
