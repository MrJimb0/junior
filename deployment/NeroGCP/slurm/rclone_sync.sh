#!/usr/bin/env bash
# rclone_sync.sh — move patient data between your secure Box tenancy and Nero.
# (rclone is a file-copy tool; "remote" = a named cloud/Box location it can read/write.)
# Nero is <YOUR_INSTITUTION>'s secure compute cluster. "pull" brings raw charts down to
# the cluster before a run; "push" sends a finished run's outputs back to Box.
#
# Point this at your own Box remote before the first run:
#   Remote name: <YOUR_BOX_REMOTE>:
#   Inputs:      <YOUR_BOX_REMOTE>:<YOUR_BOX_FOLDER>/InputsToCluster/
#   Outputs:     <YOUR_BOX_REMOTE>:<YOUR_BOX_FOLDER>/OutputsFromCluster/<RUN_ID>/
#
# Usage:
#   bash deployment/NeroGCP/slurm/rclone_sync.sh pull
#   bash deployment/NeroGCP/slurm/rclone_sync.sh push <run_id>
#
# Environment overrides:
#   BOX_REMOTE    - default <YOUR_BOX_REMOTE>
#   BOX_INPUTS    - default <YOUR_BOX_FOLDER>/InputsToCluster
#   BOX_OUTPUTS   - default <YOUR_BOX_FOLDER>/OutputsFromCluster
#   LOCAL_RAW     - default <YOUR_CHART_SOURCE>
#   LOCAL_OUT     - default $JR_OUT_ROOT (the run's output root), else
#                   <YOUR_OUTPUT_ROOT> — so push reads from wherever
#                   the pipeline actually wrote this run
#
# Notes:
#   * rclone copy is safe to re-run (it skips unchanged files by size + mtime),
#     so an interrupted transfer can simply be started again.
#   * Never add --verbose in automation — filenames can leak hints about
#     protected health information (PHI).
#   * "ml rclone" loads rclone from the Nero software-module system; rclone is
#     not always on the command PATH otherwise.

set -euo pipefail

BOX_REMOTE="${BOX_REMOTE:-<YOUR_BOX_REMOTE>}"
BOX_INPUTS="${BOX_INPUTS:-<YOUR_BOX_FOLDER>/InputsToCluster}"
BOX_OUTPUTS="${BOX_OUTPUTS:-<YOUR_BOX_FOLDER>/OutputsFromCluster}"
LOCAL_RAW="${LOCAL_RAW:-<YOUR_CHART_SOURCE>}"
LOCAL_OUT="${LOCAL_OUT:-${JR_OUT_ROOT:-<YOUR_OUTPUT_ROOT>}}"

# Ensure rclone is on PATH (module-loaded on Nero).
if ! command -v rclone >/dev/null 2>&1; then
  ml rclone 2>/dev/null || module load rclone 2>/dev/null || true
fi
if ! command -v rclone >/dev/null 2>&1; then
  echo "[rclone] FATAL: rclone not available; run 'ml rclone' first." >&2
  exit 2
fi

case "${1:-}" in
  pull)
    echo "[rclone] pull ${BOX_REMOTE}:${BOX_INPUTS}/ -> ${LOCAL_RAW}"
    mkdir -p "${LOCAL_RAW}"
    rclone copy --progress "${BOX_REMOTE}:${BOX_INPUTS}/" "${LOCAL_RAW}/"
    ;;
  push)
    RUN_ID="${2:?push requires a run_id: bash deployment/NeroGCP/slurm/rclone_sync.sh push <run_id>}"
    SRC="${LOCAL_OUT}/CONTAINS_PHI/pipeline_run_receipts/${RUN_ID}"
    if [[ ! -d "$SRC" ]]; then
      echo "[rclone] FATAL: local run dir missing: $SRC" >&2
      exit 2
    fi
    echo "[rclone] push ${SRC}/ -> ${BOX_REMOTE}:${BOX_OUTPUTS}/${RUN_ID}/"
    rclone copy --progress "${SRC}/" "${BOX_REMOTE}:${BOX_OUTPUTS}/${RUN_ID}/"
    ;;
  *)
    echo "usage: $0 pull|push <run_id>" >&2
    exit 2
    ;;
esac
