#!/usr/bin/env bash
# failed.sh — list the patients that failed so you can re-run just those.
#
# On the cluster, one batch of patients runs as a SLURM "array" job (SLURM is the
# cluster's job scheduler; an "array" runs the same job once per patient, each task
# numbered). This script reads a finished job and prints the task numbers that did
# NOT succeed, as a comma-separated list you can hand straight back to sbatch (the
# command that submits a job) to re-run only those patients.
#
# Usage:
#     bash deployment/NeroGCP/slurm/failed.sh <job_id>
#     # prints something like: 7,42,91
#     # then:
#     sbatch --array="$(bash deployment/NeroGCP/slurm/failed.sh 12345)" --export=ALL,CFG=... deployment/NeroGCP/slurm/embed.sh

set -euo pipefail

JOB_ID="${1:?must pass the base SLURM job id (e.g. 12345)}"

# sacct reports each task as "JobID State". Keep the per-task "<job>_<N>" entries that
# failed (including ones that ran out of time or ran out of memory) and print their N.
sacct -X -n -j "$JOB_ID" -o "JobID%40,State" \
  | awk -v jid="$JOB_ID" '
      $1 ~ "^" jid "_" && ($2 == "FAILED" || $2 == "TIMEOUT" || $2 == "OUT_OF_MEMORY" || $2 == "CANCELLED") {
          split($1, parts, "_");
          print parts[2]
      }
  ' \
  | sort -n -u \
  | paste -sd, -
