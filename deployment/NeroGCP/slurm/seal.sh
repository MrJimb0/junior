#!/usr/bin/env bash
# seal.sh — run ONCE on the Nero LOGIN NODE before submitting any jobs (sbatch).
#
# "Sealing" records an exact snapshot of the pipeline code for this run, so every
# patient in the batch is processed by provably identical code (an audit/
# reproducibility guarantee). It writes a "sealed code bundle":
# <JR_OUT_ROOT>/CONTAINS_PHI/pipeline_run_receipts/<JR_RUN_ID>/code/code.lock.json
#
# A per-patient array task no longer refuses without that bundle: the first task to
# reach a run writes it, under a lock, so a fan-out that all starts at once cannot race
# (verified with concurrent tasks and no prior seal — one bundle, no leftover locks).
# Sealing here is still worth it for a different reason: a bad config, an unreadable
# recipe tree or a missing model fails ONCE on the login node, in front of you, instead
# of failing identically in every one of 500 array tasks twenty minutes later.
# --patients-file also records the real cohort in the manifest, which nothing else does.
# (The LOGIN NODE is the head node you ssh into, separate from the compute nodes
# that run the jobs.)
#
# Usage (from JR_CODE_ROOT, corpus-build env exported):
#   export JR_RUN_ID=<batch_id> JR_OUT_ROOT=<YOUR_OUTPUT_ROOT>
#   CFG=deployment/NeroGCP/slurm/ingest_override.yaml \
#   PATIENT_LIST=<YOUR_CODE_ROOT>/patient_ids.txt \
#   bash deployment/NeroGCP/slurm/seal.sh
#
# Any one of the corpus-build override files works as CFG here — at step time
# only the keys that stay the same for the whole run (run_id, output_root) are
# checked, and those match across the ingest/embed/index override files.

set -euo pipefail

CFG="${CFG:?must set CFG=deployment/NeroGCP/slurm/ingest_override.yaml}"
PATIENT_LIST="${PATIENT_LIST:?must set PATIENT_LIST=/path/to/patient_ids.txt}"
: "${JR_RUN_ID:?must export JR_RUN_ID for this batch}"
: "${JR_OUT_ROOT:?must export JR_OUT_ROOT (e.g. <YOUR_OUTPUT_ROOT>)}"

python -m jr_pipeline seal --config "$CFG" --patients-file "$PATIENT_LIST"

echo "[seal] sealed code bundle ready at:"
echo "       ${JR_OUT_ROOT}/CONTAINS_PHI/pipeline_run_receipts/${JR_RUN_ID}/code/"
echo "[seal] you may now sbatch ingest -> embed -> index for run ${JR_RUN_ID}."
