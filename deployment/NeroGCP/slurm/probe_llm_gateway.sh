#!/usr/bin/env bash
# probe_llm_gateway.sh — one-off check that the cluster can reach the LLM gateway
# language-model service before a real run depends on it. ("Egress" means data
# leaving the machine over the network; this confirms that outbound call works.)
#
# Some outbound traffic is blocked from Nero (e.g. SSH on port 22); ordinary
# secure web traffic (HTTPS on port 443) usually works, but the exact the LLM gateway
# address and any institutional proxy MUST be confirmed from inside an interactive
# session on the "dev" partition before a scheduled cluster job relies on it.
#
# Usage:
#   LLM_GATEWAY_ENDPOINT=https://<host>/path bash deployment/NeroGCP/slurm/probe_llm_gateway.sh
#
# The script grabs an interactive shell (no GPU needed) on the dev partition, loads
# the conda environment, then makes a single header-only web request to the address.
# Exit 0 on HTTP 200/401/403 — meaning the service answered; even an auth-required
# response is fine, it proves we can reach it. Exit non-zero on connection refused,
# timeout, or a DNS (name-lookup) failure.

set -euo pipefail

ENDPOINT="${LLM_GATEWAY_ENDPOINT:?set LLM_GATEWAY_ENDPOINT=https://<host>/path}"

echo "[probe] endpoint=${ENDPOINT}"

srun --pty --partition=dev --time=00:10:00 bash -lc "
  module load anaconda3 2>/dev/null || true
  CONDA_ENV=<YOUR_CONDA_ENV>
  # shellcheck disable=SC1091
  source activate \"\$CONDA_ENV\" 2>/dev/null || conda activate \"\$CONDA_ENV\" 2>/dev/null || true
  echo '[probe] host:' \$(hostname)
  echo '[probe] testing HTTPS reachability to ${ENDPOINT}...'
  curl --max-time 10 -sS -I '${ENDPOINT}' | head -20 || {
    echo '[probe] FAIL: could not reach endpoint — open a ticket with your research computing group about outbound network access' >&2
    exit 3
  }
"
