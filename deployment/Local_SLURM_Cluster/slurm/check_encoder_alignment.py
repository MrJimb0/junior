"""Safety check before running a whole cohort: confirm the laptop is set up to use the
SAME text-to-vector model the cluster already used to build the patient search index.

Background: the pipeline turns each chunk of a patient's chart into an "embedding" — a
numeric vector capturing that text's meaning — using a model
called the "encoder." Those vectors are stored on the cluster (the cluster) as a search index
(the "corpus"). When the laptop later searches that index to answer questions, it must turn
the QUERY into a vector with the exact same encoder, or the vectors live in different
"languages" and the matches are meaningless. This script catches that mismatch up front.

How it works: it computes a short "fingerprint" of the laptop's configured encoder — a value
derived purely from the model's files (content hashes; no model is loaded, so it is fast and
cheap) — and compares it against the fingerprint that was saved alongside the index when the
cluster built it. That saved fingerprint lives in a small companion metadata file (a
"sidecar") named chunk_index.parquet.meta.json. verify_corpus_encoder_alignment returns the
verdict. 'ok' => search will work; 'mismatch' => it prints which model setting(s) differ.
Note: model_id is just the file PATH to the model, which legitimately differs between the
cluster and the laptop, so it is shown for reference but EXCLUDED from the pass/fail verdict.

Run from the repo root, on ONE already-synced patient, BEFORE extracting a whole cohort:
  PYTHONPATH=src .venv-arm64/bin/python \\
    deployment/Local_SLURM_Cluster/slurm/check_encoder_alignment.py \\
    --config deployment/local/extract_consume_override.yaml --patient <pid>
"""
import argparse
import json
import sys
from pathlib import Path

from jr_pipeline.pipeline_steps.step_2_embed_chunks import build_encoder
from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
    VECTOR_AFFECTING_FINGERPRINT_FIELDS,
)
from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrievers.embedding.embedding_v1 import (
    verify_corpus_encoder_alignment,
)
from jr_pipeline.runtime_infrastructure.config_loading import load_config
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    data_root,
    phi_patient_run_dir,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="laptop consume config (run_id + encoder)")
parser.add_argument("--patient", required=True)
args = parser.parse_args()

cfg = load_config(args.config)
run_id = cfg["run_id"]
# Resolve the data root the SAME way every pipeline step does: an explicit output_root in
# the config, else JR_DATA_ROOT (default ./data) via data_root(). Otherwise, with
# JR_DATA_ROOT exported and no output_root key, this check would inspect the wrong tree.
resolved_data_root = Path(cfg.get("output_root") or data_root()).expanduser().resolve()
patient_root = phi_patient_run_dir(run_id, args.patient, resolved_data_root)
sidecar = patient_root / "chunk_index.parquet.meta.json"
if not sidecar.is_file():
    sys.exit(f"FAIL: no corpus sidecar at {sidecar}\n  (synced corpus missing for run={run_id} patient={args.patient})")

stored = json.loads(sidecar.read_text(encoding="utf-8"))["payload"]["encoder"]
query = build_encoder(cfg["encoder"]).fingerprint()

verdict = verify_corpus_encoder_alignment(query, stored, source_present=True)
print(f"\nencoder alignment: {verdict.upper()}\n")
# The vector-affecting fields are owned by encoder.py; iterate that list (minus model_id,
# the load path, which is printed separately) so a future fingerprint field is compared
# here automatically instead of showing MISMATCH with no differing row.
for key in (f for f in VECTOR_AFFECTING_FINGERPRINT_FIELDS if f != "model_id"):
    stored_val, query_val = stored.get(key), query.get(key)
    flag = "" if stored_val == query_val else "   <-- DIFFERS"
    print(f"  {key:14} corpus={stored_val!r}  consume={query_val!r}{flag}")
print(f"  {'model_id':14} corpus={stored.get('model_id')!r}")
print(f"  {'(excluded)':14} consume={query.get('model_id')!r}  (load path — not compared)")

if verdict == "ok":
    print("\nOK -> hybrid/embedding retrieval will work against this corpus. Safe to scale the cohort.")
    sys.exit(0)
sys.exit(f"\n{verdict.upper()} -> fix the DIFFERS field(s) in the consume config's encoder before the cohort.")
