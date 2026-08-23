# Deployment example — Nero (GCP)

A worked deployment for a secure GCP-backed research platform, with every site-specific
fact replaced by a placeholder. Copy the folder, rename it for your site, and fill it in.

This variant assumes **a managed gateway in front of cloud models**, which is the shape
most institutions use: the platform authenticates each call and holds the agreements.
Chart text leaves the machine, so every route in the allowlist is marked
`attestation: baa` — and if yours is not covered by one, do not mark it so.

## What is here

```
NeroGCP/
  llm_allowlist.yaml   which model services this deployment may send chart text to
  slurm/
    seal.sh              run ONCE on the login node before any job — snapshots the code
    ingest.sh            stage 1, one array task per patient
    embed.sh             stage 2, needs a GPU
    index.sh             stage 3
    *_override.yaml      the settings each stage runs under here
    check_encoder_alignment.py   the corpus and the encoder still agree
    probe_llm_gateway.sh one-off: can a compute node reach the model service
    rclone_sync.sh       move charts in and results out
    failed.sh            which array tasks failed, and why
    RUNBOOK.md           the order to run them in, and what to check between
```

## Before your first run

Every `<ANGLE_BRACKET>` is a placeholder. They are angle-bracketed rather than filled
with a plausible-looking value on purpose: a template with realistic defaults gets
copied and shipped, and the wrong path or the wrong endpoint is not something you want
to discover from a finished run. A placeholder left in fails immediately and says which.

1. `grep -rn '<YOUR_' .` — that is your whole edit list.
2. Set the paths in `slurm/*_override.yaml` and the `#SBATCH` lines in each `.sh`:
   partition names, time limits and GPU flags are site-specific. `sinfo` lists what
   this cluster actually has.
3. Edit `llm_allowlist.yaml`. The pipeline refuses any endpoint not named there, which
   is the point — it is the one file between a patient's notes and a service nobody
   signed an agreement with. Mark `attestation: baa` only where your institution
   genuinely has one.
4. Run `slurm/probe_llm_gateway.sh` from a compute node before submitting anything. A
   node that cannot reach the model service fails every task in the array, slowly.

## What these scripts already handle

They are a working deployment with the site facts removed, not a sketch. Worth knowing
before you simplify any of it away:

- **One array task per patient.** Patient isolation is the property the whole pipeline
  is built on; the array is how it survives a cluster.
- **HuggingFace is forced offline** during embed. The weights come from a cache on
  disk, so a compute node with no egress is a supported case rather than a failure.
- **The GPU is checked before the work starts**, so a task that landed on a CPU-only
  node fails in seconds rather than after an hour of silent CPU inference.
- **Embeddings are validated after they are written** — NaNs, wrong shape, vectors that
  are not normalised — so corruption fails the task that caused it.
- **The code is sealed once, on the login node.** Every run records the snapshot it ran,
  and nothing re-seals a run that already exists.

## Cloud-specific things to check

- **Egress is the whole risk.** `probe_llm_gateway.sh` tells you a node can reach the
  gateway; it cannot tell you the gateway is covered by an agreement. That is a question
  for whoever signed it.
- **Storage is not local.** `rclone_sync.sh` assumes a remote; point it at your bucket
  and confirm the service account can read charts and write results, before a run rather
  than during one.
- **Instances are billed while they idle.** The array's concurrency limit (`%10` in the
  submission lines) is what keeps a stalled stage from running up a bill.
