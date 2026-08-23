# Apps and Interfaces

This directory holds human-facing ways to run or inspect Junior. The pipeline
itself lives under `src/jr_pipeline/`; these files are wrappers around that
core code.

## Entry Points

### `quickstart.py`

Location: repo root.

This is the smallest runnable example. It runs the full local pipeline on the
synthetic `Test_Patient` fixture when local model weights are available. Use it
to understand the flow, not as the place to configure a real cohort.

For a real cohort, create a project file:

```bash
cd /path/to/my_study
junior new-project --input /path/to/patient_folders
junior ingest
```

### `shiny_review_app/`

Single-page Shiny app for local chart Q&A, extracted-value review, and feedback
capture. It can load a real Junior run from disk, or fall back to the synthetic
demo data so the app remains runnable.

Review feedback is written under:

```text
data/CONTAINS_PHI/expert_label_corrections/<run_id>/
```

## PHI Rule

Apps may display or collect patient-derived content. Anything generated from a
real run must stay under `data/CONTAINS_PHI/` unless it has passed the explicit
shareable-export path.
