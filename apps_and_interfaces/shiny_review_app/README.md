# Workbench

A local Shiny for Python app for running Junior over a cohort and reviewing what
it extracted. It is the CLI's browser counterpart: the Run button spawns
`junior run` itself, and every inspection panel reads the run's own artifacts.

## Screens

- **Start** — the operator's job end to end: point at a folder whose subfolders
  are patients and scan it; tick the variables to extract (one recipe each, read
  from `var_extraction_recipes/`); run the pipeline (ingest → embed → index →
  extract), or preflight / ingest-only first, which need no model. The run
  happens in a subprocess so the app stays responsive; its progress is tailed
  into the page. When it finishes, the app points Review at the run it just
  produced and switches to it.
- **Review** — the selected run's per-variable values, each beside the chart
  text its evidence chunk id resolves to; CSV export; and review capture.
- **Workbench** — read-only inspection of the selected run: the recipe that ran
  (from the run's sealed code bundle when it has one), the prepared evidence the
  model saw, the evidence-selection sizes, the exact prompts and responses per
  recipe step, the validation verdicts, the patient's run files, the NO_PHI
  exhaust inventory, and the shareable metadata export (the same zip
  `junior export-metadata` writes).
- **How it works** — what each pipeline stage does.

No chart data is sent to a public service by this app. Extracted values are
redacted from the run log the page shows.

## What it writes

Pipeline output lands under:

```text
data/CONTAINS_PHI/pipeline_run_receipts/<run_id>/
```

Reviewer feedback is treated as patient data and is written to:

```text
data/CONTAINS_PHI/expert_label_corrections/<run_id>/
```

The feedback types are:

- extraction correction;
- extraction confirmation;
- sampling-frame record;
- evidence chunk relevance (plus a de-identified `relevance_label` twin in the
  run's NO_PHI exhaust).

`junior eval-values --run-id <id> --variable <v>` scores a run against the
corrections and confirmations captured here; the sampling frame is the
denominator that makes those figures interpretable. Every writer refuses when
the view on screen is not a real run — a judgement that attaches to nothing must
not become expert ground truth.

## Run the app

The normal way in resolves your project and pins the app to its output tree:

```bash
junior workbench
```

Or directly, from the repo root:

```bash
python -m pip install -e '.[app]'   # if Shiny is not installed yet
python -m shiny run apps_and_interfaces/shiny_review_app/app.py \
  --port 8765 --launch-browser
```

The app loads the newest run on disk by default. To pin a run:

```bash
export JR_REVIEW_RUN_ID=<run_id>
export JR_REVIEW_PATIENT_ID=Test_Patient
```

Running the full pipeline needs `torch`, `transformers`, and the local models
under `models/`. Start reports anything missing before you press Run; preflight
and ingest-only still work without a model.

## Main files

- `app.py`: Shiny UI and reactive wiring.
- `pipeline_launcher.py`: recipe discovery, readiness checks, and supervision of
  the `junior run` subprocess.
- `cohort_ingest.py`: folder scan, preflight, and Step-1 ingest.
- `run_results.py`: loads run outputs and resolves evidence text.
- `run_inspection.py`: read-only run inspection — evidence, prompts, verdicts,
  exhaust, and the shareable export.
- `feedback_capture.py`: writes corrections, confirmations, sampling frames,
  and chunk-relevance labels.

Patient-derived run receipts and feedback belong under
`$JR_DATA_ROOT/CONTAINS_PHI/` or `./data/CONTAINS_PHI/` by default. They must
not be committed.
