# Junior

Junior is an extraction tool to be used in HIPAA compliant environments. It should not be used with public APIs.

The download comes with a small local Qwen model for testing. The extraction prompts are the same as the published paper, but this model is not great for the tasks and the number of chunks retrieved is limited so it won't hang a laptop.

This pipeline is designed to work with frontier models. Use the deployment module to access your institution's specific API for true performance. For Stanford specific deployment for Carina, NeroGCP, and STARR please reach out to [jcdicker@stanford.edu](mailto:jcdicker@stanford.edu).

Junior is an extraction tool. It ensures internal consistency but does not check outputs. You do.

## Use Junior

### Install

Clone the repo, open Terminal, `cd` into it, and install:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[app]"
```

Check which Python you are using:

```bash
python3 -c "import sys, platform; print(sys.version.split()[0], platform.machine())"
```

You need Python 3.11 or newer. On Apple Silicon, it should report `arm64`.

The `/usr/bin/python3` that ships with macOS is 3.9. If an Apple Silicon Mac reports `x86_64`, that Python is running under Rosetta. PyTorch does not publish macOS x86_64 wheels, so the app may install while the `torch` extra needed for the encoder and local model does not.

### Run it

There are two ways in. They use the same pipeline underneath. The main advantage of the workbench is for recipe development and iterative testing.

**The shiny app** which has more features but is buggy on first release.

To use, Double-click **Junior Workbench** or run:

```bash
junior workbench
```

**The CLI** is for running the pipeline locally, on SLURM, or on a cloud VM once recipes are finalized. 

```bash
junior status     # current config, run, and progress

junior ingest     # 1. chart files -> parquet
junior embed      # 2. chunks -> vectors
junior index      # 3. build the vector index
junior extract    # 4. evidence -> answers
```

That is the normal workflow.

`ingest`, `embed`, and `index` build the patient corpus. Re-run `extract` when you change an extraction recipe.

By default, each command runs the whole cohort. Note! to choose what you want to embed, make sure you map the columns 
for your data. For example, if there is a csv with a column called note text, embed just that; other values such as dates
are maintained and can be queried but don't need to be embedded. 

### Start a new cohort

From the directory where you want the project to live:

```bash
cd /path/to/my_study

junior new-project
junior columns
junior ingest
junior embed
junior index
junior extract
```

`new-project` asks where the charts are and where output should go, then writes the project config and creates the needed folders.

Run `junior columns` before ingest. It maps the column names in your source export to the names Junior expects.

After that, the pipeline is the same for every cohort.

### Shipped Models

It ships with the embedding model used for the initial study and a local Qwen for testing on a laptop. To run local extraction, install the PyTorch extra:

```bash
.venv/bin/python -m pip install -e ".[app,torch]"
```

and place the required weights under `models/`.

The included example uses a local Qwen model registered as:

```text
qwen_3b_local
```

### Find the answers

The main outputs are here:

```text
data/
└── CONTAINS_PHI/
    └── answers/
        └── <run id>/
            ├── <variable>_results.xlsx
            ├── ...
            └── run_metadata.xlsx
```

There is one results workbook per recipe and normally one row per patient. Recipes that read tables can instead produce one row per record.

Each result includes the extracted value, the evidence passage, and where that passage came from.

Two columns answer different questions:

* `ok` — did the extraction machinery run successfully?
* `any_value_found` — did it actually find an answer?

The output columns come from the recipe's sealed schema, so the same recipe produces the same headers across cohorts.

`run_metadata.xlsx` stores run-level information such as the model, code hash, and recipe hashes.

Everything in the answer files is PHI.

## Orientation

### Runs

A run is one execution of a set of recipes against a cohort.

The first stage you run writes a reproducibility snapshot containing the code, recipes, and config, with hashes over them. Later stages check against that snapshot.

If the config changes in the middle of a run, Junior stops rather than silently accepting the change.

When a whole-cohort `extract` finishes, the run is summarized, validated, and verified.

Run outputs are kept separately by run ID, so repeating an extraction creates a new result set instead of overwriting the old one.

### Recipes

Variable-specific behavior lives in:

```text
var_extraction_recipes/
```

A recipe usually contains:

* a YAML file describing retrieval, evidence packing, LLM use, and validation;
* a prompt;
* an output schema;
* sometimes a small Python helper for variable-specific finalization.

Retrieval and reranking are part of the recipe. They are not separate stages you run yourself.

Shared validation code lives in:

```text
var_extraction_recipes/_shared_validation_rules/
```

The repository ships with eight extraction recipes:

```text
date_of_birth
date_of_death
date_of_diagnosis
stage
breast_receptors
second_opinion_or_not
genetics_germline
genetics_somatics
```

### Data and PHI

Patient-derived data belongs under:

```text
data/
├── CONTAINS_PHI/
│   ├── patient_data_input/
│   ├── answers/<run id>/
│   ├── pipeline_run_receipts/
│   └── expert_label_corrections/
└── NO_PHI__shareable/
```

The idea is that the shareable becomes the 'exhaust' from iterative recipe development and can be used for improving retrieval methods and evidence presentation

These rules are enforced in code and the security and pipeline tests are an attempt to make sure no PHI leaks. Again, use this entirely in a secure env

## License

PolyForm Noncommercial License 1.0.0.

Free to use, modify, and distribute for noncommercial purposes. For commercial licensing, contact [jcdicker@stanford.edu](mailto:jcdicker@stanford.edu).

See `LICENSE` for the full terms.
