# Deployment

One folder per place Junior runs. `local/` is a laptop; `Local_SLURM_Cluster/` and
`NeroGCP/` are worked examples for an on-prem SLURM cluster and a cloud SLURM
deployment. A new site is a new folder — no pipeline code changes.

## Adding An Institution

```text
deployment/<institution>/
    chart_columns.yaml     # which of your columns holds each piece of chart metadata
    llm_allowlist.yaml     # the model endpoints your institution approves
    <stage>_override.yaml  # cluster/queue/model-path settings for your hardware
```

Copy `Local_SLURM_Cluster/` as a starting point. Nothing scans this directory: a run
names the files it uses, so an institution's folder is only ever read on purpose.

## Chart Column Maps

Junior reads the same small set of metadata off every chunk — `document_date`,
`doc_type`, `encounter_id`, `author`, `linked_author`, `title`, `specialty` — and
recipes filter evidence by those names. Your export names its columns whatever it
names them, so each institution maps them once.

`example_site/` holds the worked example map — the shape of the bundled
`examples/Test_Patient` chart.

Maps are written **per document type**, because one name does not fit every table —
a note's author is `author`, a lab's is `authorizing_provider`:

```yaml
# deployment/<institution>/<your>_Column_Name_Map.yaml
chunk_metadata_columns:
  clinical_note:                 # the source file's stem, matched case-insensitively
    document_date: note_dt
    author: AUTHOR_NM
    title: NOTE_TITLE
  labs:
    document_date: result_date
    author: authorizing_provider
  encounters:
    specialty: dept_specialty
```

A field the table genuinely has no column for is left out rather than guessed; a file
you don't list falls back to Junior's generic defaults. A value may be a list when the
same table spells a field more than one way across cohorts.

### Starting From Your Own Export

Don't write the map by hand. Point this at one patient's folder and it reads your
headers — header lines only, never a row of data — and prints a map to edit:

```bash
junior columns /path/to/cohort/PATIENT_1 \
    > deployment/<institution>/<institution>_Column_Name_Map.yaml
```

It guesses what it recognizes, names everything else under `data_columns` (so the
whole header is in front of you and nothing is dropped), and marks the two lines
worth checking hardest: the document's date, and which column holds the free text.

`data_columns` is `{our name: your column}` — the table's own data. Junior doesn't
filter on these, but a recipe reading a table directly (`kind: direct_parquet`) asks
for our name and gets your column, which is what lets one recipe run at two sites.

Then preflight your cohort. It fails before a run starts if no file has free text
(nothing could be searched), and gives a heads-up per file for any metadata field it
could not find a column for. Fix the map, preflight again, and run.

Point a run at it from that run's stage config:

```yaml
chart_columns_file: deployment/<institution>/<your>_Column_Name_Map.yaml
```

Three layers resolve, each overriding the one below: `chunk_metadata_columns:`
written inline in the config (same per-file shape), then the file above, then
Junior's generic defaults. So you can adopt your institution's map and still
override one file for an odd cohort.

Names are matched case-insensitively, and a field you don't map falls back to the
generic defaults rather than being dropped. Ingest resolves the mapping against
each real file, records the answer in that file's sidecar, and preflight reports
any field it could not find a column for — before a run starts, per file. A field
with no column is not an error: most tables carry only some metadata. It matters
because a recipe filtering on that field will drop all of that file's evidence,
which is why you see it up front rather than as an empty result later.

The standard field names themselves are fixed in
`src/jr_pipeline/runtime_infrastructure/chart_metadata_fields.py`. Adding a new
one is a code change there, deliberately — it lets a recipe's filters be checked
when the recipe loads, before any run starts.
