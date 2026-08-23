"""The site column mapping: one vocabulary, config supplies the names.

Chart metadata (author, date, doc type, specialty ...) reaches recipes under fixed
names, but every site's export spells its columns differently. These pin the seam —
what a site may override, how a miss is reported, and that the fields a recipe can
filter on are the same fields the chunk index carries.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import (
    _extract_row_metadata,
    _index_columns,
    _resolve_metadata_columns,
)
from jr_pipeline.pipeline_steps.step_5_rerank_chunks.filter_candidates import FILTERABLE_FIELDS
from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    DEFAULT_COLUMN_ALIASES,
    STANDARD_METADATA_FIELDS,
    ambiguous_columns,
    columns_from_config,
    data_columns_for,
    find_column,
    identifier_columns_for,
    resolve_column_aliases,
    text_columns_for,
)

# The note/report header the example EHR export ships.
EXPORT_NOTE_COLUMNS = [
    "patient_id", "date", "age", "type", "accession_number", "title",
    "author", "linked_author", "pat_enc_csn_id", "order_proc_id", "text",
]


def test_a_site_adds_names_without_losing_the_defaults():
    resolved = resolve_column_aliases({"author": ["AUTHOR_NM"]})

    assert resolved["author"][0] == "AUTHOR_NM"      # the site's name is tried first
    assert "author" in resolved["author"]            # the default still works
    assert resolved["title"] == ["title", "note_title", "document_title"]  # untouched


def test_configuring_an_unknown_field_is_refused():
    # A typo would otherwise sit in the config looking configured while nothing read it.
    with pytest.raises(ValueError, match="not a chart metadata field"):
        resolve_column_aliases({"autor": ["AUTHOR_NM"]})


def test_column_matching_ignores_case():
    # An export shipping uppercase headers used to read as "this file has no author",
    # silently, on every row.
    assert find_column(["author"], ["PATIENT_ID", "AUTHOR", "TEXT"]) == "AUTHOR"
    assert find_column(["note_date"], ["Note_Date"]) == "Note_Date"
    assert find_column(["author"], ["provider"]) is None


def test_priority_order_decides_between_two_present_columns():
    aliases = resolve_column_aliases({"document_date": ["result_date", "date"]})

    assert find_column(aliases["document_date"], ["date", "result_date"]) == "result_date"


def test_export_columns_resolve_without_any_config():
    # The bundled defaults already read this export's note header; a site only configures what
    # differs from them.
    resolved = _resolve_metadata_columns(EXPORT_NOTE_COLUMNS, resolve_column_aliases(None))

    assert resolved["document_date"] == "date"
    assert resolved["doc_type"] == "type"
    assert resolved["author"] == "author"
    assert resolved["linked_author"] == "linked_author"
    assert resolved["title"] == "title"
    assert resolved["specialty"] is None       # no specialty column on a note table


def test_a_site_mapping_reaches_the_extracted_row():
    aliases = resolve_column_aliases({"author": ["AUTHOR_NM"], "specialty": ["dept_specialty"]})
    columns = ["AUTHOR_NM", "dept_specialty", "date"]
    resolved = _resolve_metadata_columns(columns, aliases)

    values = _extract_row_metadata(
        {"AUTHOR_NM": "Smith, MD", "dept_specialty": "Medical Oncology", "date": "2024-01-02"},
        resolved,
    )

    assert values["author"] == "Smith, MD"
    assert values["specialty"] == "Medical Oncology"
    assert values["document_date"] == "2024-01-02"
    assert values["title"] is None             # absent column -> no value, not an error


def test_every_standard_field_is_filterable_and_indexed():
    # The three consumers read one definition, so a field cannot be written to the index
    # but be unfilterable (encounter_id used to be exactly that).
    assert FILTERABLE_FIELDS == STANDARD_METADATA_FIELDS
    for field_name in STANDARD_METADATA_FIELDS:
        assert field_name in _index_columns()


def test_an_institution_map_loads_from_its_deployment_folder():
    # Adding a site is adding a folder: a run names the file, nothing scans for it.
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    # The export spells the same concept differently per table, which is why maps are
    # written per document type rather than as one list.
    assert columns_from_config(cfg, "clinical_note.csv")["author"][0] == "author"
    assert columns_from_config(cfg, "labs.csv")["author"][0] == "authorizing_provider"
    assert columns_from_config(cfg, "procedures.csv")["author"][0] == "performing_provider"
    assert columns_from_config(cfg, "encounters.csv")["specialty"][0] == "dept_specialty"
    assert columns_from_config(cfg, "labs.csv")["document_date"][0] == "result_date"


def test_a_file_the_map_does_not_mention_falls_back_to_the_defaults():
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    aliases = columns_from_config(cfg, "some_new_export.csv")

    assert aliases["author"] == DEFAULT_COLUMN_ALIASES["author"]


def test_source_names_match_by_stem_and_ignore_case():
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    for name in ("labs.csv", "labs", "LABS.CSV", "Labs.parquet"):
        assert columns_from_config(cfg, name)["author"][0] == "authorizing_provider"


def test_inline_config_overrides_the_institution_map_for_one_file():
    cfg = {
        "chart_columns_file": "deployment/example_site/example_export_column_map.yaml",
        "chunk_metadata_columns": {"labs": {"author": "ODD_COHORT_AUTHOR"}},
    }

    assert columns_from_config(cfg, "labs.csv")["author"][0] == "ODD_COHORT_AUTHOR"
    # every other file keeps the institution's map
    assert columns_from_config(cfg, "procedures.csv")["author"][0] == "performing_provider"


def test_a_flat_map_is_refused_with_the_shape_it_expected():
    # The older field -> column form has no file, so it would silently map nothing.
    with pytest.raises(ValueError, match="keyed by source file"):
        columns_from_config({"chunk_metadata_columns": {"author": ["AUTHOR_NM"]}}, "labs")


def test_a_missing_map_file_fails_loudly():
    with pytest.raises(FileNotFoundError, match="chart column map not found"):
        columns_from_config({"chart_columns_file": "deployment/nowhere/no_such_map.yaml"})


# ---- things that broke under stress ----------------------------------------

def test_a_relative_map_path_resolves_from_outside_the_repo(monkeypatch, tmp_path):
    # A SLURM job or a cron run does not start in the checkout. A config naming
    # deployment/<site>/<your>_Column_Name_Map.yaml has to work from anywhere.
    monkeypatch.chdir(tmp_path)

    aliases = columns_from_config(
        {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}, "encounters.csv"
    )

    assert aliases["specialty"][0] == "dept_specialty"


def test_case_collision_in_one_file_is_deterministic_and_reported():
    # Only one of two same-named columns can ever be read. Which one must not vary
    # between runs, and the collision must not stay invisible.
    assert find_column(["author"], ["Author", "AUTHOR"]) == "Author"   # first in the file
    assert find_column(["author"], ["AUTHOR", "Author"]) == "AUTHOR"
    assert ambiguous_columns(["Author", "AUTHOR", "text"]) == {"author": ["Author", "AUTHOR"]}
    assert ambiguous_columns(["author", "text"]) == {}


def test_padded_column_names_still_match_and_keep_their_real_name():
    # Excel round-trips add the padding; the value still has to be readable by its
    # actual key.
    assert find_column(["author"], [" author "]) == " author "


def test_the_map_is_not_reparsed_on_every_call(tmp_path):
    # Preflight asks per file, and a cohort scan multiplies that by every patient.
    site_map = tmp_path / "columns.yaml"
    site_map.write_text("chunk_metadata_columns:\n  clinical_note:\n    author: FIRST\n")
    cfg = {"chart_columns_file": str(site_map)}
    assert columns_from_config(cfg, "clinical_note")["author"][0] == "FIRST"

    reads = {"n": 0}
    original = Path.read_text

    def counting_read_text(self, *a, **kw):
        if self == site_map:
            reads["n"] += 1
        return original(self, *a, **kw)

    with mock.patch.object(Path, "read_text", counting_read_text):
        for _ in range(50):
            columns_from_config(cfg, "clinical_note")
    assert reads["n"] == 0, "cached parse expected; re-reading the yaml per call"


def test_editing_the_map_between_runs_is_picked_up(tmp_path):
    # The cache is keyed by mtime, so a map edited in one session still takes effect.
    site_map = tmp_path / "columns.yaml"
    site_map.write_text("chunk_metadata_columns:\n  clinical_note:\n    author: BEFORE\n")
    cfg = {"chart_columns_file": str(site_map)}
    assert columns_from_config(cfg, "clinical_note")["author"][0] == "BEFORE"

    site_map.write_text("chunk_metadata_columns:\n  clinical_note:\n    author: AFTER\n")
    os.utime(site_map, (0, 0))  # force a distinct mtime
    assert columns_from_config(cfg, "clinical_note")["author"][0] == "AFTER"


# ---- the map accounts for whole files, not just the fields we happen to use ----

def test_a_map_entry_accounts_for_every_column_of_its_header():
    # The map is checkable against a real export: each column is a metadata field, the
    # embedded text, an identifier, or explicitly unmapped payload — never forgotten.
    import yaml

    example = yaml.safe_load(
        Path("deployment/example_site/example_export_column_map.yaml").read_text()
    )["chunk_metadata_columns"]
    # the header every note/report table ships
    header = ["patient_id", "patient_name", "mrn", "date", "age", "type",
              "accession_number", "title", "author", "linked_author",
              "pat_enc_csn_id", "order_proc_id", "text"]

    entry = example["clinical_note"]
    claimed = []
    for value in entry.values():
        if isinstance(value, dict):          # data_columns: {our name: their column}
            claimed += list(value.values())
        elif isinstance(value, list):        # text_columns / identifiers
            claimed += value
        else:
            claimed.append(value)

    assert sorted(claimed) == sorted(header)


def test_text_columns_and_identifiers_are_roles_not_metadata_fields():
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    assert text_columns_for(cfg, "clinical_note.csv") == ["text"]
    assert text_columns_for(cfg, "clinical_note_meta.csv") == []   # no text to embed
    assert "patient_name" in identifier_columns_for(cfg, "clinical_note.csv")
    # and they never leak into the field lookup
    aliases = columns_from_config(cfg, "clinical_note.csv")
    assert set(aliases) == set(STANDARD_METADATA_FIELDS)


def test_age_is_mapped_and_filterable():
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    assert columns_from_config(cfg, "clinical_note.csv")["age"][0] == "age"
    assert columns_from_config(cfg, "med_admin.csv")["age"][0] == "taken_date_age"
    assert "age" in FILTERABLE_FIELDS


# ---- drafting a map for a site that has never used Junior --------------------

def test_the_drafter_finds_the_free_text_column_a_site_named_differently(tmp_path):
    # The failure a new site hits first: their notes column is not called "text". The
    # drafter has to surface it, since nothing else will until embed refuses to run.
    from jr_pipeline.runtime_infrastructure.column_map_drafting import render

    (tmp_path / "progress_notes.csv").write_text(
        "MRN,PatientName,NOTE_DATE,NOTE_TYPE,SERVICE_LINE,ATTENDING,NOTE_TITLE,NOTE_TEXT\n"
    )

    drafted = render(tmp_path)

    assert "text_columns: [NOTE_TEXT]" in drafted
    assert "document_date: NOTE_DATE" in drafted
    assert "title: NOTE_TITLE" in drafted
    assert "identifiers: [MRN, PatientName]" in drafted
    # what it could not place is named rather than dropped, so the human can rename it
    assert "data_columns:" in drafted
    assert "SERVICE_LINE: SERVICE_LINE" in drafted
    assert "ATTENDING:    ATTENDING" in drafted
    # and it says which fields still need a decision
    assert "not found:" in drafted


def test_a_drafted_map_is_valid_input_to_the_loader(tmp_path):
    # The whole point: edit what it prints, save it, and it loads.
    import yaml

    from jr_pipeline.runtime_infrastructure.column_map_drafting import render

    (tmp_path / "progress_notes.csv").write_text("MRN,NOTE_DATE,NOTE_TEXT\n")
    site_map = tmp_path / "site.yaml"
    site_map.write_text(render(tmp_path))

    cfg = {"chart_columns_file": str(site_map)}
    assert text_columns_for(cfg, "progress_notes") == ["NOTE_TEXT"]
    assert columns_from_config(cfg, "progress_notes")["document_date"][0] == "NOTE_DATE"
    assert yaml.safe_load(site_map.read_text())["chunk_metadata_columns"]


def test_the_cli_offers_the_drafter_where_a_newcomer_looks():
    # A CLI-only user meets the column map on purpose, not via a failed run.
    from apps_and_interfaces.command_line_interface import START_COMMANDS, main

    assert "columns" in START_COMMANDS
    assert "columns" in main.commands
    assert not main.commands["columns"].hidden


def test_data_columns_let_a_recipe_use_our_name_for_a_site_column():
    # A recipe may ask for `race_detail`. This export has no such column — it calls
    # it `ethnic_background`. The map is what closes that gap, and what lets the
    # same recipe run somewhere that calls it something else again.
    cfg = {"chart_columns_file": "deployment/example_site/example_export_column_map.yaml"}

    demographics = data_columns_for(cfg, "demographics")

    assert demographics["race_detail"] == "ethnic_background"
    assert demographics["race"] == "race"
    assert data_columns_for(cfg, "labs")["result_value"] == "value"


def test_data_columns_must_be_named_not_listed():
    # A bare list would record that a column exists while saying nothing about what it
    # is, which is the thing this replaced.
    with pytest.raises(ValueError, match="maps our name to this site's column"):
        data_columns_for(
            {"chunk_metadata_columns": {"labs": {"data_columns": ["value", "units"]}}},
            "labs",
        )


# ---- headers as messy as real exports ----------------------------------------

def test_a_drafted_map_parses_even_from_an_ugly_header(tmp_path):
    # Real exports ship colons, hashes, spaces and YAML keywords in headers. A draft
    # that does not parse defeats the whole point of drafting one.
    import yaml

    from jr_pipeline.runtime_infrastructure.column_map_drafting import render

    (tmp_path / "notes.csv").write_text(
        'MRN,Note Date,Result: Final,value #1,"quoted col",Ünïcode,yes,null,text\n'
    )

    drafted = render(tmp_path)
    entry = yaml.safe_load(drafted)["chunk_metadata_columns"]["notes"]

    # every column survives the round trip under its real name
    named = list(entry.get("data_columns", {})) + entry.get("text_columns", []) \
        + entry.get("identifiers", []) + [entry.get("document_date")]
    for column in ("Result: Final", "value #1", "quoted col", "Ünïcode", "yes", "null"):
        assert column in named, f"{column!r} lost in the draft"


def test_a_space_separated_header_matches_an_underscored_alias():
    # "Note Date" is note_date. Without this it matched nothing, and the drafter then
    # offered the DATE column as free text to embed because "note" looked texty.
    assert find_column(["note_date"], ["Note Date"]) == "Note Date"
    assert find_column(["document_date"], ["Document  Date"]) is None   # double space is a different name
    assert find_column(["author"], ["author"]) == "author"


def test_two_of_our_names_may_point_at_one_column_without_breaking_the_read():
    # A site can legitimately map race_detail and ethnicity to the same column; asking
    # polars for it twice is an error, so the reader keeps each column once.
    resolved = {"race": "race", "race_detail": "race", "ethnicity": "ethnicity"}
    available = ["race", "ethnicity"]

    keep = list(dict.fromkeys(c for c in resolved.values() if c in available))

    assert keep == ["race", "ethnicity"]
