"""The chart metadata Junior understands, and how to find it in a site's columns.

Every chunk carries the same small set of metadata — who wrote the document, when, what
kind it is — and a recipe filters evidence by those names (``filters: [{field: author,
op: contains, ...}]``). The names are fixed here so a recipe means the same thing
everywhere; the columns they live in are not, so the mapping is configuration.

Three consumers share this one definition:

  * step 2 (embed) writes one chunk_index column per field, filling it from whichever
    source column the mapping resolves to;
  * step 5 lets a recipe filter on these fields;
  * step 7 checks a recipe's filters at load time, before any run starts.

The aliases below are generic defaults. Each institution keeps its own map beside its
other deployment settings — ``deployment/<institution>/<your>_Column_Name_Map.yaml`` — and a run
picks one with ``chart_columns_file:`` in its stage config. Adding a site is adding a
folder; nothing here changes. ``deployment/example_site/example_export_column_map.yaml`` is the worked
example (an Epic-style research export).

Adding a whole new FIELD is a code change here rather than config, so a recipe's filters
can still be validated at load time, before a run starts, against a known vocabulary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# What each chunk metadata field HOLDS, in the order the fields appear in the chunk index.
# The kind is what decides how a recipe may compare the field: a date as a range, a number
# numerically, text as text. Declaring it here rather than special-casing a field name
# downstream is what lets the next ordered field — another date, a measurement — get range
# filtering the day it is added, with no change to the filters that read it.
#
#   date    ISO-8601, possibly partial (``2020-XX-XX``)
#   number  a quantity, compared numerically ("9" is not greater than "10")
#   text    everything else, compared as text. An identifier is text even when it looks
#           like a number: no clinical question orders encounters by their id.
METADATA_FIELD_KINDS: dict[str, str] = {
    "document_date": "date",
    "age": "number",
    "doc_type": "text",
    "encounter_id": "text",
    "author": "text",
    "linked_author": "text",
    "title": "text",
    "specialty": "text",
}

# The kinds an ordered comparison (>, >=, <, <=) means anything for. Text is excluded on
# purpose: "author greater than Smith" is not a question a chart answers.
ORDERED_METADATA_KINDS = ("date", "number")

# The fields a chunk carries and a recipe may filter on. Adding one to
# METADATA_FIELD_KINDS above makes it flow through embed, the index, and recipe
# validation; nothing else needs to change.
STANDARD_METADATA_FIELDS: tuple[str, ...] = tuple(METADATA_FIELD_KINDS)

# Keys in a per-file entry that name a ROLE rather than a metadata field. They carry
# lists of columns, and each has a consumer:
#   text_columns  the free text embedded from this file. Named here so a site's odd
#                 column ("note_text", "report") is embedded without a second config;
#                 without it, embed falls back to its configured text_column ("text").
#   identifiers   direct identifiers (patient name, MRN, accession). Embed REFUSES to
#                 embed one, so a map cannot put a name into every chunk by accident.
#   data_columns  the table's own payload, named: {our name: their column}. A lab
#                 value, a vital sign, an ICD code. Junior does not filter on these —
#                 they are not chunk metadata — but a recipe reading a table directly
#                 (``kind: direct_parquet``) asks for OUR name and gets this site's
#                 column, which is what lets one recipe run at two institutions.
TEXT_COLUMNS_KEY = "text_columns"
IDENTIFIERS_KEY = "identifiers"
DATA_COLUMNS_KEY = "data_columns"
ROLE_KEYS = (TEXT_COLUMNS_KEY, IDENTIFIERS_KEY, DATA_COLUMNS_KEY)

# Source column names to look for, per field, in priority order. Kept generic — a site's
# own names belong in that site's config, not here.
DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "document_date": ["date", "note_date", "document_date", "encounter_date", "order_time", "result_time"],
    "age": ["age", "age_at_event", "patient_age"],
    "doc_type": ["type", "doc_type", "document_type", "note_type", "order_type"],
    "encounter_id": ["encounter_id", "csn", "accession_number"],
    "author": ["author", "provider_name", "note_author", "ordering_provider"],
    "linked_author": ["linked_author", "cosigner", "co_signer"],
    "title": ["title", "note_title", "document_title"],
    "specialty": ["specialty", "department", "service"],
}


def normalize_source_name(name: str) -> str:
    """A source file's key in a column map: its stem, lowercased. ``clinical_note.csv``,
    ``Clinical_Note`` and ``clinical_note`` are the same document type."""
    return Path(name).stem.strip().lower()


def resolve_column_aliases(configured: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """The alias list to search per field: the site's names first, then the defaults.

    A site adds names rather than replacing them, so a config that fixes one field does
    not silently break the others. An unknown field name in the config is an error — it
    would otherwise sit there looking configured while nothing ever read it.
    """
    resolved = {field: list(aliases) for field, aliases in DEFAULT_COLUMN_ALIASES.items()}
    for field, aliases in (configured or {}).items():
        if field in ROLE_KEYS:
            continue  # a role (text_columns / identifiers), not a metadata field
        if field not in STANDARD_METADATA_FIELDS:
            raise ValueError(
                f"chunk_metadata_columns: {field!r} is not a chart metadata field; "
                f"choose one of {list(STANDARD_METADATA_FIELDS)}"
            )
        site_names = [aliases] if isinstance(aliases, str) else list(aliases)
        defaults = resolved.get(field, [])
        resolved[field] = site_names + [d for d in defaults if d not in site_names]
    return resolved


def _locate_column_map(path: Path | str) -> Path:
    """Find a column map named in a config, whether or not the run starts in the repo.

    A config says ``deployment/<site>/<your>_Column_Name_Map.yaml``, which only resolves from the
    repo root — and a SLURM job or a cron run starts somewhere else. So a relative path
    is tried against the working directory first, then against the checkout this package
    is running from.
    """
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (  # noqa: E501
            resolve_repo_root,
        )
        from_repo = resolve_repo_root() / candidate
        if from_repo.is_file():
            return from_repo
    raise FileNotFoundError(
        f"chart column map not found: {path} (tried the working directory and the "
        "repo root)"
    )


# Parsed maps, keyed by resolved path + mtime. Preflight asks for the mapping per file,
# and a cohort scan multiplies that by every patient; without this, a 500-patient scan
# re-parses the same yaml tens of thousands of times. Keyed by mtime so editing the map
# between runs in one session is still picked up.
_MAP_CACHE: dict[tuple[str, float], dict[str, dict[str, list[str]]]] = {}


def load_column_map(path: Path | str) -> dict[str, dict[str, list[str]]]:
    """Read one institution's column map: a yaml holding a ``chunk_metadata_columns``
    block. See ``deployment/example_site/example_export_column_map.yaml`` for the shape.

    A site's map is a file rather than a block pasted into every config so that adding an
    institution is adding a folder under ``deployment/``, and two runs at the same site
    cannot drift apart by editing one config and not another.
    """
    import yaml

    resolved = _locate_column_map(path)
    cache_key = (str(resolved), resolved.stat().st_mtime)
    cached = _MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    mapping = loaded.get("chunk_metadata_columns")
    if not isinstance(mapping, dict):
        raise ValueError(
            f"{resolved} has no 'chunk_metadata_columns:' block — see "
            "deployment/example_site/example_export_column_map.yaml"
        )
    by_source = _validate_column_map(mapping, str(resolved))
    _MAP_CACHE[cache_key] = by_source
    return by_source


def _validate_column_map(mapping: dict, where: str) -> dict[str, dict]:
    """A column map is keyed by source file, then by field. Reject anything else with a
    message showing the shape — a flat ``field: column`` map (no file) would otherwise
    fail further down, where the cause is much harder to see."""
    for source_name, entry in mapping.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{where}: chunk_metadata_columns is keyed by source file, then by "
                f"field — got {source_name!r}: {entry!r}. Expected e.g.\n"
                "  chunk_metadata_columns:\n"
                "    clinical_note:\n"
                "      document_date: date\n"
                "      author: author"
            )
    return {normalize_source_name(name): entry for name, entry in mapping.items()}


def columns_from_config(cfg: dict, source_name: str | None = None) -> dict[str, list[str]]:
    """The alias list to search per field, for one source file.

    Maps are written per document type, because one name does not fit every table: a
    note's author is ``author``, a lab's is ``authorizing_provider``, a procedure's is
    ``performing_provider``. Per-file entries also make a map readable as an inventory of
    what a site's export actually offers.

    Three layers, each overriding the one below:

      1. ``chunk_metadata_columns:`` written inline in the config — a one-off tweak;
      2. ``chart_columns_file:`` pointing at an institution's map under deployment/;
      3. the generic defaults above, used for any file a map does not mention.
    """
    from_file = load_column_map(cfg["chart_columns_file"]) if cfg.get("chart_columns_file") else {}
    inline = _validate_column_map(cfg.get("chunk_metadata_columns") or {}, "config")
    by_source = {**from_file, **inline}
    entry = by_source.get(normalize_source_name(source_name)) if source_name else None
    return resolve_column_aliases(entry)


def match_key(name: str) -> str:
    """How a column name and an alias are compared: case, padding and word separator
    ignored.

    An export that ships ``AUTHOR``, ``Note_Date``, ``"Note Date"`` or ``" author "``
    (Excel round-trips add the padding) means the same column as ``author`` or
    ``note_date``. Any of those used to read as "this file has no author" — silently, on
    every row — and a space-separated header would then be mistaken for something else
    entirely, since nothing had claimed it.
    """
    return name.strip().lower().replace(" ", "_")


def _role_columns(cfg: dict, source_name: str, key: str) -> list[str]:
    from_file = load_column_map(cfg["chart_columns_file"]) if cfg.get("chart_columns_file") else {}
    inline = _validate_column_map(cfg.get("chunk_metadata_columns") or {}, "config")
    entry = {**from_file, **inline}.get(normalize_source_name(source_name)) or {}
    named = entry.get(key) or []
    return [named] if isinstance(named, str) else list(named)


def text_columns_for(cfg: dict, source_name: str) -> list[str]:
    """The free-text columns this file's map says to embed; empty when it says nothing
    (the caller then falls back to its configured default column)."""
    return _role_columns(cfg, source_name, TEXT_COLUMNS_KEY)


def identifier_columns_for(cfg: dict, source_name: str) -> list[str]:
    """Columns this file's map marks as direct identifiers — never embedded."""
    return _role_columns(cfg, source_name, IDENTIFIERS_KEY)


def data_columns_for(cfg: dict, source_name: str) -> dict[str, str]:
    """``{our name: this site's column}`` for one table's own data.

    Naming a column is not the same as consuming it: most of these are read by nothing
    today. They are named so a recipe can ask for ``race`` or ``result_value`` instead
    of whatever this export happens to call it — and so the map reads as a complete
    dictionary of the table rather than a list of the few fields Junior happens to use.
    """
    from_file = load_column_map(cfg["chart_columns_file"]) if cfg.get("chart_columns_file") else {}
    inline = _validate_column_map(cfg.get("chunk_metadata_columns") or {}, "config")
    entry = {**from_file, **inline}.get(normalize_source_name(source_name)) or {}
    named = entry.get(DATA_COLUMNS_KEY) or {}
    if not isinstance(named, dict):
        raise ValueError(
            f"{source_name}: {DATA_COLUMNS_KEY} maps our name to this site's column, "
            f"e.g. {{result_value: value}} — got {named!r}"
        )
    return dict(named)


def find_column(aliases: list[str], available_columns: list[str]) -> str | None:
    """The first alias present among ``available_columns``, or None. The name returned is
    the column's real name, so the caller can read it straight off the row.

    When two columns match the same alias (``Author`` beside ``AUTHOR``), the one that
    appears FIRST in the file wins — arbitrary but fixed, so two runs over the same file
    never disagree. ``ambiguous_columns`` reports the collision so it does not stay
    invisible."""
    by_key: dict[str, str] = {}
    for column in available_columns:
        by_key.setdefault(match_key(column), column)   # first occurrence wins
    for alias in aliases:
        found = by_key.get(match_key(alias))
        if found is not None:
            return found
    return None


def metadata_value_as_text(raw: Any) -> str | None:
    """One metadata value as the text every later step compares against, or None when the
    row carries no value for that field.

    A source column can be typed — a parquet date column, a numeric age — while the
    filters, the date sort and the evidence header all read text. Converting in one place
    is what lets an embedded chunk's ``document_date`` and a ``:TABLE`` row's compare the
    same way: ``str`` of a date gives the ISO form the date parse expects.
    """
    if raw is None or raw == "":
        return None
    return raw if isinstance(raw, str) else str(raw)


def ambiguous_columns(available_columns: list[str]) -> dict[str, list[str]]:
    """Groups of columns in one file that differ only by case or padding, keyed by the
    name they collapse to. Empty when the header is unambiguous.

    Only one of each group can ever be read, so a file carrying both ``Author`` and
    ``AUTHOR`` is a data-prep problem the operator should see rather than a coin flip."""
    groups: dict[str, list[str]] = {}
    for column in available_columns:
        groups.setdefault(match_key(column), []).append(column)
    return {key: names for key, names in groups.items() if len(names) > 1}
