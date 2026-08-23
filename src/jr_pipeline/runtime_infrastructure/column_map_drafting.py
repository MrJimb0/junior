"""Draft a site's column map from the headers of a real export (``junior columns``).

Junior reads the same chart metadata off every chunk — who wrote a document, when, what
kind — and recipes filter evidence by those names. Your columns are named whatever your
export names them, so each site maps them once. This reads one patient's files, guesses
what it can, and prints a map to edit and save as
``deployment/<institution>/<name>_Column_Name_Map.yaml``.

Reads HEADER LINES ONLY — never a row of patient data.

Guesses are a starting point, not an answer: the two lines worth checking hardest are
which column is the document's date and which holds its free text. Anything it could not
place is listed under ``data_columns`` named after itself, so the whole header is in
front of the reader and nothing is silently dropped.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from jr_pipeline.runtime_infrastructure.chart_metadata_fields import (
    DEFAULT_COLUMN_ALIASES,
    STANDARD_METADATA_FIELDS,
    find_column,
)

READABLE = (".csv", ".tsv", ".txt")
# A column whose name says it holds a document's prose. Junior embeds these.
TEXT_HINTS = ("text", "note", "report", "narrative", "summary", "impression", "comment")
# A column naming the patient rather than describing the document. Never embedded.
IDENTIFIER_HINTS = ("patient_id", "patient_name", "name", "mrn", "ssn", "dob",
                    "accession", "_id", "id_", "key", "zipcode", "address")


# A column name safe to write in YAML with no quotes. Real exports ship headers with
# colons, commas, hashes and spaces ("Result: Final", "author, primary"), any of which
# would produce a file that does not parse — so anything outside this shape is quoted.
_PLAIN_SCALAR = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
# Words YAML reads as booleans or null rather than as the string they look like.
_RESERVED_WORDS = frozenset(
    "y yes n no true false on off null none ~".split()
)


def yaml_scalar(name: str) -> str:
    """``name`` as a YAML scalar, quoted only when it has to be."""
    if _PLAIN_SCALAR.match(name) and name.lower() not in _RESERVED_WORDS:
        return name
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def header_of(path: Path) -> list[str]:
    """The first line's column names. Nothing else in the file is read."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        return next(csv.reader(handle, delimiter=delimiter), [])


def guess_text_columns(columns: list[str], claimed: set[str]) -> list[str]:
    return [c for c in columns
            if c not in claimed and any(h in c.lower() for h in TEXT_HINTS)]


def guess_identifiers(columns: list[str], claimed: set[str]) -> list[str]:
    return [c for c in columns
            if c not in claimed and any(h in c.lower() for h in IDENTIFIER_HINTS)]


def draft_entry(columns: list[str]) -> tuple[dict, list[str], list[str], list[str]]:
    """(field -> column) for what we recognized, plus the text, identifier and
    left-over columns."""
    fields: dict[str, str] = {}
    for field_name in STANDARD_METADATA_FIELDS:
        found = find_column(DEFAULT_COLUMN_ALIASES[field_name], columns)
        if found is not None:
            fields[field_name] = found
    claimed = set(fields.values())
    text_columns = guess_text_columns(columns, claimed)
    claimed |= set(text_columns)
    identifiers = guess_identifiers(columns, claimed)
    claimed |= set(identifiers)
    return fields, text_columns, identifiers, [c for c in columns if c not in claimed]


def render(folder: Path) -> str:
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in READABLE)
    if not files:
        raise FileNotFoundError(f"no {'/'.join(READABLE)} files in {folder}")

    # Deliberately not the folder's name. This map describes a site's schema and is
    # meant to be shared — with a colleague, with another institution, in a repository —
    # and the folder it was drafted from is one patient, named by patient id in every
    # layout Junior ships. The file count says the same useful thing about provenance.
    out = [
        f"# Column map drafted from one patient's {len(files)} file(s) — CHECK EVERY LINE",
        "# before using it. Guessed from column names only. The two worth checking",
        "# hardest: the document date, and which column holds the free text to embed.",
        "#",
        "# Point this project at it:",
        "#     chart_columns_file: <this file>       (or run `junior columns --edit`)",
        "chunk_metadata_columns:",
    ]
    for path in files:
        columns = header_of(path)
        if not columns:
            continue
        fields, text_columns, identifiers, leftover = draft_entry(columns)
        out.append("")
        out.append(f"  # {path.name}: {', '.join(columns)}")
        out.append(f"  {path.stem}:")
        if text_columns:
            out.append(f"    text_columns: [{', '.join(yaml_scalar(c) for c in text_columns)}]")
        else:
            out.append("    # text_columns: []   # nothing here looks like free text — "
                       "add the column if this file has prose to embed")
        for field_name in STANDARD_METADATA_FIELDS:
            if field_name in fields:
                out.append(f"    {field_name}: {yaml_scalar(fields[field_name])}")
        missing = [f for f in STANDARD_METADATA_FIELDS if f not in fields]
        if missing:
            out.append(f"    # not found: {', '.join(missing)} — map one if a column "
                       "holds it, else leave it out")
        if identifiers:
            out.append(f"    identifiers: [{', '.join(yaml_scalar(c) for c in identifiers)}]")
        if leftover:
            # Named, not discarded: a recipe reading this table directly asks for our
            # name. Defaulted to the site's own name — rename the left side to whatever
            # your recipes should call it.
            out.append("    data_columns:")
            quoted = [yaml_scalar(c) for c in leftover]
            width = max(len(name) for name in quoted)
            for name in quoted:
                out.append(f"      {name + ':':<{width + 1}} {name}")
    return "\n".join(out) + "\n"


def resolve_patient_folder(folder: Path) -> Path:
    """A cohort folder holds patient folders; drop into the first one so either level
    works. Returns the folder whose headers were read."""
    if any(p.suffix.lower() in READABLE for p in folder.iterdir() if p.is_file()):
        return folder
    patients = sorted(p for p in folder.iterdir() if p.is_dir())
    return patients[0] if patients else folder
