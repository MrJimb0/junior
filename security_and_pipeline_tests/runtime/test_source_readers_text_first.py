"""Ingest reads every source file as text — including JSON/JSONL, whose parsers
would otherwise infer numeric/boolean column types and diverge from how the same
value reads out of a CSV (which is read with ``infer_schema_length=0``)."""
from __future__ import annotations

import json

import polars as pl

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.source_format_readers import _read_source


def _all_text(df: pl.DataFrame) -> bool:
    return all(dtype == pl.Utf8 for dtype in df.dtypes)


def test_json_array_is_read_as_text(tmp_path):
    path = tmp_path / "clinical_note.json"
    path.write_text(json.dumps([
        {"note_id": 1, "flag": True, "text": "hello"},
        {"note_id": 2, "flag": False, "text": "world"},
    ]), encoding="utf-8")

    df = _read_source(path)

    assert _all_text(df)
    assert df["note_id"].to_list() == ["1", "2"]           # integers stringified
    assert all(isinstance(v, str) for v in df["flag"].to_list())  # booleans stringified


def test_jsonl_is_read_as_text(tmp_path):
    path = tmp_path / "labs.jsonl"
    path.write_text(
        '{"lab_id": 10, "value": 4.5}\n{"lab_id": 11, "value": 9.0}\n',
        encoding="utf-8",
    )

    df = _read_source(path)

    assert _all_text(df)
    assert df["lab_id"].to_list() == ["10", "11"]
    assert all(isinstance(v, str) for v in df["value"].to_list())


def test_json_nested_fields_survive_instead_of_crashing_ingest(tmp_path):
    # A JSON file with a list- or object-valued field must still ingest: scalar fields
    # become text, nested fields are left intact. A blanket cast-to-text would raise on
    # the list column and fail the whole patient's ingest.
    path = tmp_path / "clinical_note.json"
    path.write_text(json.dumps([
        {"note_id": 1, "codes": ["E11.9", "I10"], "coder": {"id": 7, "name": "auto"}, "text": "hi"},
    ]), encoding="utf-8")

    df = _read_source(path)

    assert df.schema["note_id"] == pl.Utf8               # scalar cast to text
    assert df["note_id"].to_list() == ["1"]
    assert df["codes"].to_list() == [["E11.9", "I10"]]   # list preserved, not cast/crashed
    assert isinstance(df.schema["coder"], pl.Struct)     # struct preserved
