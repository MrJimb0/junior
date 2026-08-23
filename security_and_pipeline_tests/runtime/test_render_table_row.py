"""render_table_row is the single source of the col:value evidence shape.

Both an embedded :TABLE chunk (PatientChunkStore._text_for_table) and a builder-table
evidence row render through this one function, so they read identically in the evidence
bundle instead of relying on a comment to keep two copies in sync.
"""
from __future__ import annotations

from jr_pipeline.runtime_infrastructure.patient_chunk_store import render_table_row


def test_renders_col_value_lines_skipping_blank_cells():
    row = {"stage": "IIA", "grade": None, "note": "  ", "site": "lung"}
    assert render_table_row(row) == "stage: IIA\nsite: lung"


def test_empty_or_all_blank_row_renders_empty_string():
    assert render_table_row({}) == ""
    assert render_table_row({"a": None, "b": "   "}) == ""
