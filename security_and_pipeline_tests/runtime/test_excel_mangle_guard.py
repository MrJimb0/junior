"""Ingest flags Excel-mangled columns. Opening/saving an EHR export CSV in Excel rewrites a
big integer ID (e.g. 131019000000) to scientific notation ("1.31019E+11"), silently
destroying the exact value. We detect that and warn so the file is re-exported."""
from __future__ import annotations

import polars as pl

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import _excel_mangled_columns


def test_detects_scientific_notation_id_column():
    df = pl.DataFrame({
        "pat_enc_csn_id": ["1.31019E+11", "1.31020E+11"],   # Excel-mangled
        "text": ["note one — patient seen", "note two"],     # free text, fine
        "mrn": ["00123", "00456"],                           # leading zeros preserved, fine
    })
    found = _excel_mangled_columns(df)
    assert [c for c, _ in found] == ["pat_enc_csn_id"]
    assert found[0][1] == "1.31019E+11"


def test_clean_table_flags_nothing():
    df = pl.DataFrame({
        "pat_enc_csn_id": ["131019000000", "131020000000"],  # exact integer strings
        "date": ["2024-02-15", "2024-03-01"],
        "value": ["5.1", "6.2"],                              # decimals are not sci-notation
        "text": ["a", "b"],
    })
    assert _excel_mangled_columns(df) == []
