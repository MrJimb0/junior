"""Embed rejects multi-text-column file specs loudly.

The chunk_id is {patient}:{stem}:{row}:{chunk_idx} — it omits the source column,
so two text columns on the same row would collide (and crash the uniqueness
check mid-run). validate_embed_config rejects multi-column specs up front.
"""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.config_loading import validate_embed_config


def _cfg(files):
    return {"run_id": "r", "encoder": {"model_id": "m"}, "files": files}


def test_multi_text_column_spec_rejected_loudly():
    cfg = _cfg([{"stem": "notes", "embed": True, "text_columns": ["note", "summary"]}])
    with pytest.raises(ValueError, match="one text column per file spec"):
        validate_embed_config(cfg)


def test_duplicate_same_text_column_also_rejected():
    cfg = _cfg([{"stem": "notes", "embed": True, "text_columns": ["note", "note"]}])
    with pytest.raises(ValueError, match="one text column per file spec"):
        validate_embed_config(cfg)


def test_single_text_column_spec_ok():
    cfg = _cfg([{"stem": "notes", "embed": True, "text_columns": ["note"]}])
    validate_embed_config(cfg)  # no raise


def test_auto_files_ok():
    validate_embed_config({"run_id": "r", "encoder": {"model_id": "m"}, "files": "auto"})
