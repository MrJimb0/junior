"""Guardrail: the chunk_index doc schema must match the columns embed actually writes.

The doc schema (chunk_index.parquet.schema.json) is documentation-only — nothing
loads it at runtime — so it silently drifted from embed._CHUNK_INDEX_SCHEMA before
(13 documented columns vs 17 written). This test makes the drift a CI failure.
"""
from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import _index_columns
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    load_schema,
)


def test_chunk_index_doc_schema_matches_written_columns():
    doc = load_schema("chunk_index.parquet")
    documented = [c["name"] for c in doc["properties"]["columns"]["const"]]
    written = _index_columns()
    assert documented == written, (
        "chunk_index.parquet.schema.json drifted from embed._CHUNK_INDEX_SCHEMA:\n"
        f"  documented: {documented}\n"
        f"  written:    {written}"
    )
