"""Step 8 reconstructs the PHI evidence artifacts from the step-6 packet.

prepare_evidence is pure; step 8 owns these writes. The artifacts must be byte-faithful
to the step-6 packet so the workbench panes and the slow-spine prepared_evidence_text/
assertion keep working.
"""
from __future__ import annotations

import json

from jr_pipeline.pipeline_steps.step_8_organize_output.write_evidence_artifacts import (
    write_evidence_artifacts,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    evidence_selection_metadata_dir,
    prepared_evidence_text_dir,
)


def _packet():
    return {
        "patient_id": "P",
        "variable": "v/s",
        "formatted_evidence_text": "[Evidence 1 | chunk_id=P:notes:0:0 | ...]\nbody text",
        "block_count": 1,
        "total_evidence_tokens": 42,
        "evidence_tokens_by_doc_type": {"progress_note": 42},
        "max_context_tokens": 200_000,
        "included_chunk_ids": ["P:notes:0:0"],
        "blocks": [{
            "chunk_id": "P:notes:0:0", "rank": 1, "score": 0.9, "source_file": "notes",
            "doc_type": "progress_note", "document_date": "2024-01-02",
            "record_index": 0, "chunk_index": 0, "text": "body text", "token_count": 5,
        }],
    }


def test_writes_prepared_evidence_and_selection(tmp_path):
    run_id, pid, var = "20990101_000000", "P", "v/s"
    write_evidence_artifacts(
        run_id, pid, [{"variable": var, "evidence_packet": _packet()}], data_root=tmp_path
    )

    ev_dir = prepared_evidence_text_dir(run_id, pid, tmp_path) / var
    assert (ev_dir / "formatted_evidence.txt").read_text(encoding="utf-8").startswith("[Evidence 1")
    blocks = json.loads((ev_dir / "evidence_blocks.json").read_text(encoding="utf-8"))
    assert blocks[0]["chunk_id"] == "P:notes:0:0"
    assert blocks[0]["text"] == "body text"            # full provenance incl. chunk text
    assert blocks[0]["doc_type"] == "progress_note"

    sel_dir = evidence_selection_metadata_dir(run_id, pid, tmp_path) / var
    sel = json.loads((sel_dir / "evidence_selection.json").read_text(encoding="utf-8"))
    assert sel["block_count"] == 1
    assert sel["total_evidence_tokens"] == 42           # chars//4 unit
    assert sel["evidence_tokens_by_doc_type"] == {"progress_note": 42}
    assert sel["blocks"][0]["chunk_id"] == "P:notes:0:0"
    assert "text" not in sel["blocks"][0]               # selection metadata is id-level, no chunk text
    assert "doc_type" not in sel["blocks"][0]


def test_empty_step_evidence_writes_nothing(tmp_path):
    # a recipe with no retrieve_and_prompt step hands over nothing -> no files, no error
    write_evidence_artifacts("20990101_000000", "P", [], data_root=tmp_path)
    assert not prepared_evidence_text_dir("20990101_000000", "P", tmp_path).exists()
