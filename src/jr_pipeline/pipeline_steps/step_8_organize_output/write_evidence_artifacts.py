"""Step 8: write the patient-identifiable (CONTAINS_PHI) evidence files from the
packet step 6 assembled (the source-text spans that backed each answer).

Step 8 owns the output writes. These are rebuilt straight from the evidence packet
handed across the step7 -> step8 boundary -- no need to re-read the source corpus:
    CONTAINS_PHI/prepared_evidence_text/<variable>/formatted_evidence.txt
    CONTAINS_PHI/prepared_evidence_text/<variable>/evidence_blocks.json
    CONTAINS_PHI/evidence_selection_metadata/<variable>/evidence_selection.json

These hold PHI because the chunk_ids embed the patient id. The shareable, text-free
counterpart is the de-identified (NO_PHI)
evidence_selection_trace.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    evidence_selection_metadata_dir,
    prepared_evidence_text_dir,
)

# An evidence block carries full provenance in evidence_blocks.json; the lighter
# id-level selection metadata records only this subset of fields (no chunk text, no
# document type, no date).
_SELECTION_BLOCK_KEYS = (
    "chunk_id", "rank", "score", "source_file", "record_index", "chunk_index", "token_count",
)


def write_evidence_artifacts(
    run_id: str,
    patient_id: str,
    step_evidence: list[dict[str, Any]],
    data_root: Path | None = None,
) -> None:
    """Write the patient-identifiable (PHI) evidence files for each step's packet.
    ``step_evidence`` is the list of ``{"variable", "evidence_packet"}`` step 7 handed
    over (one entry per retrieve-and-prompt step). Safe to re-run: re-running rewrites
    the same files."""
    for entry in step_evidence:
        variable = entry["variable"]
        packet = entry["evidence_packet"]
        _write_prepared_evidence_text(run_id, patient_id, variable, packet, data_root)
        _write_evidence_selection_metadata(run_id, patient_id, variable, packet, data_root)


def _write_prepared_evidence_text(
    run_id: str, patient_id: str, variable: str, packet: dict[str, Any], data_root: Path | None,
) -> None:
    out_dir = prepared_evidence_text_dir(run_id, patient_id, data_root) / variable
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted_path = out_dir / "formatted_evidence.txt"
    tmp = formatted_path.with_suffix(".txt.tmp")
    tmp.write_text(packet.get("formatted_evidence_text", ""), encoding="utf-8")
    tmp.rename(formatted_path)  # same-dir rename: the file appears fully written or not at all

    atomic_write_json(out_dir / "evidence_blocks.json", packet.get("blocks", []))


def _write_evidence_selection_metadata(
    run_id: str, patient_id: str, variable: str, packet: dict[str, Any], data_root: Path | None,
) -> None:
    out_dir = evidence_selection_metadata_dir(run_id, patient_id, data_root) / variable
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = packet.get("blocks", [])
    atomic_write_json(out_dir / "evidence_selection.json", {
        "block_count": packet.get("block_count", len(blocks)),
        "total_evidence_tokens": packet.get("total_evidence_tokens"),
        "evidence_tokens_by_doc_type": packet.get("evidence_tokens_by_doc_type", {}),
        "max_context_tokens": packet.get("max_context_tokens"),
        "blocks": [{k: b.get(k) for k in _SELECTION_BLOCK_KEYS} for b in blocks],
    })
