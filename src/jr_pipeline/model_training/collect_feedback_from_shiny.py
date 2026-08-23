"""Read clinician feedback from the Shiny review app and route to training pipelines.

Feedback lives at data/CONTAINS_PHI/expert_label_corrections/<run_id>/<patient_id>.json.
Each file contains one review session with multiple feedback types.

This script filters by feedback type and exports raw reviewer labels only —
never chunk text — so the PHI boundary is explicit. Hydrating text into a
training dataset is a separate, offline step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_all_feedback(feedback_root: Path) -> list[dict[str, Any]]:
    """Read all feedback files and return as a flat list."""
    records = []
    for f in sorted(feedback_root.rglob("*.json")):
        records.append(json.loads(f.read_text(encoding="utf-8")))
    return records


def filter_by_type(records: list[dict], feedback_type: str) -> list[dict]:
    """Extract feedback entries of a specific type across all records."""
    results = []
    for r in records:
        for entry in r.get("feedback", []):
            if entry.get("type") == feedback_type:
                results.append({
                    "patient_id": r["patient_id"],
                    "variable": r["variable"],
                    "run_id": r["run_id"],
                    "annotator": r.get("annotator"),
                    **entry,
                })
    return results


def export_chunk_relevance_labels(feedback_root: Path, output_path: Path) -> int:
    """Export raw chunk relevance labels.

    The rows intentionally contain chunk ids, not chunk text. Use
    ``export_relevance_training_jsonl`` when the trainer needs hydrated PHI text.
    """
    records = collect_all_feedback(feedback_root)
    relevance = filter_by_type(records, "chunk_relevance")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in relevance:
            f.write(json.dumps({
                "patient_id": item["patient_id"],
                "variable": item["variable"],
                "run_id": item["run_id"],
                "chunk_id": item["chunk_id"],
                "relevant": 1 if item["relevant"] else 0,
            }, ensure_ascii=False) + "\n")

    return len(relevance)


def export_extraction_corrections(feedback_root: Path, output_path: Path) -> int:
    """Export extraction corrections as gold labels."""
    records = collect_all_feedback(feedback_root)
    corrections = filter_by_type(records, "extraction_correction")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in corrections:
            f.write(json.dumps({
                "patient_id": item["patient_id"],
                "variable": item["variable"],
                "correct_value": item.get("correct_value"),
            }, ensure_ascii=False) + "\n")

    return len(corrections)
