"""The collect-feedback CLI + the chunk-export.

collect-feedback wires the two Shiny-feedback exporters (corrections + chunk relevance)
to training JSONL; export_for_labeling reads the REAL retrieval envelopes.
"""
from __future__ import annotations

import json

from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import main
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    clinician_feedback_dir,
)

RUN = "20260101_000000"


def test_collect_feedback_cli_exports_corrections_and_relevance(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    fb = clinician_feedback_dir(RUN)
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "p__v.json").write_text(json.dumps({
        "patient_id": "p", "variable": "v", "run_id": RUN, "annotator": "a",
        "feedback": [
            {"type": "extraction_correction", "correct_value": "1957-04-15"},
            {"type": "chunk_relevance", "chunk_id": "p:notes:0:0", "relevant": True},
        ],
    }), encoding="utf-8")

    out_dir = tmp_path / "export"
    res = CliRunner().invoke(main, ["collect-feedback", "--run-id", RUN, "--output-dir", str(out_dir)])
    assert res.exit_code == 0, res.output
    corr = (out_dir / "extraction_corrections.jsonl").read_text().splitlines()
    rel = (out_dir / "chunk_relevance.jsonl").read_text().splitlines()
    assert len(corr) == 1 and json.loads(corr[0])["correct_value"] == "1957-04-15"
    assert len(rel) == 1 and json.loads(rel[0])["relevant"] == 1
