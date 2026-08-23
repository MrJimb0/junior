"""Structured logging must write FLAT fields and a durable per-run JSONL sink.

Two invariants this guards:
- ``log.info("e", extra_={...})`` flattens the dict to top-level fields, so the JSON
  reads ``{"event": "e", "chunks": 1}`` rather than nesting under a literal ``extra_`` key.
- ``set_run_log_file`` tees every event into a run-scoped file, so a crashed or
  SLURM-array run does not lose stdout-only output.

Verifying via the file sink (not capsys) sidesteps structlog's cached-PrintLogger stdout.
"""
import json

from jr_pipeline.runtime_infrastructure.json_event_logging import (
    clear_run_log_file,
    get_logger,
    set_run_log_file,
)


def test_run_log_sink_writes_flat_jsonl(tmp_path):
    log_file = tmp_path / "logs" / "run_log.jsonl"  # parent created by set_run_log_file
    set_run_log_file(log_file)
    try:
        get_logger("test").info("embed_done", extra_={"chunks": 7, "dim": 768})
    finally:
        clear_run_log_file()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "the run log sink wrote nothing"
    rec = json.loads(lines[-1])
    assert rec["event"] == "embed_done"
    assert rec["chunks"] == 7   # flattened to top level, not nested under extra_
    assert rec["dim"] == 768
    assert "extra_" not in rec


def test_clear_stops_teeing(tmp_path):
    log_file = tmp_path / "run_log.jsonl"
    set_run_log_file(log_file)
    get_logger("test").info("first_event", extra_={"n": 1})
    clear_run_log_file()
    get_logger("test").info("second_event", extra_={"n": 2})

    events = [json.loads(line)["event"] for line in log_file.read_text().strip().splitlines()]
    assert "first_event" in events
    assert "second_event" not in events  # cleared -> no longer teed to this file
