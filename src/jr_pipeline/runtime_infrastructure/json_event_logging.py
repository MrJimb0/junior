"""PHI-safe structured logging — thin wrapper around structlog.

Junior owns the PHI hygiene rules. structlog owns the formatting and output.

PHI rules (enforced here, not in structlog):
- Log structured fields, not patient text in messages
- Exception locals are stripped by structlog's built-in processor

Two run-level behaviors layered on top of plain structlog:

- ``extra_`` flattening. Call sites pass structured fields as a single
  ``extra_={...}`` keyword (``log.info("embed_done", extra_={"chunks": n})``).
  Without help structlog nests that whole dict under a literal ``extra_`` key,
  so the JSON read ``{"event": "embed_done", "extra_": {"chunks": 1}}`` instead
  of a flat ``{"event": "embed_done", "chunks": 1}``. ``_flatten_extra`` lifts
  those fields to the top level (never clobbering ``event``/``level``/``timestamp``).

- A durable per-run sink. ``PrintLoggerFactory`` only writes stdout, which a
  SLURM array or a crashed run loses. ``set_run_log_file`` points a tee at a
  run-scoped ``run_log.jsonl`` so every event is also appended there; the
  cohort runner sets it at run start and clears it in its finally.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED = False

# Run-scoped durable sink. A module global (not a contextvar) because a cohort
# run is one sequential pass in one process; the runner overwrites it at the
# start of every run_cohort and clears it in finally, so it can't leak between runs.
_run_log_file: Path | None = None


def set_run_log_file(path: Path | str) -> None:
    """Tee every subsequent log event (as a JSON line) into ``path`` as well as stdout."""
    global _run_log_file
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _run_log_file = p


def clear_run_log_file() -> None:
    """Stop teeing to the run-scoped file (back to stdout only)."""
    global _run_log_file
    _run_log_file = None


# Whether routine events reach the terminal. A stage emits one per file per patient,
# which buries whatever the operator was actually watching for; the run log keeps
# them regardless, so nothing is lost by not printing them.
_console_quiet = False


def set_console_quiet(quiet: bool) -> None:
    """Keep info-level events out of the terminal. Warnings and errors still print,
    and the run log still receives everything either way."""
    global _console_quiet
    _console_quiet = quiet


def _log_level_from_env() -> int:
    """Accept either numeric levels (20) or standard names (INFO)."""
    raw = os.environ.get("JR_LOG_LEVEL", "INFO").strip()
    if raw.isdigit():
        return int(raw)
    level = getattr(logging, raw.upper(), None)
    if isinstance(level, int):
        return level
    raise ValueError(
        f"JR_LOG_LEVEL must be a numeric level or logging name; got {raw!r}"
    )


def _flatten_extra(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Lift ``extra_={...}`` fields to the top level so JSON output is flat.

    Uses setdefault so a stray key inside ``extra_`` can never clobber the
    reserved ``event`` / ``level`` / ``timestamp`` fields added upstream."""
    extra = event_dict.pop("extra_", None)
    if isinstance(extra, dict):
        for k, v in extra.items():
            event_dict.setdefault(k, v)
    return event_dict


def _tee_to_run_log(logger: Any, method_name: str, rendered: str) -> str:
    """Final processor: append the rendered JSON line to the run-scoped file.

    Runs after JSONRenderer, so ``rendered`` is the JSON string. Best-effort —
    a sink write failure must never sink the pipeline, so it is swallowed.

    The file write happens before the quiet check, so silencing the terminal never
    costs the durable record anything — the two sinks differ in what they show, not
    in what they know."""
    if _run_log_file is not None:
        try:
            with _run_log_file.open("a", encoding="utf-8") as f:
                f.write(rendered + "\n")
        except OSError:
            pass
    if _console_quiet:
        if method_name in ("warning", "error", "critical", "exception"):
            # A warning still belongs on screen — but as a sentence. Quieting the
            # routine events and leaving these as raw JSON meant the only lines an
            # operator was left reading were the machine-formatted ones.
            _print_readable(rendered)
        raise structlog.DropEvent
    return rendered


def _print_readable(rendered: str) -> None:
    """Print one structured event as a line a person can read."""
    import json
    import sys

    try:
        event = json.loads(rendered)
    except ValueError:
        print(rendered, file=sys.stderr)
        return

    # `hint` is where these events put the sentence written for a human; everything
    # else is context. Prefer it, and fall back to the event name with its fields.
    headline = event.get("hint") or event.get("event", "").replace("_", " ")
    context = " · ".join(
        f"{key}={value}"
        for key, value in event.items()
        if key in ("patient_id", "stem", "column", "example")
    )
    print(f"  note  {headline}" + (f"\n        {context}" if context else ""),
          file=sys.stderr)


def _configure_once() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _flatten_extra,
            structlog.processors.JSONRenderer(),
            _tee_to_run_log,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(_log_level_from_env()),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "jr_pipeline") -> structlog.stdlib.BoundLogger:
    """Return a PHI-safe structured logger."""
    _configure_once()
    return structlog.get_logger(name)
