"""Run manifest builder.

The manifest is the first artifact written for any run, by ``cli.seal``,
and the only writers are seal + the terminal ``mark_completed`` flip in
summarize. Stage commands never touch it.
"""
from __future__ import annotations

import getpass
import json
import os
import socket
from datetime import UTC, datetime
from pathlib import Path

from jr_pipeline import __version__ as _pkg_version
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_artifact_payload,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
    validate_artifact,
)
from jr_pipeline.runtime_infrastructure.artifact_store import artifact_path, write_artifact
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
)


def model_sha256_from_cfg(cfg: dict) -> str | None:
    """Return the pinned ``model.safetensors`` sha256 from a step config, or None."""
    enc = cfg.get("encoder") if isinstance(cfg, dict) else None
    if not isinstance(enc, dict):
        return None
    pins = enc.get("expected_file_sha256")
    if not isinstance(pins, dict):
        return None
    raw = pins.get("model.safetensors")
    if not isinstance(raw, str) or not raw:
        return None
    # Validate here so a typo fails with a clear message, not late in seal.
    import re as _re
    m = _re.fullmatch(r"(sha256:)?([A-Fa-f0-9]{64})", raw)
    if not m:
        raise ValueError(
            f"encoder.expected_file_sha256['model.safetensors'] is not a 64-hex sha256: "
            f"{raw!r}. Expected 64 hex chars, optionally prefixed with 'sha256:'."
        )
    return f"sha256:{m.group(2).lower()}"


def build_manifest(
    *,
    run_id: str,
    code_lock_hash: str,
    entry_point_name: str,
    config_alias: str | None,
    target_patients: list[str],
    model_sha256: str | None = None,
) -> dict:
    """Build and validate a manifest envelope. Does not write to disk."""
    payload = {
        "run_id": run_id,
        "code_lock_hash": code_lock_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "started_by": {
            "user": getpass.getuser(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID") or None,
        },
        "entry_point_name": entry_point_name,
        "config_alias": config_alias,
        # The raw patient roster lives in the PHI-side run_roster.json; this
        # low-sensitivity manifest carries only a safe COUNT, never the ids.
        "n_target_patients": len(target_patients),
        "completed_at": None,
        "status": "running",
        "model_sha256": model_sha256,
    }
    env = envelope_for(
        artifact_type="manifest",
        sensitivity="low",
        stream="data",
        run_id=run_id,
        step="code_seal",
        payload=payload,
        code_lock_hash=code_lock_hash,
        code_version=f"jr_pipeline@{_pkg_version}",
    )
    env["content_hash"] = hash_artifact_payload(env)
    validate_artifact(env, "manifest")
    return env


def write_manifest(run_root: Path, manifest: dict) -> Path:
    """Write the run header, without un-finishing a run that already finished.

    build_manifest hardcodes ``status: "running"`` and ``completed_at: None``, which is
    right for a run being created and wrong for every other time this is called — and it
    is called on EVERY entry into a run, not only the first. So re-entering a closed run
    reset its header to "running" and dropped the time it completed, and the only reason
    that was survivable is that almost nothing reads either field.

    Rather than ask each caller to remember, the finished state is carried forward here:
    a run that has been marked completed stays completed. mark_completed remains the one
    way to change it."""
    run_root = Path(run_root)
    already = run_root / "manifest.json"
    if already.is_file():
        try:
            previous = (json.loads(already.read_text(encoding="utf-8")).get("payload")
                        or {})
        except (OSError, json.JSONDecodeError):
            previous = {}
        if previous.get("status") and previous.get("status") != "running":
            manifest["payload"]["status"] = previous["status"]
            manifest["payload"]["completed_at"] = previous.get("completed_at")
            # Re-hashed in place: the payload changed after build_manifest stamped it,
            # and content_hash is over the payload. Same two calls build_manifest ends
            # on, so the envelope stays whatever that function decided it should be.
            manifest["content_hash"] = hash_artifact_payload(manifest)
            validate_artifact(manifest, "manifest")
    # build_manifest already hashed+validated the envelope; write_artifact re-stamps
    # (idempotent) and routes the path through the registry (-> run_root/manifest.json).
    return write_artifact(manifest, run_root=run_root)


def build_roster(*, run_id: str, target_patients: list[str]) -> dict:
    """The PHI run roster: the raw patient ids, kept PHI-side for the
    egress scanner only and NEVER exported. Labeled ``sensitivity:"high"`` so the
    share/export scans exclude it. Not an envelope -- it is an internal scanner input,
    never a shareable artifact."""
    return {
        "run_id": run_id,
        "sensitivity": "high",
        "created_at": datetime.now(UTC).isoformat(),
        "target_patients": list(target_patients),
    }


def write_roster(run_root: Path, roster: dict) -> Path:
    # plain dict (no envelope/schema); the registry still owns the path.
    return atomic_write_json(artifact_path("run_roster", run_root=Path(run_root)), roster)


def mark_completed(run_root: Path, *, status: str = "completed") -> Path:
    """Atomically stamp ``completed_at`` + ``status`` on the manifest."""
    path = Path(run_root) / "manifest.json"
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["payload"]["completed_at"] = datetime.now(UTC).isoformat()
    manifest["payload"]["status"] = status
    return write_artifact(manifest, run_root=Path(run_root))  # re-stamps hash + validates + writes
