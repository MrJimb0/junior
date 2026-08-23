"""Export a run's NO_PHI metadata as a shareable zip bundle.

For cross-institution benchmarking: share pipeline config, aggregate metrics,
and experiment results without any patient data.

CLI: jr-pipeline export-metadata --run-id 20260524_001 --output shared_run_001.zip
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
    PHI_CONTENT_PATTERNS,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    no_phi_run_dir,
    phi_intermediate_run_dir,
)

# Files that may be exported without a text scan. EVERY non-text file fails closed:
# a scanner that can't read a file's bytes as text must not pass it.
_SCAN_EXEMPT_NAMES: frozenset[str] = frozenset()


def _run_patient_ids(run_id: str, dr: Path | None) -> list[str]:
    """Patient ids for this run, for surrogate-leak detection in the NO_PHI tree. Prefers
    the PHI/medium run_roster.json; falls back to the run manifest's
    target_patients. Empty if neither is available."""
    phi_root = phi_intermediate_run_dir(run_id, dr)
    for candidate in (phi_root / "run_roster.json", phi_root / "manifest.json"):
        try:
            env = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload = env.get("payload") or env
        ids = payload.get("target_patients") or payload.get("patient_ids") or []
        if ids:
            return [str(p) for p in ids]
    return []


def _scan_file_for_phi(path: Path, *, patient_ids: list[str]) -> list[str]:
    """Forbidden-content findings in one file (empty list = clean).

    Fails CLOSED: any file that cannot be read as UTF-8 text (parquet/npz/gzip,
    binary, unreadable) is itself a finding unless explicitly allowlisted — a scanner
    that can't read a file's bytes as text must not pass a PHI-bearing file.

    The one accepted binary is a finalized ``exhaust/<record_type>.parquet``:
    it is content-scanned (not name-allowlisted) — manifest-hash-verified, schema-valid,
    and row-by-row PHI-scanned. Any other binary stays fail-closed.

    Stream-aware: files under ``sealed_code/`` are the METHOD (source + recipe
    prompts), which legitimately contain clinical terms and example dates ("Date of
    Birth:", a date in a comment). The clinical-CONTENT patterns target patient DATA and
    would false-positive there, so code-stream files instead get the PHI path/id scrub
    (``check_scrub_file``) — they remain checked for PHI paths, MRNs, and patient ids.
    """
    if path.suffix == ".parquet" and path.parent.name == "exhaust":
        from jr_pipeline.runtime_infrastructure.exhaust.egress import verify_exhaust_parquet
        return verify_exhaust_parquet(path, patient_ids=patient_ids)
    if path.name in _SCAN_EXEMPT_NAMES:
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return [f"{path.name}: non-text/unreadable file in NO_PHI tree — cannot be PHI-scanned (fail closed)"]
    if "sealed_code" in path.parts:  # code stream: path/id scrub, not clinical-content
        from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
            check_scrub_file,
        )
        findings = [f"{path.name}: {v}" for v in check_scrub_file(path.name, path).violations]
    else:  # data stream: clinical-content scan
        findings = [f"{path.name}: {label}" for pat, label in PHI_CONTENT_PATTERNS if pat.search(content)]
    findings.extend(
        f"{path.name}: patient id {pid!r} present" for pid in patient_ids if pid and pid in content
    )
    return findings


def export_run_metadata(
    *,
    run_id: str,
    output_path: str | Path,
    dr: Path | None = None,
) -> Path:
    """Zip the NO_PHI run directory into a shareable bundle, scanning every file for
    forbidden content first. A single finding aborts the export — loud failure
    at the egress moment is correct; a leaked identifier costs the project."""
    output_path = Path(output_path)
    run_dir = no_phi_run_dir(run_id, dr)

    if not run_dir.exists():
        raise FileNotFoundError(f"No metadata found for run {run_id} at {run_dir}")

    files = [f for f in sorted(run_dir.rglob("*")) if f.is_file()]
    patient_ids = _run_patient_ids(run_id, dr)

    # Egress scan BEFORE we write anything — never ship a partial PHI-bearing zip.
    findings: list[str] = []
    for f in files:
        findings.extend(_scan_file_for_phi(f, patient_ids=patient_ids))
    # A per-patient folder structure in the NO_PHI tree is itself a leak.
    if any("patients" in f.relative_to(run_dir).parts for f in files):
        findings.append("per-patient folder structure in the NO_PHI tree")
    if findings:
        raise ValueError(
            "Export ABORTED — forbidden content in the NO_PHI tree:\n  "
            + "\n  ".join(findings[:20])
            + ("\n  ..." if len(findings) > 20 else "")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(run_dir.parent))

    print(f"Exported {len(files)} files to {output_path} (PHI scan passed)")
    return output_path
