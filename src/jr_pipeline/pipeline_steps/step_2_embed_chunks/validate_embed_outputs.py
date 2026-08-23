"""Sanity-checks the embed step's output for one patient, run right after embedding.

Every cluster job calls this once the embed command returns. These checks catch
ways the output can be quietly wrong that type checks and unit tests cannot:

  shape_aligned          — embeddings.npy has the same number of rows as
                           chunk_index.parquet (each vector must line up with its
                           chunk description)
  no_nan / no_inf        — every number in every vector is a real, finite value (a
                           single silent "not-a-number" would corrupt the search
                           index)
  no_zero_vectors        — no row has length below 1e-8 (which can come from
                           half-precision rounding, or from a chunk that tokenizes
                           to nothing but special marker tokens)
  normalized             — every vector has length ≈ 1 (within ± 1e-3); the encoder
                           is supposed to scale vectors to length 1, and if it
                           doesn't, the index's similarity math is wrong
  chunk_count_reasonable — the patient has at least min_chunks chunks; zero chunks
                           for a patient who DOES have source files almost always
                           means a chunking-config bug, not a genuinely empty chart

strict=True (default) raises on the first failed check. strict=False returns the
full report so other tooling can inspect every check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


@dataclass
class EmbedValidationReport:
    patient_id: str
    run_id: str
    ok: bool
    checks: dict[str, dict[str, Any]]

    def to_json(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "run_id": self.run_id,
            "ok": self.ok,
            "checks": self.checks,
        }


def _mk_fail(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": False, "reason": reason}
    out.update(extra)
    return out


def _mk_pass(**extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True}
    out.update(extra)
    return out


def validate_patient_output(
    patient_id: str,
    cfg: dict,
    *,
    strict: bool = True,
    min_chunks: int = 1,
    norm_tol: float = 1e-3,
) -> EmbedValidationReport:
    """Run the five checks; raises RuntimeError on the first failure when strict=True."""
    run_id = cfg["run_id"]
    pdir = phi_patient_run_dir(run_id, patient_id)

    checks: dict[str, dict[str, Any]] = {}
    emb_path = pdir / "embeddings.npy"
    idx_path = pdir / "chunk_index.parquet"

    if not emb_path.is_file():
        checks["artifacts_present"] = _mk_fail("embeddings.npy missing", path=str(emb_path))
        return _finish(patient_id, run_id, checks, strict)
    if not idx_path.is_file():
        checks["artifacts_present"] = _mk_fail("chunk_index.parquet missing", path=str(idx_path))
        return _finish(patient_id, run_id, checks, strict)
    checks["artifacts_present"] = _mk_pass()

    mat = np.load(emb_path, allow_pickle=False)
    n_emb = int(mat.shape[0])
    dim = int(mat.shape[1]) if mat.ndim == 2 else 0
    df_rows = int(pl.read_parquet(idx_path).height)

    if n_emb != df_rows:
        checks["shape_aligned"] = _mk_fail(
            f"row mismatch: embeddings={n_emb} chunk_index={df_rows}",
            n_embeddings=n_emb,
            n_chunk_index=df_rows,
        )
    else:
        checks["shape_aligned"] = _mk_pass(rows=n_emb, dim=dim)

    # kept separate from shape_aligned so a patient with no chunks (where 0 == 0 looks "aligned") still fails loudly.
    if n_emb < int(min_chunks):
        checks["chunk_count_reasonable"] = _mk_fail(
            f"only {n_emb} chunks (min={min_chunks})",
            n_chunks=n_emb,
            min_chunks=int(min_chunks),
        )
    else:
        checks["chunk_count_reasonable"] = _mk_pass(n_chunks=n_emb)

    # if there are no rows at all, skip the per-vector numeric checks rather than failing them.
    if n_emb == 0:
        for key in ("no_nan", "no_inf", "no_zero_vectors", "normalized"):
            checks[key] = {"ok": True, "skipped": "no rows"}
        return _finish(patient_id, run_id, checks, strict)

    nan_rows = int(np.isnan(mat).any(axis=1).sum())
    if nan_rows:
        checks["no_nan"] = _mk_fail(f"{nan_rows} rows contain NaN", nan_rows=nan_rows)
    else:
        checks["no_nan"] = _mk_pass()

    inf_rows = int(np.isinf(mat).any(axis=1).sum())
    if inf_rows:
        checks["no_inf"] = _mk_fail(f"{inf_rows} rows contain inf", inf_rows=inf_rows)
    else:
        checks["no_inf"] = _mk_pass()

    norms = np.linalg.norm(mat, axis=1)

    zero_rows = int((norms < 1e-8).sum())
    if zero_rows:
        checks["no_zero_vectors"] = _mk_fail(
            f"{zero_rows} rows have ||v|| < 1e-8",
            zero_rows=zero_rows,
        )
    else:
        checks["no_zero_vectors"] = _mk_pass()

    max_dev = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    if max_dev > float(norm_tol):
        checks["normalized"] = _mk_fail(
            f"max |||v||-1| = {max_dev:.3e} exceeds tol {norm_tol}",
            max_deviation=max_dev,
            tol=float(norm_tol),
        )
    else:
        checks["normalized"] = _mk_pass(max_deviation=max_dev)

    return _finish(patient_id, run_id, checks, strict)


def _finish(
    patient_id: str,
    run_id: str,
    checks: dict[str, dict[str, Any]],
    strict: bool,
) -> EmbedValidationReport:
    ok = all(c.get("ok", False) for c in checks.values())
    report = EmbedValidationReport(patient_id=patient_id, run_id=run_id, ok=ok, checks=checks)
    if strict and not ok:
        first_fail = next(
            (name for name, c in checks.items() if not c.get("ok", False)),
            "unknown",
        )
        reason = checks[first_fail].get("reason", "failed")
        raise RuntimeError(
            f"validate_patient_output[{patient_id}]: {first_fail} — {reason}"
        )
    return report


def write_validation_report(report: EmbedValidationReport, cfg: dict) -> Path:
    """Write the report next to the patient's embed output files. This is an
    operational check (did embedding produce sane output?), not a provenance
    record."""
    run_id = cfg["run_id"]
    pdir = phi_patient_run_dir(run_id, report.patient_id)
    out = pdir / "validate_embed.json"
    tmp = out.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(report.to_json(), f, indent=2, sort_keys=True)
    tmp.rename(out)
    return out
