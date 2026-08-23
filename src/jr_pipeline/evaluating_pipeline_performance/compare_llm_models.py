"""Model A/B: run one variable against N allowlist endpoints.

Retrieval is shared implicitly because the runner re-runs the full recipe
for each provider — which executes the same deterministic retrieval
steps. Receipts record ``resolved_model.fingerprint`` so silent backend
drift is visible.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import run_extract_one
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_intermediate_run_dir,
    phi_patient_run_dir,
)


def _materialize_scratch_corpus(
    *, source_run_id: str, scratch_run_id: str, patient_id: str, dr: Path | None = None
) -> Path:
    """Link the patient's read-only corpus (ingest/embed/index artifacts) from the sealed
    source run into the scratch run, so a comparison extraction can read the same corpus
    while writing its own outputs under the scratch run_id.

    ``extract/`` and ``retrieve/`` are skipped on purpose: they are written fresh per model
    under the scratch run, so the source run's sealed extract artifacts are neither read nor
    overwritten. The scratch run needs this patient dir linked in, or ``run_extract_one``
    raises ``FileNotFoundError``."""
    src = phi_patient_run_dir(source_run_id, patient_id, dr)
    dst = phi_patient_run_dir(scratch_run_id, patient_id, dr)
    dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for entry in sorted(src.iterdir()):
            if entry.name in {"extract", "retrieve"}:
                continue
            link = dst / entry.name
            if not link.exists():
                link.symlink_to(entry.resolve())
    return dst

def compare_models(
    *,
    cfg: dict,
    patient_id: str,
    variable: str,
    model_names: list[str],
) -> dict[str, Any]:
    """Run ``variable`` once per endpoint name; report per-model outputs + fingerprints."""
    outputs: list[dict[str, Any]] = []
    for name in model_names:
        subcfg = copy.deepcopy(cfg)
        subcfg["recipes"] = [variable]
        # Runner consults model_override before recipe.llm.model, so the recipe file is untouched.
        subcfg["model_override"] = name
        # run each model under its own scratch run_id so it never overwrites the
        # sealed extract artifacts (or the previous model's outputs) in place.
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        scratch_run_id = f"{cfg['run_id']}__compare__{safe_name}"
        subcfg["run_id"] = scratch_run_id
        # The corpus lives under the ORIGINAL run_id; link it into the scratch run so
        # extraction can read it while its outputs land under the scratch run_id.
        _materialize_scratch_corpus(
            source_run_id=cfg["run_id"], scratch_run_id=scratch_run_id, patient_id=patient_id
        )
        # force: this comparison exists to run the SAME variable under a different
        # endpoint, and the scratch run id is derived from the source run, so it is the
        # same id every time. Without this, extract's resume sees the previous model's
        # answer sitting there and skips — and every model after the first is reported
        # with the first one's values and the first one's receipts.
        result = run_extract_one(cfg=subcfg, patient_id=patient_id, force=True)
        var_result = result["variables"].get(variable) or {}
        fingerprint = _read_fingerprint(
            phi_intermediate_run_dir(scratch_run_id), patient_id, variable
        )
        outputs.append({
            "endpoint_name": name,
            "ok": var_result.get("ok"),
            "data": var_result.get("data"),
            "resolved_model_fingerprint": fingerprint,
        })

    return {
        "patient_id": patient_id,
        "variable": variable,
        "outputs": outputs,
    }

def _read_fingerprint(run_root: Path, patient_id: str, variable: str) -> str | None:
    """Return resolved_model.fingerprint from the first receipt that has one."""
    stages_dir = run_root / "patients" / patient_id / "extract" / variable / "steps"
    if not stages_dir.is_dir():
        return None
    for receipt_path in sorted(stages_dir.glob("*/receipt.json")):
        try:
            env = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rm = (env.get("payload") or {}).get("resolved_model") or {}
        fp = rm.get("fingerprint")
        if fp:
            return fp
    return None
