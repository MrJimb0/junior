"""Integration smoke test for the spine: ingest -> embed -> index -> retrieve ->
rerank -> evidence packet -> extract -> result.json on the shipped Test_Patient
fixture. Slow: needs torch + transformers + the local model weights under models/.
Skips cleanly when they're absent (e.g. CI without GPUs).

Exercises ALL THREE local models: the embedding encoder
(date_of_birth + reranker_smoke retrieval), the extraction LLM (both recipes), and
the gte cross-encoder reranker (reranker_smoke sets reranking.cross_encoder=true).
A green run is impossible without the cross-encoder firing -- the reranker_smoke
rerank receipt must record the candidate ranker with cross_encoder on and finite
per-candidate logits, and models/reranking is required, not optional.
"""
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"

pytestmark = pytest.mark.slow


def _missing() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return "torch not installed"
    if importlib.util.find_spec("transformers") is None:
        return "transformers not installed"
    if _first_model_dir(MODELS / "embedding") is None:
        return "no local embedding model weights"
    if _first_model_dir(MODELS / "extraction") is None:
        return "no local extraction model weights"
    # the cross-encoder is required, not optional -- a green spine must
    # exercise it. Resolve the gte reranker through the registry entry's path.
    if _first_model_dir(MODELS / "reranking") is None:
        return "no local reranking (cross-encoder) model weights"
    return None


def _first_model_dir(parent: Path):
    """The model directory under ``parent``, ignoring macOS .DS_Store / hidden noise
    (iterdir order is arbitrary, so a stray dotfile must not be picked as the model)."""
    if not parent.is_dir():
        return None
    return next(
        (p for p in sorted(parent.iterdir()) if p.is_dir() and not p.name.startswith(".")),
        None,
    )


def test_toy_patient_spine(tmp_path):
    reason = _missing()
    if reason:
        pytest.skip(reason)

    from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
        find_unprovenanced_value_paths,
    )
    from jr_pipeline.runtime_infrastructure.cohort_runner import CohortSettings, run_cohort
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_patient_run_dir,
    )

    prior = os.environ.get("JR_DATA_ROOT")  # run_cohort mutates this process-wide
    try:
        settings = CohortSettings(
            project="smoke",
            input_folder=REPO / "examples",
            data_root=tmp_path / "data",
            patients=["Test_Patient"],
            # date_of_birth exercises embed+extract; reranker_smoke additionally routes
            # through the gte cross-encoder so all three models fire.
            variables=["date_of_birth", "reranker_smoke"],
            embedding_model_path=_first_model_dir(MODELS / "embedding"),
            llm_local_model_path=_first_model_dir(MODELS / "extraction"),
        )
        result = run_cohort(settings, security_check=False)

        assert "Test_Patient" in result.ingested
        pdir = phi_patient_run_dir(result.run_id, "Test_Patient", tmp_path / "data")
        result_json = pdir / "extract" / "date_of_birth" / "result.json"
        assert result_json.is_file(), "extract produced no result.json"
        # the carefully-prepared evidence packet was persisted
        assert (pdir / "prepared_evidence_text").is_dir()
        # the run left a complete audit record
        run_root = pdir.parents[1]
        assert (run_root / "manifest.json").is_file()
        assert (run_root / "summary.json").is_file()

        # the PHI-side result.json carries a raw evidence_chunk_id on
        # every value-bearing field -- the recursive provenance validator finds none
        # unprovenanced. Re-run it AND check the value step 8 recorded.
        dob_payload = json.loads(result_json.read_text(encoding="utf-8"))["payload"]
        assert dob_payload["unprovenanced_value_paths"] == []
        assert find_unprovenanced_value_paths(dob_payload["data"]) == []

        # reranker_smoke routed through the cross-encoder. The rerank receipt
        # records the candidate ranker with cross_encoder on, and every candidate
        # has a FINITE cross_encoder_logit -- a green spine cannot skip the third model.
        rr_receipt = (
            pdir / "extract" / "reranker_smoke" / "steps" / "retrieve_rerank" / "receipt.json"
        )
        assert rr_receipt.is_file(), "reranker_smoke step produced no receipt"
        rerank = json.loads(rr_receipt.read_text(encoding="utf-8"))["payload"]["rerank"]
        assert rerank["kind"] == "composed", f"reranker was {rerank['kind']!r}, not composed"
        # config["cross_encoder"] is the wrapped scorer's identity dict when on, False when off
        assert rerank["config"]["cross_encoder"], "cross-encoder was not enabled"
        assert rerank["config"]["cross_encoder"]["model_path"], "cross-encoder config missing model_path"
        cands = rerank["candidates"]
        assert cands, "cross-encoder reranked zero candidates (model never exercised)"
        for c in cands:
            logit = c["features"]["cross_encoder_logit"]
            assert isinstance(logit, (int, float)) and math.isfinite(logit), (
                f"candidate {c['chunk_id']} has a non-finite cross_encoder_logit {logit!r}"
            )

        # === the NO_PHI exhaust layer ===
        import polars as pl

        from jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata import (
            export_run_metadata,
        )
        from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
            no_phi_exhaust_dir,
            no_phi_manifest_path,
            phi_intermediate_run_dir,
        )

        dr = tmp_path / "data"
        # the flywheel ignition: a schema-validated extraction_outcome parquet with >0 rows
        eo_parquet = no_phi_exhaust_dir(result.run_id, dr) / "extraction_outcome.parquet"
        assert eo_parquet.is_file(), "no extraction_outcome.parquet emitted"
        assert pl.read_parquet(eo_parquet).height > 0, "extraction_outcome.parquet has no rows"
        # the shareable NO_PHI exhaust manifest, and the raw roster split out PHI-side
        assert no_phi_manifest_path(result.run_id, dr).is_file(), "no NO_PHI exhaust manifest"
        assert (phi_intermediate_run_dir(result.run_id, dr) / "run_roster.json").is_file(), (
            "no PHI-side run_roster.json"
        )
        # schema-aware PHI-containment over the WHOLE NO_PHI tree (incl the exhaust
        # parquet + shards): export scans every file and raises on any raw id / clinical
        # date / text. A clean export is the containment proof.
        export_run_metadata(run_id=result.run_id, output_path=tmp_path / "bundle.zip", dr=dr)
    finally:
        if prior is not None:
            os.environ["JR_DATA_ROOT"] = prior
        else:
            os.environ.pop("JR_DATA_ROOT", None)
