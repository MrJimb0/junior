"""per_recipe scoped hashes are keyed "<variable>/<version>"; the consumers must resolve
a variable to its entry (a bare-name lookup would null every sealed receipt's
recipe/prompt/schema sub-hash). The diff tool must read the real run_root/patients/
layout, not run_root/data/patients/."""
import json
from pathlib import Path

import pytest

from jr_pipeline.evaluating_pipeline_performance.compare_run_outputs import (
    _load_result,
    _per_recipe_for,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility import (
    code_bundle_component_fingerprints as cbcf,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_bundle_component_fingerprints import (
    assert_scoped_hashes_grounded,
    per_recipe_hashes,
    retrieval_config_hash,
)

REPO = Path(__file__).resolve().parents[2]
RECIPES = REPO / "var_extraction_recipes"


def test_per_recipe_key_resolves_by_variable_name():
    hashes = per_recipe_hashes(RECIPES)
    assert hashes, "no recipe hashes computed"
    # producer keys are versioned
    assert any(k.startswith("date_of_diagnosis/") for k in hashes)
    # the consumer resolves the bare variable name to a real (non-null) entry
    entry = _per_recipe_for({"per_recipe": hashes}, "date_of_diagnosis")
    assert entry.get("recipe_hash"), "scoped recipe_hash resolved to null"


def test_per_recipe_for_bare_key_fallback():
    legacy = {"per_recipe": {"date_of_birth": {"recipe_hash": "sha256:abc"}}}
    assert _per_recipe_for(legacy, "date_of_birth")["recipe_hash"] == "sha256:abc"


def test_diff_tool_reads_real_patient_layout(tmp_path):
    run_root = tmp_path / "run"
    rj = run_root / "patients" / "p1" / "extract" / "date_of_diagnosis" / "result.json"
    rj.parent.mkdir(parents=True, exist_ok=True)
    rj.write_text(json.dumps({"payload": {"ok": True, "data": {"x": 1}}}), encoding="utf-8")

    loaded = _load_result(run_root, "p1", "date_of_diagnosis")
    assert loaded.get("data") == {"x": 1}, "diff tool read phantom data/patients/ layout"
    # the phantom path must NOT resolve
    assert not (run_root / "data" / "patients").exists()


def test_retrieval_config_hash_is_grounded_in_real_keys():
    # the hash must key on real config keys (encoder/index), not keys nothing sets.
    cfg = {"encoder": {"model_id": "m"}, "index": {"M": 32}}
    grounded = retrieval_config_hash(cfg)
    assert grounded is not None
    # config drift changes the hash, so the seal-drift gate has teeth
    assert retrieval_config_hash({"encoder": {"model_id": "OTHER"}, "index": {"M": 32}}) != grounded


def test_assert_passes_for_a_grounded_config():
    assert_scoped_hashes_grounded({"encoder": {"model_id": "m"}, "index": {"M": 32}})


def test_seal_fails_when_a_configured_stage_hashes_none(monkeypatch):
    # if the subset keys ever drift out of the real config (hash None while the stage
    # IS configured), the seal must fail loud rather than silently going inert.
    monkeypatch.setattr(cbcf, "retrieval_config_hash", lambda cfg: None)
    with pytest.raises(ValueError, match="mis-grounded"):
        cbcf.assert_scoped_hashes_grounded({"encoder": {"model_id": "x"}})
