"""Every run seals, so every run's code/hashes.json must carry real scoped sub-hashes
(recipe/prompt/schema/retrieval) for extract to stamp on receipts and for
`code scoped-diff` to compare.

If _load_hashes_index returns {} the receipts get null sub-hashes and the comparison
tooling silently comes up empty. The cohort runner's config must therefore reach the
seal with the retrieval keys still on it.
"""
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import _load_hashes_index
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
    build_and_seal,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    validate_artifact,
)
from jr_pipeline.runtime_infrastructure.cohort_runner import (
    CohortSettings,
    _build_step_configs,
    _sealed_config,
)

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "20260101_000000_aa"


def _seal_a_cohort_run(tmp_path: Path) -> Path:
    """Seal exactly the way run_cohort does, and return the run root."""
    settings = CohortSettings(
        project="test_cohort",
        data_root=tmp_path / "data",
        recipes_root=REPO / "var_extraction_recipes",
        llm_mode="openai_compatible",  # skips materializing a local allowlist
    )
    cfgs = _build_step_configs(settings, RUN_ID)
    run_root = tmp_path / "run"
    build_and_seal(
        run_id=RUN_ID,
        run_root=run_root,
        repo_root=REPO,
        cfg=_sealed_config(settings, RUN_ID, cfgs),
        entry_point={"argv_program": "pytest", "argv": [], "step": "seal"},
        recipes_root=Path(settings.recipes_root),
    )
    return run_root


def test_sealed_hashes_index_is_loadable_and_grounded(tmp_path):
    run_root = _seal_a_cohort_run(tmp_path)
    assert (run_root / "code" / "hashes.json").is_file(), "seal did not emit code/hashes.json"

    payload = _load_hashes_index(run_root)
    assert payload.get("code_lock_hash", "").startswith("sha256:")
    # encoder+chunker survive into the sealed cfg -> non-None retrieval hash.
    assert payload.get("retrieval_config_hash", "").startswith("sha256:")
    per_recipe = payload.get("per_recipe") or {}
    assert per_recipe, "per_recipe hashes are empty — extract would null every sub-hash"
    # the index is keyed "<variable>/<version>"; date_of_birth ships in the repo.
    dob = per_recipe.get("date_of_birth/v1")
    assert dob and dob["recipe_hash"].startswith("sha256:")


def test_sealed_hashes_json_validates_against_schema(tmp_path):
    import json

    run_root = _seal_a_cohort_run(tmp_path)
    env = json.loads((run_root / "code" / "hashes.json").read_text(encoding="utf-8"))
    validate_artifact(env, "hashes")  # raises if the envelope is malformed
