"""The PHI-bearing LLM response cache defaults INSIDE the CONTAINS_PHI run tree,
never ``~/.cache`` outside the PHI seam. An explicit ``llm_cache_path`` override
still wins for an operator-vetted location."""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import resolve_llm_cache_path
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import phi_root


def test_cache_defaults_inside_phi_tree(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    cache = resolve_llm_cache_path({"run_id": "20260101_000000_abcd"})

    # lives under <data_root>/CONTAINS_PHI/..., is run-scoped, and is NOT in ~/.cache.
    assert str(cache).startswith(str(phi_root()))
    assert "20260101_000000_abcd" in cache.parts
    assert cache.name == "llm_cache.db"
    assert (Path.home() / ".cache") not in cache.parents


def test_explicit_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    vetted = tmp_path / "vetted_scratch" / "llm.db"
    cache = resolve_llm_cache_path({"run_id": "R", "llm_cache_path": str(vetted)})
    assert cache == vetted
