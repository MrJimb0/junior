"""A hard extract crash must leave a `failed` state and a closed cache, not
a phantom `running` entity with the LLM cache held open.

Per-step execute() failures are caught inside the loop; this guards the crash
*outside* it (e.g. PatientChunkStore on a missing chunk_index), which would
otherwise propagate out of run_extract_one with the entity stuck `running`.
"""
import pytest

import jr_pipeline.pipeline_steps.step_7_extract_variables.extract as extract_mod

RUN = "20990101_000000"


class _SpyCache:
    """Stand-in for LLMCache that records whether close() ran."""
    instances: list = []

    def __init__(self, **_kwargs):
        self.closed = False
        _SpyCache.instances.append(self)

    def close(self):
        self.closed = True


def _patch_cache_and_transitions(monkeypatch):
    _SpyCache.instances.clear()
    transitions = []
    monkeypatch.setattr(extract_mod, "LLMCache", _SpyCache)
    monkeypatch.setattr(extract_mod, "record_transition", lambda *a, **k: transitions.append(k))
    return transitions


def test_hard_crash_records_failed_and_closes_cache(monkeypatch, tmp_path):
    transitions = _patch_cache_and_transitions(monkeypatch)

    def _boom(**_kwargs):
        raise RuntimeError("missing chunk_index")  # e.g. extract before embed

    monkeypatch.setattr(extract_mod, "_run_extract_one_body", _boom)

    with pytest.raises(RuntimeError, match="missing chunk_index"):
        extract_mod.run_extract_one(
            cfg={"run_id": RUN, "llm_cache_path": str(tmp_path / "c.db")},
            patient_id="Test_Patient",
        )

    assert _SpyCache.instances and _SpyCache.instances[0].closed, "cache not closed on crash"
    assert any(
        k.get("to_state") == "failed" and k.get("from_state") == "running"
        for k in transitions
    ), "no running->failed transition recorded on crash"


def test_success_passes_through_and_closes_cache(monkeypatch, tmp_path):
    _patch_cache_and_transitions(monkeypatch)
    sentinel = {"patient_id": "Test_Patient", "n_failed": 0}
    monkeypatch.setattr(extract_mod, "_run_extract_one_body", lambda **_kw: sentinel)

    out = extract_mod.run_extract_one(
        cfg={"run_id": RUN, "llm_cache_path": str(tmp_path / "c.db")},
        patient_id="Test_Patient",
    )

    assert out is sentinel  # wrapper is transparent on the happy path
    assert _SpyCache.instances[0].closed, "cache not closed on success"
