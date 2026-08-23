"""iter_transitions includes fragments written after the first merge.

A merged file plus any state fragment written after the merge together form the
full history; iter_transitions yields both so the merged file alone is never
treated as the complete record.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    iter_transitions,
    merge_state_fragments,
    record_transition,
)

CODE_HASH = "sha256:" + "0" * 64


def _record(run_root, to_state):
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id="r1", patient_id="p1", step="extract"),
        from_state=None, to_state=to_state, reason=to_state,
        step_context="extract", code_lock_hash=CODE_HASH,
    )


def test_fragment_after_merge_is_not_missed(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    _record(run_root, "running")
    merge_state_fragments(run_root)   # merged file now holds the "running" transition
    _record(run_root, "completed")    # written AFTER the merge -> must still be yielded

    transitions = list(iter_transitions(run_root))
    assert len(transitions) == 2, f"expected both transitions, got {len(transitions)}"
    states = {t["payload"]["to_state"] for t in transitions}
    assert states == {"running", "completed"}
