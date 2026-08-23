"""The raw patient roster is kept separate from the low-sensitivity manifest.

The manifest (labeled sensitivity:low) carries only a count; the raw ids live in a
PHI-side run_roster.json that is never exported.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
    build_manifest,
    build_roster,
    write_roster,
)

CODE_HASH = "sha256:" + "0" * 64


def test_manifest_carries_a_count_not_the_ids():
    env = build_manifest(
        run_id="20260617_000000", code_lock_hash=CODE_HASH,
        entry_point_name="t", config_alias="t", target_patients=["STSS0123abc", "STSS0456def"],
    )
    payload = env["payload"]
    assert "target_patients" not in payload  # raw ids never in the low-sensitivity manifest
    assert payload["n_target_patients"] == 2
    # the schema (additionalProperties:false) is validated inside build_manifest, so a
    # manifest carrying target_patients would have raised before reaching here.


def test_roster_holds_the_ids_phi_side(tmp_path):
    roster = build_roster(run_id="20260617_000000", target_patients=["STSS0123abc"])
    assert roster["sensitivity"] == "high"
    assert roster["target_patients"] == ["STSS0123abc"]
    out = write_roster(tmp_path, roster)
    assert out.name == "run_roster.json" and out.is_file()
