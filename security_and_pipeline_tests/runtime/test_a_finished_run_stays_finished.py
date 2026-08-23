"""Re-entering a closed run must not un-finish it, and must not hide that it re-sealed.

Both come from one incident, reconstructed off run 20260820_105033_b74d. A long-lived
interactive session had loaded cohort_runner before the never-re-seal fix landed, so ten
minutes after `finish` had closed the run out, a second entry re-sealed the bundle and
rewrote the header. Two separate things went wrong and only one of them was survivable.

The header reset is harmless in itself — almost nothing reads status or completed_at —
but it is written by the same call that replaced the bundle, and that is not harmless:
build_and_seal rmtrees and recopies code/, so the bundle which produced nine of the ten
results stopped existing. `junior finish` then printed all three checks green, because
verify only re-hashes code/ against its own lock file and nothing compares that against
what the receipts say produced them.
"""
from __future__ import annotations

import json
from pathlib import Path

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
    build_manifest,
    mark_completed,
    write_manifest,
)

A_HASH = "sha256:" + "a" * 64
ANOTHER_HASH = "sha256:" + "b" * 64


def _a_manifest(run_root: Path, code_lock_hash: str = A_HASH):
    return build_manifest(
        run_id="20990101_000000_aa", code_lock_hash=code_lock_hash,
        entry_point_name="test", config_alias="t", target_patients=["P1"],
        model_sha256=None,
    )


def _payload(run_root: Path) -> dict:
    return json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))["payload"]


def test_a_second_entry_does_not_reset_a_finished_run(tmp_path):
    """build_manifest hardcodes status "running" and completed_at None, which is right
    when a run is created and wrong every other time — and it is called on EVERY entry
    into a run, not only the first."""
    write_manifest(tmp_path, _a_manifest(tmp_path))
    mark_completed(tmp_path, status="completed_with_errors")
    finished_at = _payload(tmp_path)["completed_at"]

    write_manifest(tmp_path, _a_manifest(tmp_path))

    after = _payload(tmp_path)
    assert after["status"] == "completed_with_errors"
    assert after["completed_at"] == finished_at


def test_a_run_still_going_is_not_frozen_as_running(tmp_path):
    """Only a FINISHED state is carried forward. A run genuinely still in flight has to
    keep getting its header rewritten, or a resumed run would never record its later
    code_lock_hash or patient count."""
    write_manifest(tmp_path, _a_manifest(tmp_path))

    write_manifest(tmp_path, _a_manifest(tmp_path, code_lock_hash=ANOTHER_HASH))

    after = _payload(tmp_path)
    assert after["status"] == "running"
    assert after["code_lock_hash"] == ANOTHER_HASH


def test_mark_completed_is_still_the_way_to_finish_one(tmp_path):
    write_manifest(tmp_path, _a_manifest(tmp_path))
    assert _payload(tmp_path)["status"] == "running"

    mark_completed(tmp_path, status="completed")

    assert _payload(tmp_path)["status"] == "completed"


def test_verify_reports_results_the_bundle_did_not_produce(tmp_path):
    """The check that would have caught the incident. verify_code_bundle re-hashes code/
    against its own lock file, so a REPLACED bundle verifies perfectly while the results
    it did not produce sit beside it naming a hash that no longer exists."""
    from apps_and_interfaces.command_line_interface import _results_that_name_a_different_bundle

    variable_dir = tmp_path / "patients" / "P1" / "extract" / "stage"
    variable_dir.mkdir(parents=True)
    (variable_dir / "result.json").write_text(json.dumps({
        "payload": {"ok": True}, "produced_by": {"code_lock_hash": A_HASH},
    }), encoding="utf-8")

    counted = _results_that_name_a_different_bundle(tmp_path)

    assert counted == {A_HASH: 1}


def test_verify_says_nothing_when_the_receipts_agree_with_the_bundle(tmp_path):
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    code = tmp_path / "code"
    code.mkdir()
    (code / "code.lock.json").write_text(
        json.dumps({"payload": {"code_lock_hash": A_HASH}}), encoding="utf-8")
    variable_dir = tmp_path / "patients" / "P1" / "extract" / "stage"
    variable_dir.mkdir(parents=True)
    (variable_dir / "result.json").write_text(json.dumps({
        "payload": {"ok": True}, "produced_by": {"code_lock_hash": A_HASH},
    }), encoding="utf-8")

    out = CliRunner().invoke(main, ["verify", "--run-root", str(tmp_path)]).output

    assert "results_from_another_bundle" not in out
