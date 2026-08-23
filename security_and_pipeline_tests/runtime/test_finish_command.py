"""`junior finish` closes a run out: summarize, then check it end to end.

It is one command because those three answer different questions but are wanted at the
same moment. Two properties matter beyond convenience: it must summarize even when the
checks then fail (a run that finished badly still has to stop looking like it is
running), and it must not report success when a check failed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from apps_and_interfaces.command_line_interface import finish, main


def _invoke(monkeypatch, tmp_path: Path, *, validate_fails=False, verify_fails=False):
    """Run `finish` with the three commands it wraps replaced by recorders."""
    order: list[str] = []

    def _step(label: str, fails: bool):
        def _run(run_root):
            order.append(label)
            if fails:
                raise SystemExit(1)
        return _run

    monkeypatch.setattr(main.commands["summarize"], "callback", _step("summarize", False))
    monkeypatch.setattr(main.commands["validate"], "callback", _step("validate", validate_fails))
    monkeypatch.setattr(main.commands["verify"], "callback", _step("verify", verify_fails))

    result = CliRunner().invoke(main, ["finish", "--run-root", str(tmp_path)])
    return result, order


def test_finish_runs_all_three_in_order(monkeypatch, tmp_path):
    result, order = _invoke(monkeypatch, tmp_path)
    assert order == ["summarize", "validate", "verify"]
    assert result.exit_code == 0, result.output


def test_finish_summarizes_even_when_a_check_fails(monkeypatch, tmp_path):
    """Summarizing is what flips the run off 'running'; skipping it on a failed check
    would leave a finished run looking like it is still going."""
    _, order = _invoke(monkeypatch, tmp_path, validate_fails=True)
    assert order[0] == "summarize"


def test_finish_keeps_going_so_one_report_covers_everything(monkeypatch, tmp_path):
    _, order = _invoke(monkeypatch, tmp_path, validate_fails=True)
    assert "verify" in order, "a failed validate stopped verify from running"


def test_finish_fails_when_a_check_fails(monkeypatch, tmp_path):
    for failing in ({"validate_fails": True}, {"verify_fails": True}):
        result, _ = _invoke(monkeypatch, tmp_path, **failing)
        assert result.exit_code != 0, f"finish reported success despite {failing}"


def test_closing_out_is_not_something_to_choose():
    """Extraction closes the run out, so none of these is on the listed surface."""
    from apps_and_interfaces.command_line_interface import APP_SIDE_COMMANDS

    for name in ("finish", "summarize", "validate", "verify"):
        assert name in APP_SIDE_COMMANDS, f"{name} is being advertised as a step to run"
        assert main.commands[name].hidden
        assert name in main.commands, f"{name} was removed rather than hidden"


def _extract_run(monkeypatch, tmp_path, extra_arguments: list[str]) -> tuple[object, list[Path]]:
    """Invoke `extract` with the pipeline stubbed out; report whether it closed out."""
    closed: list[Path] = []
    monkeypatch.setattr(
        main.commands["finish"], "callback", lambda run_root: closed.append(run_root)
    )
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface._ensure_sealed_bundle",
        lambda **_: "sha256:" + "0" * 64,
    )
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.STAGES",
        {"extract": lambda *a, **k: {"patient_id": "P1"}},
    )
    monkeypatch.setattr(
        "apps_and_interfaces.command_line_interface.patients_for_stage",
        lambda stage, cfg, run_root: ["P1", "P2"],
    )
    config = tmp_path / "c.yaml"
    # Names a real variable: extract refuses a config that does not say what to pull
    # out, and a fixture without one would exercise a path no real run can reach.
    config.write_text(
        f"run_id: R1\noutput_root: {tmp_path / 'data'}\nrecipes:\n  - date_of_birth\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["extract", "--config", str(config), *extra_arguments])
    return result, closed


def test_a_whole_cohort_extract_closes_the_run_out(monkeypatch, tmp_path):
    """The last stage finishes the run, so the operator never has to remember to."""
    result, closed = _extract_run(monkeypatch, tmp_path, [])
    assert result.exit_code == 0, result.output
    assert closed, "a cohort-wide extract did not close the run out"


def test_a_single_patient_extract_does_not_close_the_run_out(monkeypatch, tmp_path):
    """--patient means this is one task of a fan-out. Its siblings are still running,
    so summarizing would be premature and another patient's bad artifact would fail
    this task for something it did not do."""
    result, closed = _extract_run(monkeypatch, tmp_path, ["--patient", "P1"])
    assert result.exit_code == 0, result.output
    assert not closed, "a single-patient extract closed out the whole run"


def test_finish_command_exists_and_documents_itself():
    assert finish.help and "close" in finish.help.lower()


def test_a_cancelled_stage_never_reports_done(capsys):
    """The progress line printed "done in Xs" from a bare finally, so Ctrl-C during a
    stage claimed the patient finished when nothing had been written — the worst thing
    a progress display can say."""
    from apps_and_interfaces.command_line_interface import _stage_monitor

    with pytest.raises(KeyboardInterrupt):
        with _stage_monitor("embed", "P1", Path("/nonexistent"), None, (2, 5)):
            raise KeyboardInterrupt

    reported = capsys.readouterr().err
    assert "done in" not in reported, "a cancelled stage reported success"
    # "Nothing was written" was itself false for extract, which records each
    # finished variable as it goes — the honest claim is scoped to work in flight.
    assert "was not recorded" in reported
    assert "kept" in reported, "does not say finished work survives"


def test_a_failed_stage_never_reports_done(capsys):
    from apps_and_interfaces.command_line_interface import _stage_monitor

    with pytest.raises(RuntimeError):
        with _stage_monitor("embed", "P1", Path("/nonexistent"), None, (1, 3)):
            raise RuntimeError("boom")

    assert "done in" not in capsys.readouterr().err
