"""Patient-derived extracted values are REDACTED from stdout by default. Opt in with
JR_SHOW_STDOUT_VALUES=1 (a trusted terminal / synthetic demo); JR_REDACT_STDOUT forces
redaction."""
from __future__ import annotations

import json
import types

from jr_pipeline.runtime_infrastructure.cohort_runner import _stdout_values_allowed, view_results
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    extract_output_dir,
    phi_patient_run_dir,
)


def test_stdout_gate_matrix(monkeypatch):
    monkeypatch.delenv("JR_SHOW_STDOUT_VALUES", raising=False)
    monkeypatch.delenv("JR_REDACT_STDOUT", raising=False)
    assert _stdout_values_allowed() is False  # default: redact

    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    assert _stdout_values_allowed() is True

    monkeypatch.setenv("JR_REDACT_STDOUT", "1")  # the redact flag wins
    assert _stdout_values_allowed() is False


def _seed_result(run_id: str, pid: str, var: str, phi_value: str) -> None:
    rdir = extract_output_dir(phi_patient_run_dir(run_id, pid)) / var
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "result.json").write_text(json.dumps({"payload": {"data": {"value": phi_value}}}))


def test_view_results_redacts_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("JR_SHOW_STDOUT_VALUES", raising=False)
    monkeypatch.delenv("JR_REDACT_STDOUT", raising=False)
    phi = "Bassett, Kitty MRN 12345"
    _seed_result("R", "P1", "dob", phi)

    result = types.SimpleNamespace(ingested=["P1"], run_id="R")
    settings = types.SimpleNamespace(variables=["dob"])

    view_results(result, settings)
    out = capsys.readouterr().out
    assert phi not in out
    assert "redacted" in out

    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    view_results(result, settings)
    assert phi in capsys.readouterr().out


def test_error_lines_pass_the_same_gate_as_the_value_line(monkeypatch):
    """A schema-validation message quotes the extracted value, so printing it verbatim
    under a redacted value line was the leak: the value the line above hid arrived one
    line later inside the error. The field name may survive — it is a schema key, not
    chart content — so the operator still learns WHERE without learning WHAT."""
    from jr_pipeline.runtime_infrastructure.cohort_runner import stdout_safe_error

    monkeypatch.delenv("JR_SHOW_STDOUT_VALUES", raising=False)
    monkeypatch.delenv("JR_REDACT_STDOUT", raising=False)
    message = "date_of_birth: '1957-04-15 | 1957-04-15' is not valid under any of the given schemas"

    safe = stdout_safe_error(message)

    assert "1957-04-15" not in safe
    assert "date_of_birth" in safe, "the field name is a schema key and may be shown"
    assert "receipts" in safe, "does not say where the full message went"

    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    assert stdout_safe_error(message) == message

    monkeypatch.setenv("JR_REDACT_STDOUT", "1")  # the redact flag wins
    assert "1957-04-15" not in stdout_safe_error(message)


def test_verbose_summary_dump_carries_no_values(monkeypatch):
    """`junior extract -v` dumps the stage summary as JSON. The summary's `variables`
    entries carry the extracted `data`, so the dump has to pass the same gate as every
    other stdout surface — counts, paths and ok flags survive, values do not."""
    from jr_pipeline.runtime_infrastructure.cohort_runner import redact_summary_for_stdout

    monkeypatch.delenv("JR_SHOW_STDOUT_VALUES", raising=False)
    monkeypatch.delenv("JR_REDACT_STDOUT", raising=False)
    phi = "Bassett, Kitty MRN 12345"
    summary = {
        "patient_id": "P1",
        "n_failed": 1,
        "variables": {
            "dob": {"ok": False, "data": {"value": phi},
                    "errors": [f"dob: '{phi}' is not valid"]},
        },
    }

    scrubbed = redact_summary_for_stdout(summary)
    dumped = json.dumps(scrubbed)

    assert phi not in dumped
    assert scrubbed["patient_id"] == "P1", "operational fields must survive"
    assert scrubbed["n_failed"] == 1
    assert "dob" in dumped, "the field name is a schema key and may be shown"

    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    assert redact_summary_for_stdout(summary) is summary  # opt-in: untouched


def test_the_cohort_extract_loop_prints_no_values_on_failure(tmp_path, monkeypatch, capsys):
    """The run path end to end: a variable that fails validation prints its outcome and
    its errors, and neither may carry the extracted value. Drives _run_extract with a
    stand-in extractor so no model is needed."""
    import types as _types

    import jr_pipeline.pipeline_steps.step_7_extract_variables.extract as extract_module
    from jr_pipeline.runtime_infrastructure import cohort_runner

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("JR_SHOW_STDOUT_VALUES", raising=False)
    phi = "Bassett, Kitty MRN 12345"
    allowlist = tmp_path / "allowlist.yaml"
    allowlist.write_text("allowed_endpoints: []\n", encoding="utf-8")

    def fake_extract(*, cfg, patient_id, code_lock_hash, on_variable=None, **_ignored):
        if on_variable is not None:
            on_variable("dob", "running", None)
            on_variable("dob", "failed", 1.0)
        return {"n_failed": 1, "variables": {
            "dob": {"ok": False, "data": {"value": phi},
                    "errors": [f"dob: '{phi}' is not valid under any of the given schemas"]},
        }}

    monkeypatch.setattr(extract_module, "run_extract_one", fake_extract)
    settings = _types.SimpleNamespace(variables=["dob"])
    cohort_runner._run_extract(["P1"], settings, {"allowlist_path": str(allowlist)})

    out = capsys.readouterr().out
    assert phi not in out, "an extracted value reached stdout on the run path"
    assert "error:" in out and "dob" in out


def test_the_run_button_child_never_inherits_the_show_values_flag(monkeypatch):
    """The app's log panel shows the child's stdout, and its promise is values-free.
    That promise must hold even when the server's own shell exported the opt-in."""
    import sys as _sys
    from pathlib import Path as _Path

    _APP_DIR = _Path(__file__).resolve().parents[2] / "apps_and_interfaces" / "shiny_review_app"
    if str(_APP_DIR) not in _sys.path:
        _sys.path.insert(0, str(_APP_DIR))
    import pipeline_launcher

    monkeypatch.setenv("JR_SHOW_STDOUT_VALUES", "1")
    environment = pipeline_launcher.child_environment("/data/root")

    assert "JR_SHOW_STDOUT_VALUES" not in environment
    assert environment["JR_DATA_ROOT"] == "/data/root"
