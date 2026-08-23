"""Guardrail: verify no PHI leaks into the NO_PHI shareable directory.

Scans every file in NO_PHI__shareable/ for patterns that indicate
patient-derived content: patient IDs, names, dates of birth, MRNs,
clinical text fragments, chunk text, or any per-patient folder structure.

Run after every pipeline execution or as a CI check:
    pytest tests/guardrails/test_phi_containment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
    PHI_CONTENT_PATTERNS,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    NON_SENSITIVE_LABEL,
    data_root,
)

# PHI content patterns come from the single shared vocabulary.


def _get_no_phi_dir() -> Path:
    return data_root() / NON_SENSITIVE_LABEL


def _scan_file_for_phi(filepath: Path) -> list[str]:
    """Scan a single file for PHI indicator patterns. Returns list of findings."""
    findings = []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:  # a file this cannot read is a finding, not a pass (fail closed)
        return [f"{filepath}: unreadable ({type(e).__name__})"]

    for pattern, _label in PHI_CONTENT_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            findings.append(
                f"{filepath}: matched pattern '{pattern.pattern}' "
                f"({len(matches)} occurrences)"
            )

    return findings


def _scan_parquet_for_phi(filepath: Path) -> list[str]:
    """Scan an exhaust parquet by reading every cell as text: the live-tree guard
    must cover the NO_PHI/exhaust/<type>.parquet tables, not just text files."""
    import polars as pl

    try:
        blob = json.dumps(pl.read_parquet(filepath).to_dicts(), default=str)
    except Exception as e:  # corrupt/unreadable parquet is itself a finding (fail closed)
        return [f"{filepath}: unreadable parquet ({type(e).__name__})"]
    return [
        f"{filepath}: matched pattern '{pattern.pattern}'"
        for pattern, _label in PHI_CONTENT_PATTERNS
        if pattern.findall(blob)
    ]


def _scan_for_patient_folders(no_phi_dir: Path) -> list[str]:
    """Check that no per-patient folder structure exists in NO_PHI."""
    findings = []
    for path in no_phi_dir.rglob("*"):
        if "patients" in path.parts:
            findings.append(f"Per-patient path found in NO_PHI: {path}")
    return findings


# --- the scanners, proved against a tree built for the purpose ----------------------
#
# Everything below this line used to be the whole file, and all of it skipped when the
# machine had no NO_PHI directory — which is every fresh checkout, and every CI job.
# It passed for months on whatever runs happened to be lying in the developer's own
# data/, and the moment that folder was cleared it went quiet rather than red.
#
# A scanner that is never shown a positive case is a scanner nobody has checked. These
# build a tree, plant PHI in it, and require it to be found; the live-tree scan below
# is then a bonus over whatever the machine has, not the guarantee.

@pytest.fixture
def a_no_phi_tree(tmp_path):
    """A shareable tree shaped like the real one: run-level metadata, no patient data."""
    tree = tmp_path / NON_SENSITIVE_LABEL / "20260101_000000_aa"
    (tree / "exhaust").mkdir(parents=True)
    (tree / "summary.json").write_text(json.dumps(
        {"run_id": "20260101_000000_aa", "n_patients": 2, "status": "completed"}), encoding="utf-8")
    (tree / "exhaust" / "extraction_outcome.jsonl").write_text(json.dumps(
        {"patient_surrogate": "surr:v1:run:patient_surrogate:8fcf17c6c5cbeef8",
         "ok": True, "variable_key": "date_of_birth"}) + "\n", encoding="utf-8")
    return tmp_path / NON_SENSITIVE_LABEL


def test_a_clean_shareable_tree_passes(a_no_phi_tree):
    """Surrogates, counts and run ids are what belongs there, and none of it trips."""
    findings = _scan_for_patient_folders(a_no_phi_tree)
    for f in a_no_phi_tree.rglob("*"):
        if f.is_file():
            findings += _scan_file_for_phi(f)
            if f.suffix == ".json":
                _check_dict_for_patient_id(json.loads(f.read_text()), str(f), findings)
    assert findings == [], findings


def test_a_planted_patient_id_is_caught(a_no_phi_tree):
    """The failure the two-stream design exists to prevent, and the detector for it had
    never once been shown a positive case."""
    leaked = a_no_phi_tree / "20260101_000000_aa" / "leaked.json"
    leaked.write_text(json.dumps({"patient_id": "STSS10c40439"}), encoding="utf-8")

    findings: list[str] = []
    _check_dict_for_patient_id(json.loads(leaked.read_text()), str(leaked), findings)

    assert findings, "a raw patient id in the shareable tree went unnoticed"


def test_a_planted_patient_id_nested_deep_is_caught(a_no_phi_tree):
    """Buried, because that is how one would actually arrive — inside a record inside a
    list, not at the top of a file somebody would notice."""
    findings: list[str] = []
    _check_dict_for_patient_id(
        {"run": {"records": [{"ok": True}, {"patient_id": "STSS10c40439"}]}}, "x", findings)

    assert findings, "a patient id nested in a record went unnoticed"


def test_planted_chart_text_is_caught(a_no_phi_tree):
    """Content, not just identifiers: a chunk of note text carries the patient whether
    or not their id is beside it."""
    leaked = a_no_phi_tree / "20260101_000000_aa" / "notes.txt"
    leaked.write_text("Chief Complaint: chest pain\nHistory of Present Illness: ...",
                      encoding="utf-8")

    assert _scan_file_for_phi(leaked), "clinical note text in the shareable tree went unnoticed"


def test_a_planted_per_patient_folder_is_caught(a_no_phi_tree):
    """Structure leaks as surely as content: a folder named for a patient says who was
    in the cohort even when every file inside it is clean."""
    (a_no_phi_tree / "20260101_000000_aa" / "patients" / "STSS10c40439").mkdir(parents=True)

    assert _scan_for_patient_folders(a_no_phi_tree), "a per-patient folder went unnoticed"


class TestPHIContainment:
    """Scan whatever NO_PHI tree this machine actually has, if it has one.

    A bonus pass over real output, not the guarantee — the tests above are that. It
    skips on a fresh checkout, which is correct here: there is genuinely nothing to
    scan, and the scanners have already been proved."""

    def test_no_phi_dir_has_no_patient_folders(self):
        no_phi = _get_no_phi_dir()
        if not no_phi.exists():
            pytest.skip("NO_PHI directory does not exist yet")
        findings = _scan_for_patient_folders(no_phi)
        assert findings == [], (
            "Per-patient data found in NO_PHI:\n" +
            "\n".join(findings)
        )

    def test_no_phi_files_contain_no_phi_patterns(self):
        no_phi = _get_no_phi_dir()
        if not no_phi.exists():
            pytest.skip("NO_PHI directory does not exist yet")

        all_findings = []
        for f in no_phi.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix == ".parquet":  # exhaust tables are scanned too
                all_findings.extend(_scan_parquet_for_phi(f))
            elif f.suffix in (".json", ".jsonl", ".txt", ".csv", ".yaml", ".yml"):
                all_findings.extend(_scan_file_for_phi(f))

        assert all_findings == [], (
            "PHI indicators found in NO_PHI files:\n" +
            "\n".join(all_findings)
        )

    def test_no_phi_json_files_have_no_patient_ids(self):
        """Check JSON files for patient_id fields with actual values."""
        no_phi = _get_no_phi_dir()
        if not no_phi.exists():
            pytest.skip("NO_PHI directory does not exist yet")

        findings = []
        for f in no_phi.rglob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
                # Fail closed, as the parquet scanner does: json this cannot parse may
                # still carry ids, and skipping it silently reports the tree as clean.
                findings.append(f"{f}: unreadable json ({type(e).__name__})")
                continue
            # A JSON Schema names patient_id to DECLARE the field, e.g.
            # properties.patient_id = {"type": "string"}. Every sealed run ships the
            # scrubbed code bundle here, schemas included, so walking those as data
            # would report the pipeline's own field definitions as leaked ids.
            if isinstance(data, dict) and "$schema" in data:
                continue
            _check_dict_for_patient_id(data, str(f), findings)

        assert findings == [], (
            "patient_id fields found in NO_PHI JSON:\n" +
            "\n".join(findings)
        )


def _check_dict_for_patient_id(obj, filepath: str, findings: list) -> None:
    """Recursively check a dict/list for patient_id keys carrying an actual id."""
    if isinstance(obj, dict):
        value = obj.get("patient_id")
        # Only a scalar is an identifier. A dict or list under this key is structure
        # (a schema node, a per-patient mapping) whose leaves the walk below reaches
        # anyway — flagging the container would report the shape, not a value.
        if isinstance(value, str | int) and value:
            findings.append(f"{filepath}: contains patient_id = {value!r}")
        for v in obj.values():
            _check_dict_for_patient_id(v, filepath, findings)
    elif isinstance(obj, list):
        for item in obj:
            _check_dict_for_patient_id(item, filepath, findings)


class TestLogPHISafety:
    """Verify the logging module enforces PHI-safe patterns."""

    def test_structlog_configured_with_no_patient_text(self):
        """The logger module must not allow raw patient text in log messages."""
        from pathlib import Path
        log_file = Path(__file__).parent.parent.parent / "src/jr_pipeline/runtime_infrastructure/json_event_logging.py"
        content = log_file.read_text()
        assert "structlog" in content, "Logger must use structlog, not custom formatting"

    def test_no_phi_logs_policy_documented(self):
        """The layout module must document log PHI policy."""
        from pathlib import Path
        layout_file = Path(__file__).parent.parent.parent / "src/jr_pipeline/runtime_infrastructure/data_directory_layout_and_safe_writes.py"
        content = layout_file.read_text()
        assert "run-level logs" in content.lower() or "no patient" in content.lower(), (
            "Layout module must document that NO_PHI logs contain no patient data"
        )


def test_parquet_scan_flags_phi_and_passes_clean(tmp_path):
    """The parquet-aware scan flags a full clinical date in an exhaust cell and
    passes a surrogate/vocab-only table (exercised here, not skip-vacuous)."""
    import polars as pl

    dirty = tmp_path / "extraction_outcome.parquet"
    pl.DataFrame([{"patient_surrogate": "surr:v1:run:p:" + "a" * 24, "note": "seen 2026-01-15"}]).write_parquet(dirty)
    assert _scan_parquet_for_phi(dirty), "a full clinical date in a parquet cell must be flagged"

    clean = tmp_path / "clean.parquet"
    pl.DataFrame([{"patient_surrogate": "surr:v1:run:p:" + "a" * 24, "doc_type": "pathology"}]).write_parquet(clean)
    assert _scan_parquet_for_phi(clean) == []
