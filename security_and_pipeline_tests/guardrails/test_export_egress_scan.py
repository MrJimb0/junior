"""The export gateway re-scans the NO_PHI tree at the egress moment and aborts
on any forbidden content, so a leaked file cannot ship cross-institution."""
import pytest

from jr_pipeline.evaluating_pipeline_performance.export_shareable_metadata import (
    export_run_metadata,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    no_phi_run_dir,
)

RUN = "20260101_000000_aa"


def _seed_clean(tmp_path):
    nd = no_phi_run_dir(RUN, tmp_path)
    (nd / "evidence_selection").mkdir(parents=True, exist_ok=True)
    # legit NO_PHI content: surrogates, scores, the MONTH stamp (not a full date)
    (nd / "evidence_selection" / "abc123.json").write_text(
        '{"site_id": "s1", "emitted_month": "2026-01", "score": 1.5, "chunk_surrogate": "deadbeef"}',
        encoding="utf-8",
    )
    return nd


def test_clean_tree_exports(tmp_path):
    _seed_clean(tmp_path)
    out = export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)
    assert out.exists()


def test_absolute_date_aborts_export(tmp_path):
    nd = _seed_clean(tmp_path)
    (nd / "leak.json").write_text('{"document_date": "2024-02-15"}', encoding="utf-8")  # HIPAA date
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_mrn_aborts_export(tmp_path):
    nd = _seed_clean(tmp_path)
    (nd / "leak.txt").write_text("Patient MRN: 9999999", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_per_patient_folder_aborts_export(tmp_path):
    nd = _seed_clean(tmp_path)
    (nd / "patients" / "p1").mkdir(parents=True)
    (nd / "patients" / "p1" / "x.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_binary_file_aborts_export_fail_closed(tmp_path):
    # a non-UTF-8 file the scanner can't read as text must FAIL CLOSED, so binary
    # that can't be scanned never passes as empty text.
    nd = _seed_clean(tmp_path)
    (nd / "secret.parquet").write_bytes(b"\x89PAR1\x00\xff\xfe\x00binary\x80\x81PAR1")
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_target_patient_id_in_no_phi_aborts_export(tmp_path):
    # a target patient id appearing in a NO_PHI file is a leak, detected from the
    # PHI-side run_roster.json (the roster, not the low-sensitivity manifest, holds ids).
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
        build_roster,
        write_roster,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    nd = _seed_clean(tmp_path)
    phi_root = phi_intermediate_run_dir(RUN, tmp_path)
    phi_root.mkdir(parents=True, exist_ok=True)
    write_roster(phi_root, build_roster(run_id=RUN, target_patients=["STSS0123abc"]))
    (nd / "leak.json").write_text('{"note": "value for patient STSS0123abc"}', encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_sealed_code_clinical_literal_passes_export(tmp_path):
    # the sealed code bundle is METHOD (source + prompts). Clinical terms and
    # example dates in it are not patient data, so the stream-aware egress does NOT apply
    # the clinical-content patterns to sealed_code/ -- a "Date of Birth:" prompt ships.
    nd = _seed_clean(tmp_path)
    sealed = nd / "sealed_code" / "recipes" / "date_of_birth"
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / "prompt.md").write_text(
        "Extract the Date of Birth: field. Example: 1957-04-15 -> 1957-04-15.\n", encoding="utf-8"
    )
    out = export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)
    assert out.exists()


def test_sealed_code_phi_path_still_aborts_export(tmp_path):
    # but the PHI path/id scrub STILL applies to code stream: a real PHI path in a
    # sealed config is a leak even though it's under sealed_code/.
    nd = _seed_clean(tmp_path)
    sealed = nd / "sealed_code"
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / "config_resolved.yaml").write_text(
        "model_path: /oak/stanford/raw/patients/notes\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)


def test_data_stream_clinical_literal_still_aborts(tmp_path):
    # the exemption is scoped to sealed_code/ only: the same literal in a DATA file aborts.
    nd = _seed_clean(tmp_path)
    (nd / "notes.json").write_text('{"text": "Date of Birth: 1957-04-15"}', encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden content"):
        export_run_metadata(run_id=RUN, output_path=tmp_path / "out.zip", dr=tmp_path)
