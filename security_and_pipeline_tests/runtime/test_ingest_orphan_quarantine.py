"""Re-ingest reconciles structured/ with the sources still on disk.

Embedding globs ``structured/*.parquet``, so a parquet left behind after its
source file was removed would feed embedding indefinitely. On a fresh ingest we
reconcile: under ``files: auto`` the orphaned parquet + sidecar are quarantined
(moved out of the glob, never deleted); under an explicit ``files:`` list they
are only warned about (the operator controls that set on purpose).
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_1_ingest_raw_files import ingest
from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import run_ingest_one
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)


def _touch_structured(structured_dir, stem):
    structured_dir.mkdir(parents=True, exist_ok=True)
    (structured_dir / f"{stem}.parquet").write_bytes(b"PAR1")
    (structured_dir / f"{stem}.parquet.meta.json").write_text("{}")


class _RecordingLog:
    def __init__(self):
        self.events = []

    def warning(self, event, extra_=None):
        self.events.append((event, extra_ or {}))


# --- _reconcile_structured_dir: both branches ---

def test_reconcile_auto_quarantines_orphan_and_keeps_live(tmp_path):
    structured = tmp_path / "structured"
    quarantine = tmp_path / "quarantined_structured"
    _touch_structured(structured, "live")
    _touch_structured(structured, "gone")
    log = _RecordingLog()

    result = ingest._reconcile_structured_dir(
        structured_dir=structured, quarantine_dir=quarantine,
        files_cfg="auto", present_source_stems={"live"}, log=log,
    )

    assert result == {"quarantined": ["gone"], "warned": []}
    # the live parquet is untouched
    assert (structured / "live.parquet").is_file()
    assert (structured / "live.parquet.meta.json").is_file()
    # the orphan (parquet + sidecar) left structured/ and lives in quarantine
    assert not (structured / "gone.parquet").exists()
    assert not (structured / "gone.parquet.meta.json").exists()
    assert (quarantine / "gone.parquet").is_file()
    assert (quarantine / "gone.parquet.meta.json").is_file()
    assert any(e[0] == "ingest_quarantined_orphan_parquets" for e in log.events)


def test_reconcile_explicit_list_only_warns_never_moves(tmp_path):
    structured = tmp_path / "structured"
    quarantine = tmp_path / "quarantined_structured"
    _touch_structured(structured, "configured")
    _touch_structured(structured, "stray")
    log = _RecordingLog()

    result = ingest._reconcile_structured_dir(
        structured_dir=structured, quarantine_dir=quarantine,
        files_cfg=[{"stem": "configured"}], present_source_stems={"configured"}, log=log,
    )

    assert result == {"quarantined": [], "warned": ["stray"]}
    # explicit list: the stray parquet is left exactly where it was
    assert (structured / "stray.parquet").is_file()
    assert not quarantine.exists()
    assert any(e[0] == "ingest_orphan_parquets_unmanaged" for e in log.events)


def test_reconcile_orphan_is_source_gone_not_merely_deselected(tmp_path):
    # An orphan is a parquet whose SOURCE is gone (stem absent from
    # present_source_stems), independent of which stems the config selects. A
    # present-but-deselected file (source still on disk) must NOT be flagged,
    # and a configured stem whose source was removed MUST be.
    structured = tmp_path / "structured"
    quarantine = tmp_path / "q"
    _touch_structured(structured, "removed")     # source deleted
    _touch_structured(structured, "deselected")  # source present, not in files:
    log = _RecordingLog()

    result = ingest._reconcile_structured_dir(
        structured_dir=structured, quarantine_dir=quarantine,
        files_cfg=[{"stem": "kept"}],                    # explicit list
        present_source_stems={"deselected", "kept"},     # both still have sources
        log=log,
    )

    assert result == {"quarantined": [], "warned": ["removed"]}
    assert (structured / "deselected.parquet").is_file()  # not flagged
    assert (structured / "removed.parquet").is_file()     # warn-only, never moved
    assert not quarantine.exists()


def test_reconcile_no_orphans_does_not_create_quarantine_dir(tmp_path):
    structured = tmp_path / "structured"
    quarantine = tmp_path / "q"
    _touch_structured(structured, "a")

    result = ingest._reconcile_structured_dir(
        structured_dir=structured, quarantine_dir=quarantine,
        files_cfg="auto", present_source_stems={"a"}, log=_RecordingLog(),
    )

    assert result == {"quarantined": [], "warned": []}
    assert (structured / "a.parquet").is_file()
    assert not quarantine.exists()


# --- the wiring: a real re-ingest quarantines a removed source's parquet ---

def test_run_ingest_one_auto_quarantines_orphan_on_source_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    src_root = tmp_path / "src"
    pdir = src_root / "P1"
    pdir.mkdir(parents=True)
    (pdir / "a.csv").write_text("text\nhello\n")
    (pdir / "b.csv").write_text("text\nworld\n")
    cfg = {"run_id": "20260101_ingest", "source_root": str(src_root), "files": "auto"}

    run_ingest_one(cfg=cfg, patient_id="P1")
    structured = phi_patient_run_dir("20260101_ingest", "P1") / "structured"
    assert (structured / "a.parquet").is_file()
    assert (structured / "b.parquet").is_file()

    # remove source b; re-ingest must quarantine b's orphaned parquet
    (pdir / "b.csv").unlink()
    run_ingest_one(cfg=cfg, patient_id="P1")

    assert (structured / "a.parquet").is_file()
    assert not (structured / "b.parquet").exists()
    quarantine = phi_patient_run_dir("20260101_ingest", "P1") / "quarantined_structured"
    assert (quarantine / "b.parquet").is_file()
    assert (quarantine / "b.parquet.meta.json").is_file()


def test_run_ingest_one_explicit_list_does_not_quarantine_removed_source(tmp_path, monkeypatch):
    # Under an explicit files: list, a removed source is warned about but its
    # parquet is left in place — never quarantined (the operator owns that set).
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    src_root = tmp_path / "src"
    pdir = src_root / "P1"
    pdir.mkdir(parents=True)
    (pdir / "a.csv").write_text("text\nhello\n")
    (pdir / "b.csv").write_text("text\nworld\n")
    cfg = {"run_id": "20260101_exp", "source_root": str(src_root),
           "files": [{"stem": "a"}, {"stem": "b", "optional": True}]}

    run_ingest_one(cfg=cfg, patient_id="P1")
    structured = phi_patient_run_dir("20260101_exp", "P1") / "structured"
    assert (structured / "b.parquet").is_file()

    # remove b's source; explicit list -> warn only, parquet stays put
    (pdir / "b.csv").unlink()
    run_ingest_one(cfg=cfg, patient_id="P1")

    assert (structured / "a.parquet").is_file()
    assert (structured / "b.parquet").is_file()  # NOT quarantined
    quarantine = phi_patient_run_dir("20260101_exp", "P1") / "quarantined_structured"
    assert not quarantine.exists()
