"""`extract --new-run` starts a fresh run for the ANSWERS and keeps the corpus.

Ingest, embed and index describe the chart. Extraction asks questions of it. A recipe
change touches only the last of those, but every artifact is written under the run that
made it, so "start a fresh run" used to mean rebuilding passages and vectors that the
change could not possibly have affected — minutes to hours of embedding to re-ask a
question of text that never moved.

The distinction has to stay real in both directions: this command must never quietly
rebuild a corpus, and `run --new-run` must never quietly reuse one. Otherwise they are
the same command with two spellings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
    CORPUS_AFFECTING_CFG_KEYS,
    corpus_entry_names,
    inherit_corpus_for_patient,
    why_the_corpus_cannot_be_reused,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_patient_run_dir,
)

SOURCE, FRESH = "20260101_000000_aaaa", "20260102_000000_bbbb"


def _a_finished_run(tmp_path: Path, patient: str = "P1") -> Path:
    """A source run with a corpus AND extraction output, so the tests can tell which
    of the two crosses over."""
    d = phi_patient_run_dir(SOURCE, patient, tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunk_index.parquet").write_bytes(b"passages")
    (d / "chunk_index.parquet.meta.json").write_text('{"payload": {"encoder": {}}}')
    (d / "embeddings.npy").write_bytes(b"vectors" * 100)
    (d / "hnsw.bin").write_bytes(b"index")
    (d / "hnsw.bin.meta.json").write_text('{"payload": {}}')
    (d / "source_snapshot.json").write_text('{"files": []}')
    (d / "structured").mkdir()
    (d / "structured" / "clinical_note.parquet").write_bytes(b"notes")
    answers = d / "extract" / "stage"
    answers.mkdir(parents=True)
    (answers / "result.json").write_text('{"payload": {"ok": true}}')
    (d / "prepared_evidence_text").mkdir()
    (d / "clinical_invariants.json").write_text("{}")
    return d


def test_the_corpus_crosses_over_and_the_answers_do_not(tmp_path):
    _a_finished_run(tmp_path)

    inherit_corpus_for_patient(
        source_run_id=SOURCE, destination_run_id=FRESH, patient_id="P1", data_root=tmp_path
    )

    fresh = phi_patient_run_dir(FRESH, "P1", tmp_path)
    for corpus_artifact in ("chunk_index.parquet", "embeddings.npy", "hnsw.bin",
                            "source_snapshot.json", "structured/clinical_note.parquet"):
        assert (fresh / corpus_artifact).exists(), f"{corpus_artifact} did not carry over"
    for answer_artifact in ("extract", "prepared_evidence_text", "clinical_invariants.json"):
        assert not (fresh / answer_artifact).exists(), (
            f"{answer_artifact} was inherited; the new run must produce its own answers"
        )


def test_it_is_one_copy_on_disk(tmp_path):
    """The whole point is not re-embedding; copying gigabytes of vectors per iteration
    would only move the cost from time to disk."""
    _a_finished_run(tmp_path)

    record = inherit_corpus_for_patient(
        source_run_id=SOURCE, destination_run_id=FRESH, patient_id="P1", data_root=tmp_path
    )

    source = phi_patient_run_dir(SOURCE, "P1", tmp_path) / "embeddings.npy"
    fresh = phi_patient_run_dir(FRESH, "P1", tmp_path) / "embeddings.npy"
    assert os.stat(source).st_ino == os.stat(fresh).st_ino
    assert record["how"] == ["hard link"]


def test_the_new_run_survives_the_old_one_being_deleted(tmp_path):
    """A hard link, not a symlink. The scratch runs behind compare-models symlink, which
    is right for something built to be thrown away and wrong for a run somebody keeps."""
    import shutil

    _a_finished_run(tmp_path)
    inherit_corpus_for_patient(
        source_run_id=SOURCE, destination_run_id=FRESH, patient_id="P1", data_root=tmp_path
    )
    fresh = phi_patient_run_dir(FRESH, "P1", tmp_path)

    shutil.rmtree(phi_patient_run_dir(SOURCE, "P1", tmp_path).parent.parent)

    assert (fresh / "embeddings.npy").read_bytes() == b"vectors" * 100
    assert (fresh / "structured" / "clinical_note.parquet").exists()


def test_rewriting_in_the_new_run_cannot_reach_into_the_old_one(tmp_path):
    """Hard links share an inode, so this is the property that makes them safe here:
    every write in this pipeline is a temp file plus os.replace, which swaps a directory
    entry rather than editing bytes in place."""
    _a_finished_run(tmp_path)
    inherit_corpus_for_patient(
        source_run_id=SOURCE, destination_run_id=FRESH, patient_id="P1", data_root=tmp_path
    )
    fresh = phi_patient_run_dir(FRESH, "P1", tmp_path) / "embeddings.npy"
    source = phi_patient_run_dir(SOURCE, "P1", tmp_path) / "embeddings.npy"

    replacement = fresh.with_suffix(".tmp")
    replacement.write_bytes(b"rebuilt")
    os.replace(replacement, fresh)

    assert source.read_bytes() == b"vectors" * 100, "the source run was modified through the link"


def test_a_patient_with_no_corpus_is_refused_not_half_inherited(tmp_path):
    with pytest.raises(FileNotFoundError):
        inherit_corpus_for_patient(
            source_run_id=SOURCE, destination_run_id=FRESH, patient_id="nobody",
            data_root=tmp_path,
        )


def test_what_invalidates_reuse(tmp_path):
    """Anything that shapes the corpus. Nothing that happens downstream of one."""
    sealed = {
        "source_root": "/charts", "chunker": {"kind": "token_window", "overlap": 128},
        "encoder": {"model_id": "m", "max_tokens": 512}, "index": {"M": 32},
        "chart_columns_file": "map.yaml",
    }
    assert why_the_corpus_cannot_be_reused(dict(sealed), sealed) == []

    for key, changed_value in (
        ("source_root", "/other_charts"),
        ("chunker", {"kind": "token_window", "overlap": 64}),
        ("encoder", {"model_id": "different", "max_tokens": 512}),
        ("index", {"M": 64}),
        ("chart_columns_file", "other_map.yaml"),
    ):
        assert why_the_corpus_cannot_be_reused({**sealed, key: changed_value}, sealed) == [key]


def test_extraction_settings_do_not_invalidate_a_corpus():
    """max_chunks_per_prompt is run-invariant, which is a different property from
    shaping the corpus. It is applied when a prompt is assembled, long after the index
    exists: it caps how much of the corpus is shown, not what the corpus contains.
    Treating it as corpus-affecting would rebuild vectors to change a prompt."""
    for downstream in ("max_chunks_per_prompt", "max_provenance_retries", "recipes",
                       "recipes_root", "allowlist_path"):
        assert downstream not in CORPUS_AFFECTING_CFG_KEYS
        assert why_the_corpus_cannot_be_reused({downstream: "new"}, {downstream: "old"}) == []


def test_the_corpus_set_is_read_from_the_layout_guide_not_a_second_list():
    """Two lists of what ingest/embed/index own would drift, and the drift would show up
    as a new artifact silently not being inherited."""
    names = corpus_entry_names()

    assert {"chunk_index.parquet", "embeddings.npy", "hnsw.bin", "structured",
            "source_snapshot.json"} <= names
    assert not ({"extract", "retrieve", "prepared_evidence_text", "derived_tables",
                 "evidence_selection_metadata", "clinical_invariants.json"} & names)


# --- the command surface ----------------------------------------------------------

def test_the_three_commands_mean_three_different_things():
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    runner = CliRunner()
    assert "--new-run" in runner.invoke(main, ["extract", "--help"]).output
    assert "--new-run" in runner.invoke(main, ["ingest", "--help"]).output
    # --run-id survives as the explicit escape hatch, not as routine guidance.
    assert "Rarely needed" in runner.invoke(main, ["extract", "--help"]).output


def test_asking_for_both_at_once_is_still_refused():
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    result = CliRunner().invoke(
        main, ["extract", "--new-run", "--run-id", "20990101_000000_ab"]
    )
    assert result.exit_code != 0
    assert "--new-run starts a run" in result.output


def test_a_corpus_change_is_refused_and_points_at_the_other_command(tmp_path, monkeypatch):
    """Rebuilding quietly under a command that says it will not is worse than refusing:
    the two commands would stop meaning different things."""
    import click
    import pytest as _pytest

    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "_sealed_config_of", lambda _run: {"encoder": {"model_id": "old"}})

    with _pytest.raises(click.ClickException) as refusal:
        cli._refuse_if_the_corpus_cannot_be_inherited(
            {"run_id": FRESH, "encoder": {"model_id": "new"}}, SOURCE, ["P1"]
        )

    message = str(refusal.value)
    assert "encoder" in message
    assert "junior ingest --new-run" in message


def test_it_records_where_the_corpus_came_from(tmp_path, monkeypatch):
    from apps_and_interfaces import command_line_interface as cli

    _a_finished_run(tmp_path)
    monkeypatch.setattr(cli, "_sealed_config_of", lambda _run: {})
    monkeypatch.setattr(
        cli, "phi_intermediate_run_dir",
        lambda run_id, *a, **k: tmp_path / "CONTAINS_PHI" / "pipeline_run_receipts" / run_id,
    )
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))

    cli._inherit_the_corpus_into_a_fresh_run({"run_id": FRESH}, SOURCE, ["P1"])

    record = json.loads(
        (tmp_path / "CONTAINS_PHI" / "pipeline_run_receipts" / FRESH / "corpus_inheritance.json")
        .read_text()
    )
    assert record["corpus_source_run_id"] == SOURCE
    assert record["parent_run_id"] == SOURCE
    assert record["patients"][0]["patient_id"] == "P1"
    assert "embeddings.npy" in record["patients"][0]["inherited"]


def test_compare_models_still_uses_throwaway_symlinks():
    """Its scratch runs are built to be deleted, so sharing this module's lifecycle
    would be wrong. Sharing the primitives is fine; sharing the semantics is not."""
    import inspect

    from jr_pipeline.evaluating_pipeline_performance import compare_llm_models

    source = inspect.getsource(compare_llm_models._materialize_scratch_corpus)
    assert "symlink_to" in source, "compare-models scratch behaviour changed"


def test_status_shows_an_inherited_corpus_as_reused_not_unstarted(tmp_path, monkeypatch):
    """Progress is counted from state transitions, and an inherited stage records none.
    So a run started by `extract --new-run` reported ingest, embed and index as "not
    started yet" while their artifacts sat in the folder — and somebody acting on that
    would rebuild a corpus they already have, which is the exact cost this command
    exists to avoid."""
    from click.testing import CliRunner

    from apps_and_interfaces.command_line_interface import main

    charts = tmp_path / "charts" / "P1"
    charts.mkdir(parents=True)
    (charts / "clinical_note.csv").write_text("patient_id,text\nP1,a note\n")
    project = tmp_path / "junior_p.yaml"
    project.write_text(
        f"project: p\nrun_id: {FRESH}\nsource_root: {tmp_path / 'charts'}\n"
        f"output_root: {tmp_path / 'out'}\nrecipes: [stage]\n"
        f"recipes_root: {tmp_path / 'recipes'}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "out"))
    run_root = phi_patient_run_dir(FRESH, "P1", tmp_path / "out").parent.parent
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "corpus_inheritance.json").write_text(
        json.dumps({"corpus_source_run_id": SOURCE, "parent_run_id": SOURCE, "patients": []}),
        encoding="utf-8",
    )

    out = CliRunner().invoke(main, ["status", "--config", str(project)]).output

    assert f"reused from {SOURCE}" in out, out
    for stage in ("ingest", "embed", "index"):
        line = next((row for row in out.splitlines() if f" {stage} " in row), "")
        assert "not started yet" not in line, f"{stage}: {line}"


def test_nothing_is_created_when_the_corpus_cannot_be_inherited(tmp_path, monkeypatch):
    """The refusal is asked before anything is made. A run directory that exists only
    because a command failed becomes the newest run — so it is what the next command
    continues, and what the next `--new-run` inherits from."""
    import click
    import pytest as _pytest

    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "_sealed_config_of", lambda _run: {})

    with _pytest.raises(click.ClickException) as refusal:
        cli._refuse_if_the_corpus_cannot_be_inherited({"run_id": FRESH}, SOURCE, [])

    # Not "your chart settings changed", which is what an unsealed source used to say.
    assert "no sealed settings" in str(refusal.value)
    assert "junior ingest --new-run" in str(refusal.value)


def test_a_source_run_with_no_finished_corpus_says_so(monkeypatch):
    """Reachable: _patients_with_a_corpus keys on chunk_index.parquet, and an
    ingest-only run has structured/ without one. It used to report success and write a
    provenance record naming a run it had taken nothing from."""
    import click
    import pytest as _pytest

    from apps_and_interfaces import command_line_interface as cli

    monkeypatch.setattr(cli, "_sealed_config_of", lambda _run: {"encoder": {"model_id": "m"}})

    with _pytest.raises(click.ClickException) as refusal:
        cli._refuse_if_the_corpus_cannot_be_inherited(
            {"run_id": FRESH, "encoder": {"model_id": "m"}}, SOURCE, []
        )

    assert "no patient with a finished corpus" in str(refusal.value)


def test_the_settings_that_pick_text_and_tables_also_invalidate_reuse(tmp_path):
    """text_column decides which column became the embedded text; files and
    files_to_embed decide which tables exist at all; chunk_metadata_columns is the
    column map written inline. Every one of them shapes the corpus, and every one
    used to pass the gate — an operator could change what the corpus IS and be told
    it was being reused."""
    sealed = {"source_root": "/charts"}
    for key, changed_value in (
        ("text_column", "note_text"),
        ("files", ["clinical_note"]),
        ("files_to_embed", ["clinical_note", "radiology_report"]),
        ("chunk_metadata_columns", {"clinical_note": {"document_date": ["date"]}}),
    ):
        assert why_the_corpus_cannot_be_reused({**sealed, key: changed_value}, sealed) == [key]


def test_a_map_edited_in_place_invalidates_reuse(tmp_path):
    """The column map is a FILE, and editing its content between runs is the documented
    workflow — so two equal path strings say nothing. A map modified after the source
    run sealed is a change, path equality notwithstanding."""
    the_map = tmp_path / "site_column_map.yaml"
    the_map.write_text("chunk_metadata_columns: {}\n", encoding="utf-8")
    cfg = {"source_root": "/charts", "chart_columns_file": str(the_map)}
    sealed_long_after_the_edit = the_map.stat().st_mtime + 100
    sealed_before_the_edit = the_map.stat().st_mtime - 100

    assert why_the_corpus_cannot_be_reused(
        dict(cfg), cfg, source_sealed_at=sealed_long_after_the_edit,
    ) == []
    changed = why_the_corpus_cannot_be_reused(
        dict(cfg), cfg, source_sealed_at=sealed_before_the_edit,
    )
    assert changed and "chart_columns_file" in changed[0]
    assert "edited since" in changed[0]


def test_a_chart_that_changed_since_the_source_run_is_named(tmp_path):
    """source_root equality says the runs point at the same folder, nothing about its
    content. A corrected CSV dropped into a patient's folder after the source run must
    not be silently ignored by an inherited corpus."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
        hash_file,
    )
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
        patients_whose_charts_changed,
    )

    charts = tmp_path / "charts" / "P1"
    charts.mkdir(parents=True)
    csv = charts / "clinical_note.csv"
    csv.write_text("text\nhello\n", encoding="utf-8")
    patient_out = phi_patient_run_dir(SOURCE, "P1", tmp_path)
    patient_out.mkdir(parents=True)
    (patient_out / "source_snapshot.json").write_text(json.dumps({
        "payload": {"files": [
            {"relpath": "clinical_note.csv", "content_hash": hash_file(csv)},
        ]},
    }), encoding="utf-8")

    unchanged = patients_whose_charts_changed(
        source_run_id=SOURCE, patients=["P1"],
        source_root=tmp_path / "charts", data_root=tmp_path,
    )
    assert unchanged == []

    csv.write_text("text\nhello\ncorrected addendum\n", encoding="utf-8")

    assert patients_whose_charts_changed(
        source_run_id=SOURCE, patients=["P1"],
        source_root=tmp_path / "charts", data_root=tmp_path,
    ) == ["P1"]


def test_an_interrupted_inheritance_blocks_every_door_back_in(tmp_path, monkeypatch):
    """A crash mid-copy leaves a half-corpus that looks whole patient by patient. The
    marker written before the first file moves has to make that state loud: no stage
    resumes such a run, and no --new-run inherits from it."""
    import click as _click

    from apps_and_interfaces.command_line_interface import (
        _refuse_if_the_corpus_cannot_be_inherited,
    )
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import (
        INCOMPLETE_INHERITANCE_MARKER,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    source_root = phi_intermediate_run_dir(SOURCE)
    (source_root / "code").mkdir(parents=True)
    (source_root / "code" / "config_resolved.yaml").write_text(
        "source_root: /charts\n", encoding="utf-8",
    )
    _a_finished_run(Path(os.environ["JR_DATA_ROOT"]))
    (source_root / INCOMPLETE_INHERITANCE_MARKER).write_text("interrupted\n")

    with pytest.raises(_click.ClickException, match="half-copied corpus"):
        _refuse_if_the_corpus_cannot_be_inherited(
            {"source_root": "/charts"}, SOURCE, ["P1"],
        )


def test_a_named_patient_with_no_corpus_is_refused_before_anything_exists(
    tmp_path, monkeypatch,
):
    """--patient with no corpus used to fail AFTER the new run directory existed; the
    empty run then became the newest, and every later --new-run resolved it as the
    inherit-from source and refused with a false message."""
    import click as _click

    from apps_and_interfaces.command_line_interface import (
        _refuse_if_the_corpus_cannot_be_inherited,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    source_root = phi_intermediate_run_dir(SOURCE)
    (source_root / "code").mkdir(parents=True)
    (source_root / "code" / "config_resolved.yaml").write_text(
        "source_root: /charts\n", encoding="utf-8",
    )
    _a_finished_run(Path(os.environ["JR_DATA_ROOT"]))

    with pytest.raises(_click.ClickException) as refusal:
        _refuse_if_the_corpus_cannot_be_inherited(
            {"source_root": "/charts"}, SOURCE, ["ghost"], requested_patient="ghost",
        )

    assert "ghost" in str(refusal.value)
    assert "P1" in str(refusal.value), "does not list the patients that do have one"


def test_a_cohort_patient_without_a_corpus_is_refused_not_silently_dropped(
    tmp_path, monkeypatch,
):
    """The inherited run's roster is what could be inherited, so a cohort patient with
    no corpus would not fail — they would simply not exist in the new run."""
    import click as _click

    from apps_and_interfaces.command_line_interface import (
        _refuse_if_the_corpus_cannot_be_inherited,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    source_root = phi_intermediate_run_dir(SOURCE)
    (source_root / "code").mkdir(parents=True)
    (source_root / "code" / "config_resolved.yaml").write_text(
        f"source_root: {tmp_path / 'charts'}\n", encoding="utf-8",
    )
    _a_finished_run(Path(os.environ["JR_DATA_ROOT"]))
    for pid in ("P1", "P2_added_later"):
        (tmp_path / "charts" / pid).mkdir(parents=True, exist_ok=True)

    with pytest.raises(_click.ClickException) as refusal:
        _refuse_if_the_corpus_cannot_be_inherited(
            {"source_root": str(tmp_path / "charts")}, SOURCE, ["P1"],
        )

    assert "P2_added_later" in str(refusal.value)
    assert "--patient" in str(refusal.value), "does not offer the narrowing escape"


def test_a_chain_of_inheritances_names_the_original_builder(tmp_path, monkeypatch):
    """Run C inherits from B, which inherited from A. The builder of the vectors is A;
    restamping B as the source at each hop loses the answer the record exists for."""
    from apps_and_interfaces.command_line_interface import _record_inherited_corpus
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )

    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    run_a, run_b, run_c = "20260101_000000_aaaa", "20260102_000000_bbbb", "20260103_000000_cccc"
    phi_intermediate_run_dir(run_b).mkdir(parents=True)
    (phi_intermediate_run_dir(run_b) / "corpus_inheritance.json").write_text(json.dumps({
        "run_id": run_b, "parent_run_id": run_a, "corpus_source_run_id": run_a,
        "patients": [{"patient_id": "P1", "corpus_source_run_id": run_a,
                      "inherited": ["chunk_index.parquet"], "how": ["hard link"]}],
    }), encoding="utf-8")

    _record_inherited_corpus(run_c, run_b, [
        {"patient_id": "P1", "corpus_source_run_id": run_b,
         "inherited": ["chunk_index.parquet"], "how": ["hard link"]},
    ])

    record = json.loads(
        (phi_intermediate_run_dir(run_c) / "corpus_inheritance.json").read_text()
    )
    assert record["parent_run_id"] == run_b
    assert record["corpus_source_run_id"] == run_a
    assert record["patients"][0]["corpus_source_run_id"] == run_a
