"""A run folder must explain itself, and must never name a file it does not have.

Three failures from one real run, all of them quiet.

The folder a run leaves behind is flat — chunk_index.parquet, embeddings.npy, hnsw.bin,
evidence_selection_metadata, prepared_evidence_text, extract, structured, and a handful
of sidecars, side by side. Re-nesting it is not available: every cached stage is a
presence test at those exact paths, so moving them would silently re-run finished work
and hide finished runs from the review app. The folder therefore has to say what it is,
and the guide it writes has to stay true as the layout grows.

Two path helpers named folders — llm_messages/ and clinical_invariants/ — that nothing
in the pipeline has ever written. The workbench called them and printed "No messages
at ..." and "No invariants at ..." on every single run, while the real prompts sat in
extract/<variable>/steps/<step>/receipt.json and the real checks in
extract/<variable>/invariants.json. Absent and present looked identical.

And each step recorded its receipt as a path relative to its variable directory, which
the reader opened relative to the working directory instead. It therefore never found
one: on the run these tests were written from, a receipt holding 645 prompt tokens and
90 completion tokens produced a shareable cost table whose token columns were null for
every row — a run that spent real money looked exactly like a run that spent none. The
directory those receipts hang off has to be the one the transcript is read from, because
these runs are written on a cluster and read on a laptop: a location the run recorded
about itself stops being true the moment the folder is copied.
"""
from __future__ import annotations

import inspect
import json
import re
import shutil
from pathlib import Path

import pytest

from jr_pipeline.pipeline_steps.step_8_organize_output.extraction_outcome import (
    emit_extraction_outcome,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
    organize_per_recipe_outputs,
)
from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = "20260804_140652_d8e8"
PATIENT = "STSS10c40439"

# The shape of a real run, transcribed from the one the complaint was about
# (CONTAINS_PHI/pipeline_run_receipts/20260804_140652_d8e8, four stages, nine variables
# across eight patients). Paths are relative to the run folder; one patient and a few of
# the nineteen chart tables stand for the rest, which repeat.
ORDINARY_RUN_CONTENTS = [
    "manifest.json",
    "run_roster.json",
    "summary.json",
    "state.jsonl",
    "state_fragments/state.pid_41647.jsonl",
    "run_log.jsonl",
    "llm_cache.db",
    "exhaust_hmac_secret",
    "code/code.lock.json",
    "code/config_resolved.yaml",
    "code/hashes.json",
    "code/recipes/basic/date_of_birth/v1/date_of_birth_v1_recipe.yaml",
    "code/src/jr_pipeline/runtime_infrastructure/command_line_interface.py",
    f"patients/{PATIENT}/source_snapshot.json",
    f"patients/{PATIENT}/structured/clinical_note.parquet",
    f"patients/{PATIENT}/structured/clinical_note.parquet.meta.json",
    f"patients/{PATIENT}/structured/labs.parquet",
    f"patients/{PATIENT}/structured/labs.parquet.meta.json",
    f"patients/{PATIENT}/chunk_index.parquet",
    f"patients/{PATIENT}/chunk_index.parquet.meta.json",
    f"patients/{PATIENT}/embeddings.npy",
    f"patients/{PATIENT}/hnsw.bin",
    f"patients/{PATIENT}/hnsw.bin.meta.json",
    f"patients/{PATIENT}/clinical_invariants.json",
    f"patients/{PATIENT}/evidence_selection_metadata/date_of_birth/demographics_grab/evidence_selection.json",
    f"patients/{PATIENT}/prepared_evidence_text/date_of_birth/demographics_grab/evidence_blocks.json",
    f"patients/{PATIENT}/prepared_evidence_text/date_of_birth/demographics_grab/formatted_evidence.txt",
    f"patients/{PATIENT}/extract/date_of_birth/result.json",
    f"patients/{PATIENT}/extract/date_of_birth/invariants.json",
    f"patients/{PATIENT}/extract/date_of_birth/transcript.json",
    f"patients/{PATIENT}/extract/date_of_birth/steps/demographics_grab/receipt.json",
]

# The shareable half of the same run, relative to the output folder rather than to the
# run folder, because that is where it sits.
ORDINARY_SHAREABLE_CONTENTS = [
    f"{layout.NON_SENSITIVE_LABEL}/{RUN}/manifest.json",
    f"{layout.NON_SENSITIVE_LABEL}/{RUN}/exhaust/extraction_outcome.parquet",
    f"{layout.NON_SENSITIVE_LABEL}/{RUN}/exhaust/selection_judgment.parquet",
    f"{layout.NON_SENSITIVE_LABEL}/{RUN}/exhaust/clinical_invariant.parquet",
    f"{layout.NON_SENSITIVE_LABEL}/{RUN}/exhaust/_shards/extraction_outcome/"
    f"{RUN}.Mac.lan.41647.d1290c2a.jsonl",
]


def _matches(pattern: str, relative_path: str) -> bool:
    """Does one guide line cover this file? ``<name>`` stands for any one folder or file
    name; a pattern ending in ``/`` covers everything beneath it."""
    covers_subtree = pattern.endswith("/")
    pattern_parts = pattern.rstrip("/").split("/")
    path_parts = relative_path.split("/")
    if covers_subtree:
        if len(path_parts) < len(pattern_parts):
            return False
        path_parts = path_parts[: len(pattern_parts)]
    elif len(path_parts) != len(pattern_parts):
        return False
    return all(
        re.fullmatch(re.sub(r"<[^>]+>", ".*", re.escape(expected)), found)
        for expected, found in zip(pattern_parts, path_parts, strict=True)
    )


def _explained(relative_path: str) -> bool:
    return any(_matches(pattern, relative_path) for _, pattern, _ in layout.RUN_ARTIFACT_GUIDE)


def _explained_as_a_folder(relative_path: str) -> bool:
    """A folder is explained by any line that names something inside it."""
    depth = len(relative_path.split("/"))
    for _, pattern, _ in layout.RUN_ARTIFACT_GUIDE:
        parts = pattern.rstrip("/").split("/")
        if len(parts) >= depth and _matches("/".join(parts[:depth]) + "/", relative_path):
            return True
    return False


@pytest.mark.parametrize("relative_path", ORDINARY_RUN_CONTENTS + ORDINARY_SHAREABLE_CONTENTS)
def test_no_file_a_real_run_left_behind_is_unexplained(relative_path):
    """The complaint, made checkable: open the folder, and every single thing in it has
    a line saying what it is."""
    assert _explained(relative_path), (
        f"{relative_path} is in a real run folder and no line of the guide accounts for it"
    )


def test_walking_a_run_folder_turns_up_nothing_the_guide_missed(tmp_path):
    """The same check the way an operator meets it — walking the folder rather than
    reading a list — so a file that appears only in a nested corner is still caught."""
    for relative_path in ORDINARY_RUN_CONTENTS:
        written = tmp_path / relative_path
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text("", encoding="utf-8")
    layout.write_run_artifact_guide(tmp_path)

    unexplained = [
        str(found.relative_to(tmp_path))
        for found in sorted(tmp_path.rglob("*"))
        if found.is_file() and not _explained(str(found.relative_to(tmp_path)))
    ]
    assert unexplained == [], f"a run folder holds files the guide does not name: {unexplained}"


def _path_helpers():
    """Every layout helper that answers "where does X live" — recognised by its arguments
    being nothing but the identity of a run, a patient, or a root, and by its answer
    being a path. Anything needing content (``atomic_write_json``) is not one."""
    identity_arguments = {"run_id", "patient_id", "dr", "patient_root", "run_root"}
    for name, func in vars(layout).items():
        if name.startswith("_") or not inspect.isfunction(func):
            continue
        signature = inspect.signature(func)
        if signature.return_annotation == "Path" and set(signature.parameters) <= identity_arguments:
            yield name, func


def _call_helper(func, output_root: Path):
    parameters = inspect.signature(func).parameters
    arguments = {}
    if "run_id" in parameters:
        arguments["run_id"] = RUN
    if "patient_id" in parameters:
        arguments["patient_id"] = PATIENT
    if "patient_root" in parameters:
        arguments["patient_root"] = layout.phi_patient_run_dir(RUN, PATIENT, output_root)
    if "run_root" in parameters:
        arguments["run_root"] = layout.phi_intermediate_run_dir(RUN, output_root)
    if "dr" in parameters:
        arguments["dr"] = output_root
    return func(**arguments)


def test_a_folder_the_layout_can_name_cannot_be_left_out_of_the_guide(tmp_path):
    """The guide is a hand-written table, so the thing to guard is drift: add a helper
    that puts something new in a run folder and forget the line explaining it, and the
    folder is confusing again in exactly the way it was before."""
    run_folder = layout.phi_intermediate_run_dir(RUN, tmp_path)
    shareable_folder = layout.no_phi_run_dir(RUN, tmp_path)

    missing = []
    for name, func in _path_helpers():
        answer = _call_helper(func, tmp_path)
        if not isinstance(answer, Path):
            continue
        for folder, prefix in (
            (run_folder, ""),
            (shareable_folder, f"{layout.NON_SENSITIVE_LABEL}/<run>/"),
        ):
            if answer == folder or folder not in answer.parents:
                continue
            relative_path = prefix + str(answer.relative_to(folder))
            if not _explained_as_a_folder(relative_path):
                missing.append(f"{name}() -> {relative_path}")
    assert missing == [], (
        "these live in a run folder and no line of the guide mentions them: " + ", ".join(missing)
    )


def test_the_guide_is_written_when_a_runs_layout_is_ensured(tmp_path):
    layout.ensure_layout(RUN, tmp_path)

    guide = layout.phi_intermediate_run_dir(RUN, tmp_path) / "WHAT_IS_IN_HERE.txt"
    assert guide.is_file(), "a run was created without the file that explains it"
    assert "embeddings.npy" in guide.read_text(encoding="utf-8")


def test_a_run_already_on_disk_gains_the_guide_when_a_command_touches_it(tmp_path):
    """Runs made before the guide existed are the ones whose folders are hardest to read,
    so ensuring a layout has to add it to them too — without disturbing what is there."""
    run_folder = layout.phi_intermediate_run_dir(RUN, tmp_path)
    (run_folder / "patients" / PATIENT).mkdir(parents=True)
    (run_folder / "patients" / PATIENT / "embeddings.npy").write_text("older run", encoding="utf-8")

    layout.ensure_layout(RUN, tmp_path)

    assert (run_folder / "WHAT_IS_IN_HERE.txt").is_file()
    assert (run_folder / "patients" / PATIENT / "embeddings.npy").read_text(encoding="utf-8") == (
        "older run"
    ), "ensuring the layout rewrote a finished run's work"


def test_the_guide_reads_in_the_order_the_run_happened():
    """Reading the table top to bottom is meant to be reading the pipeline in order; the
    stage names are the ones the operator typed, so nothing needs translating."""
    from apps_and_interfaces.command_line_interface import PIPELINE_PHASES

    stages_in_order = list(dict.fromkeys(stage for stage, _, _ in layout.RUN_ARTIFACT_GUIDE))
    commands_in_order = [name for phase in PIPELINE_PHASES.values() for name in phase]

    assert stages_in_order[-1] == "run", "run-level bookkeeping belongs after the stages"
    assert [label.split(" ", 1)[1] for label in stages_in_order[:-1]] == commands_in_order
    assert [label.split(" ", 1)[0] for label in stages_in_order[:-1]] == ["1", "2", "3", "4"]


def test_no_layout_helper_points_at_a_folder_nothing_writes():
    """What made llm_messages/ and clinical_invariants/ possible: a helper whose only
    callers were outside the pipeline, so nothing ever created what it pointed at and the
    only thing it could produce was a message saying the file was not there."""
    # Resolved by parsing, not by counting text. Counting mentions made a comment count
    # as a caller, and a dead helper is exactly the kind of thing that attracts an
    # explanatory comment — so the two helpers this test exists to prevent could have
    # come back with one line of prose attached.
    import ast
    from itertools import chain

    called: set[str] = set()
    # The engine and the operator surfaces both count as callers: the interfaces in
    # apps_and_interfaces (the CLI, the review app) write several run artifacts
    # themselves — the seal command's shareable code copy, the app's feedback files.
    source_trees = (REPO_ROOT / "src", REPO_ROOT / "apps_and_interfaces")
    for source in chain.from_iterable(tree.rglob("*.py") for tree in source_trees):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except SyntaxError:  # not importable anyway; the import graph check owns that
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    called.add(function.id)
                elif isinstance(function, ast.Attribute):
                    called.add(function.attr)
    never_called = [name for name, _ in _path_helpers() if name not in called]
    assert never_called == [], (
        "no pipeline code calls these, so nothing writes what they point at; delete them "
        f"or point them at the real artifact: {never_called}"
    )


def _transcript_with_a_receipt(variable_dir: Path) -> dict:
    receipt = variable_dir / "steps" / "demographics_grab" / "receipt.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"payload": {"economics": {
            "prompt_tokens": 645, "completion_tokens": 90, "total_tokens": 735, "cached": False,
        }}}),
        encoding="utf-8",
    )
    return {
        "variable": "date_of_birth",
        "recipe": {"name": "date_of_birth", "version": "v1"},
        "provider_config_hash": "sha256:" + "a" * 64,
        "code_lock_hash": "sha256:" + "b" * 64,
        # Relative to the variable directory, exactly as the extract step records it —
        # and the transcript says nothing about where that directory is.
        "steps": [{"step_id": "demographics_grab", "status": "completed",
                   "receipt_path": "steps/demographics_grab/receipt.json"}],
    }


def _emitted_record(tmp_path: Path) -> dict:
    shards = layout.no_phi_exhaust_shards_dir(RUN, tmp_path) / "extraction_outcome"
    rows = [
        json.loads(line)
        for shard in shards.glob("*.jsonl")
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    return rows[0]


def test_the_token_count_survives_a_receipt_that_is_not_under_the_working_directory(
    tmp_path, monkeypatch
):
    """The defect exactly: the receipt is real and readable, it is simply not below wherever
    the command was typed. Reported as null, the cost table says the run was free."""
    secret_lifecycle._CACHE.clear()
    variable_dir = layout.phi_patient_run_dir(RUN, PATIENT, tmp_path) / "extract" / "date_of_birth"
    variable_dir.mkdir(parents=True)
    transcript = _transcript_with_a_receipt(variable_dir)
    monkeypatch.chdir(tmp_path.parent)  # anywhere but the run

    emit_extraction_outcome(
        run_id=RUN, patient_id=PATIENT, transcript=transcript, variable_dir=variable_dir,
        ok=True, data_final={"date_of_birth": "1957-04-15"}, errors=[], data_root=tmp_path,
    )

    record = _emitted_record(tmp_path)
    assert record["prompt_tokens"] == 645
    assert record["completion_tokens"] == 90
    assert record["total_tokens"] == 735


def test_a_run_whose_receipts_cannot_be_found_reports_no_cost_rather_than_no_spend(
    tmp_path, monkeypatch
):
    """A caller that cannot say where the transcript it holds came from must leave the
    token columns empty. Filling them with zeros would be the same lie in the other
    direction."""
    secret_lifecycle._CACHE.clear()
    variable_dir = layout.phi_patient_run_dir(RUN, PATIENT, tmp_path) / "extract" / "date_of_birth"
    variable_dir.mkdir(parents=True)
    transcript = _transcript_with_a_receipt(variable_dir)
    monkeypatch.chdir(tmp_path.parent)

    emit_extraction_outcome(
        run_id=RUN, patient_id=PATIENT, transcript=transcript,
        ok=True, data_final={"date_of_birth": "1957-04-15"}, errors=[], data_root=tmp_path,
    )

    record = _emitted_record(tmp_path)
    assert record.get("prompt_tokens") is None
    assert record.get("total_tokens") is None


def test_the_token_count_survives_the_run_being_copied_off_the_cluster(
    tmp_path, monkeypatch
):
    """The whole point of these runs is that they move: written on the cluster, read on a
    laptop. Organize the moved run and the receipt is right where the transcript is —
    anchoring on a path the run wrote down about itself puts the reader back on the
    cluster, and the shareable cost table goes null for every row again."""
    on_the_cluster = tmp_path / "on_the_cluster"
    variable_dir = layout.phi_patient_run_dir(RUN, PATIENT, on_the_cluster) / "extract" / "date_of_birth"
    variable_dir.mkdir(parents=True)
    transcript = _transcript_with_a_receipt(variable_dir)
    transcript.update({
        "output_schema_path": str(_a_schema_that_accepts_a_date(tmp_path)),
        "validation_rules": [],
        "data_final": {"date_of_birth": "1957-04-15"},
    })
    (variable_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")

    on_the_laptop = tmp_path / "on_the_laptop"
    shutil.move(str(on_the_cluster), str(on_the_laptop))
    secret_lifecycle._CACHE.clear()
    monkeypatch.setenv("JR_DATA_ROOT", str(on_the_laptop))

    organize_per_recipe_outputs(
        run_id=RUN, patient_id=PATIENT,
        patient_out=layout.phi_patient_run_dir(RUN, PATIENT, on_the_laptop),
        recipes_root=tmp_path / "recipes", ordered_names=["date_of_birth"],
        code_lock_hash=None,
    )

    record = _emitted_record(on_the_laptop)
    assert record["prompt_tokens"] == 645
    assert record["total_tokens"] == 735


def _a_schema_that_accepts_a_date(tmp_path: Path) -> Path:
    path = tmp_path / "date_of_birth_v1_output_schema.json"
    path.write_text(
        json.dumps({"type": "object", "properties": {"date_of_birth": {"type": ["string", "null"]}}}),
        encoding="utf-8",
    )
    return path
