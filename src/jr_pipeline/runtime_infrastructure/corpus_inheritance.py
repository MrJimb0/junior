"""Carry a finished corpus into a new run, so changing a recipe does not re-embed.

Ingest, embed and index turn a patient's chart files into passages, vectors and a search
index. Extraction reads all three and writes answers. A recipe change alters only the
last of those, but every artifact is written under the run that produced it, so
"start a fresh run" meant rebuilding a corpus the change could not have affected --
minutes to hours of embedding to re-ask a question of text that never moved.

So a new extraction run may INHERIT the corpus instead of rebuilding it. That is a claim
about the world and it has to be true, which is what most of this module is about:

- Only artifacts ingest/embed/index actually own are inherited, and which those are is
  read out of RUN_ARTIFACT_GUIDE rather than listed again here. A new corpus artifact
  added to one of those stages is inherited without anyone remembering to come back.
- Inheritance is refused outright if anything that shapes the corpus has changed since
  the source run was sealed. Rebuilding quietly under a command that says it will not
  is worse than refusing, because the two commands would stop meaning different things.
- The new run records what it inherited and from where, so the answer to "where did
  these vectors come from" is on disk rather than in somebody's memory.

Links, not symlinks. The scratch runs behind `compare-models` symlink, which is right
for something built to be thrown away, and wrong for a run that has to stay readable
after the run beside it is deleted. A hard link keeps one copy on disk while leaving
both runs independently valid. It is safe here for a specific reason: every write in
this pipeline goes through a temp file and ``os.replace``, which swaps a directory entry
rather than modifying bytes in place, so nothing written in the new run can reach
through a link into the old one.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    RUN_ARTIFACT_GUIDE,
    phi_patient_run_dir,
)

# The stages whose output describes the CHART rather than an answer about it.
_CORPUS_STAGE_LABELS = ("1 ingest", "2 embed", "3 index")

# Settings that change what the corpus IS. Everything else -- which recipes run, which
# endpoint answers, how many chunks reach one prompt -- happens downstream of a finished
# corpus and cannot invalidate it.
#
# max_chunks_per_prompt is deliberately NOT here despite being run-invariant. It is
# applied in retrieve_and_prompt_step when assembling a prompt, long after the index
# exists; it caps how much of the corpus is shown, not what the corpus contains. Being
# fixed for a run and shaping the corpus are different properties, and only one of them
# is the question here.
CORPUS_AFFECTING_CFG_KEYS = (
    "source_root",              # which charts were read
    "files",                    # which of their tables were ingested
    "files_to_embed",           # which of those tables were embedded
    "text_column",              # which column became the embedded free text
    "chart_columns_file",       # the site column map: text AND metadata resolution
    "chunk_metadata_columns",   # the same map, inlined instead of named as a file
    "chunker",                  # how that text was cut into passages
    "encoder",                  # what turned passages into vectors
    "index",                    # how those vectors were made searchable
)

# The file a run writes while an inheritance is copying, removed only when the copy
# and its provenance record are both complete. A crash mid-copy otherwise leaves a
# half-corpus that is indistinguishable from a whole one: `junior extract` would
# resume it and answer from whichever patients happened to finish.
INCOMPLETE_INHERITANCE_MARKER = "corpus_inheritance.INCOMPLETE"


def corpus_entry_names() -> frozenset[str]:
    """Names directly under a patient folder that ingest/embed/index own.

    Read from the layout guide, which is where stage ownership is already written down,
    so this cannot fall out of step with what the stages actually write."""
    names: set[str] = set()
    for stage_label, path_pattern, _ in RUN_ARTIFACT_GUIDE:
        if stage_label not in _CORPUS_STAGE_LABELS:
            continue
        parts = path_pattern.split("/")
        # "patients/<patient>/<entry>[/...]" -- the entry is what appears on disk,
        # whether that is a file or a directory holding more.
        if len(parts) >= 3 and parts[0] == "patients":
            names.add(parts[2])
    return frozenset(names)


def why_the_corpus_cannot_be_reused(
    current_cfg: dict, sealed_cfg: dict, *, source_sealed_at: float | None = None
) -> list[str]:
    """Corpus-shaping settings that differ from the run being inherited from.

    Compared against the SEALED config of the source run rather than against the live
    working tree: the sealed copy is what actually produced those artifacts, which is
    the only thing worth comparing to.

    ``chart_columns_file`` gets one extra question the others do not need: the map is
    a FILE, and editing its content between runs is the documented workflow — so two
    equal path strings say nothing about whether the map that shaped the corpus is
    the map on disk now. When the caller supplies the source run's seal time, a map
    file modified after it is a change, path equality notwithstanding."""
    changed: list[str] = []
    for key in CORPUS_AFFECTING_CFG_KEYS:
        # A key absent from both sides never changed. A key present on one side only
        # did, and reads as a change to whoever set it.
        before, after = sealed_cfg.get(key), current_cfg.get(key)
        if before != after:
            changed.append(key)
    if "chart_columns_file" not in changed and source_sealed_at is not None:
        named = current_cfg.get("chart_columns_file")
        if named:
            map_file = Path(str(named)).expanduser()
            try:
                edited_after_sealing = (
                    map_file.is_file() and map_file.stat().st_mtime > source_sealed_at
                )
            except OSError:
                edited_after_sealing = True
            if edited_after_sealing:
                changed.append("chart_columns_file (edited since that run was built)")
    return changed


def patients_whose_charts_changed(
    *,
    source_run_id: str,
    patients: list[str],
    source_root: Path | str,
    data_root: Path | None = None,
) -> list[str]:
    """Patients whose CURRENT source files no longer match what the source run read.

    ``source_root`` equality says the runs point at the same folder; it says nothing
    about the folder's content. A corrected CSV dropped into a patient's folder after
    the source run would be silently ignored by an inherited corpus — extraction would
    answer from the old chart while the operator believes the correction is in. The
    comparison is the same one ingest's own cache makes: the per-patient source
    snapshot against the files on disk now."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.per_patient_source_snapshot import (  # noqa: E501
        snapshot_matches,
    )

    changed = []
    root = Path(source_root).expanduser()
    for patient_id in patients:
        patient_src = root / patient_id
        if not patient_src.is_dir():
            # A vanished source folder is its own problem, reported by the stage; the
            # question here is only whether an EXISTING chart moved under the corpus.
            continue
        if not snapshot_matches(
            phi_patient_run_dir(source_run_id, patient_id, data_root), patient_src
        ):
            changed.append(patient_id)
    return changed


def _link_one_file(source: Path, destination: Path) -> str:
    """One file into the new run, cheapest first. Returns how it got there.

    A hard link is the goal: one copy on disk, both runs independently valid, and
    deleting either leaves the other whole. It fails across filesystems, so a copy is
    the fallback -- correct everywhere, just larger."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "already there"
    try:
        os.link(source, destination)
        return "hard link"
    except OSError:
        # Cross-device, or a filesystem without links. shutil.copy2 uses the platform's
        # fast path where there is one (fcopyfile on macOS, copy_file_range on Linux),
        # which is where a copy-on-write clone happens when the filesystem offers it.
        shutil.copy2(source, destination)
        return "copy"


def inherit_corpus_for_patient(
    *,
    source_run_id: str,
    destination_run_id: str,
    patient_id: str,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Put one patient's ingest/embed/index artifacts into the destination run.

    Extraction output is not inherited: the new run exists to produce its own, and a
    stale answer sitting beside a fresh one, both claiming the new run, is the exact
    confusion this is meant to avoid."""
    source_dir = phi_patient_run_dir(source_run_id, patient_id, data_root)
    destination_dir = phi_patient_run_dir(destination_run_id, patient_id, data_root)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"{patient_id} has no corpus in run {source_run_id}"
        )

    wanted = corpus_entry_names()
    inherited: list[str] = []
    how: set[str] = set()
    for entry in sorted(source_dir.iterdir()):
        if entry.name not in wanted:
            continue
        target = destination_dir / entry.name
        if entry.is_dir():
            for inner in sorted(entry.rglob("*")):
                if inner.is_file():
                    how.add(_link_one_file(inner, target / inner.relative_to(entry)))
        else:
            how.add(_link_one_file(entry, target))
        inherited.append(entry.name)

    if not inherited:
        raise FileNotFoundError(
            f"{patient_id} has nothing to inherit in run {source_run_id} — "
            "that run has no finished corpus for this patient"
        )
    return {
        "patient_id": patient_id,
        "corpus_source_run_id": source_run_id,
        "inherited": sorted(inherited),
        "how": sorted(how),
    }
