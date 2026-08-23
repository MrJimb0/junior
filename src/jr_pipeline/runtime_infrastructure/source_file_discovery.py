"""Shared discovery of a patient's source files.

One canonical place for the three things the ingest step and the per-patient
source snapshot must agree on: which file extensions count as source data, the
case-insensitive rule for matching them, and the ``stem -> path`` map of a
patient folder. Both modules import this so discovery, file resolution, and the
snapshot can never disagree about which files exist.

The divergence this closes (bug B3): discovery lowercased each file's suffix
(so ``NOTES.CSV`` was discovered as stem ``NOTES``), while file resolution and
the snapshot only ever matched lowercase extensions. On a case-insensitive
filesystem (macOS) ``NOTES.CSV`` was ingested but invisible to the snapshot, so
a later edit to it never busted the cache; on a case-sensitive filesystem
(Linux) it was discovered but unresolvable, raising ``FileNotFoundError``
mid-run. Routing every caller through ``discover_source_files`` — which matches
case-insensitively and returns the real on-disk path — closes both halves.

This is a leaf module: it imports only the standard library, so the ingest step
(``pipeline_steps``) and the snapshot (``reproducibility``) can both depend on
it without an import cycle.
"""
from __future__ import annotations

from pathlib import Path

# Supported source extensions in resolution-priority order: when one stem has
# several supported files (e.g. notes.csv beside a stale notes.parquet), the
# extension earliest in this tuple wins. .parquet sits last so a fresh EHR csv
# never loses to a leftover parquet copy in the folder. Ingest's reader dispatch
# is keyed by these same extensions and asserts agreement at import time, so the
# set of readable formats and the set of discoverable formats cannot drift.
SUPPORTED_SOURCE_EXTENSIONS: tuple[str, ...] = (
    ".csv", ".tsv", ".xlsx", ".json", ".jsonl", ".parquet",
)

_EXTENSION_PRIORITY: dict[str, int] = {
    extension: rank for rank, extension in enumerate(SUPPORTED_SOURCE_EXTENSIONS)
}


def is_supported_source_file(path: Path) -> bool:
    """True for a non-hidden regular file with a supported extension.

    The extension is matched case-insensitively so ``NOTES.CSV`` and
    ``notes.csv`` are treated identically on every filesystem."""
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in _EXTENSION_PRIORITY
    )


def discover_source_files(patient_dir: Path) -> dict[str, Path]:
    """Map each supported source file in a patient folder to its stem.

    The returned paths are the real directory entries — original name and case —
    so a caller on a case-sensitive filesystem opens the file that actually
    exists on disk. When one stem has several supported files, the extension
    earliest in ``SUPPORTED_SOURCE_EXTENSIONS`` wins, matching the priority the
    old per-stem lookup applied. Hidden files are skipped. The folder is walked
    in name-sorted order so the winning file per stem is deterministic
    regardless of the order the filesystem reports entries.
    """
    by_stem: dict[str, Path] = {}
    for path in sorted(Path(patient_dir).iterdir(), key=lambda p: p.name):
        if not is_supported_source_file(path):
            continue
        incumbent = by_stem.get(path.stem)
        if incumbent is None or (
            _EXTENSION_PRIORITY[path.suffix.lower()]
            < _EXTENSION_PRIORITY[incumbent.suffix.lower()]
        ):
            by_stem[path.stem] = path
    return by_stem
