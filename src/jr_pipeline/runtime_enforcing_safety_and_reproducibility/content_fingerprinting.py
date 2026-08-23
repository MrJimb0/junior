"""Canonical content hashing for artifacts.

Every artifact carries a ``content_hash`` of the form ``sha256:<64-hex>``
(see schemas/json/artifact_envelope.schema.json). JSON is hashed after
canonical serialization (sorted keys, no whitespace, UTF-8) so the same hash
is reproducible outside Python. Files and directories are streamed in chunks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_HEX_PREFIX = "sha256:"
_CHUNK_BYTES = 1 << 20  # 1 MiB


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    """sha256:<hex> of raw bytes."""
    return _HEX_PREFIX + hashlib.sha256(data).hexdigest()


def hash_json(obj: Any) -> str:
    """sha256:<hex> of a JSON-compatible object after canonical serialization."""
    return hash_bytes(_canonical_json(obj))


def hash_file(path: Path | str) -> str:
    """Stream-hash a file on disk."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(_CHUNK_BYTES):
            h.update(chunk)
    return _HEX_PREFIX + h.hexdigest()


def hash_directory(directory: Path | str, *, exclude: frozenset[str] = frozenset()) -> str:
    """Stream-hash every file under ``directory`` recursively, in sorted relative-path order.

    Each file contributes its POSIX-relative path and bytes, NUL-separated to
    prevent concatenation ambiguity. ``exclude`` (relative POSIX paths) lets a
    bundle hold its own metadata file out of the identity it describes.
    """
    directory = Path(directory)
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = p.relative_to(directory).as_posix()
            if rel in exclude:
                continue
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            with p.open("rb") as f:
                while chunk := f.read(_CHUNK_BYTES):
                    h.update(chunk)
            h.update(b"\0")
    return _HEX_PREFIX + h.hexdigest()


def hash_artifact_payload(envelope: dict) -> str:
    """Canonical content_hash of an envelope, excluding ``content_hash`` and ``produced_by.created_at``."""
    view = {k: v for k, v in envelope.items() if k != "content_hash"}
    if "produced_by" in view and isinstance(view["produced_by"], dict):
        pb = dict(view["produced_by"])
        pb.pop("created_at", None)
        view["produced_by"] = pb
    return hash_json(view)
