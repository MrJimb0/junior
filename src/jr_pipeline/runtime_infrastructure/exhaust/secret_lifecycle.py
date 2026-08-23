"""HMAC secret lifecycle for exhaust surrogates.

The surrogate secret is the one piece of the exhaust pipeline that MUST stay
PHI-side: whoever holds it can recompute (and thus invert, by dictionary attack on a
low-entropy id space) every surrogate. So the secret lives under the CONTAINS_PHI
tree, file mode ``0600`` inside a ``0700`` parent, and is **fail-closed**: a missing
secret (when not creatable) or wrong permissions raises rather than silently minting
weak/empty-keyed surrogates.

Scopes mirror `surrogates.LinkageScope`:
  * ``run``   — ``<phi run dir>/exhaust_hmac_secret``; auto-created (one per run).
  * ``study`` — ``<phi root>/exhaust_secrets/study/<id>``; created only on explicit
    request (an init/approved-config act), never silently during ``emit``.
  * ``site``  — ``<phi root>/exhaust_secrets/site/<id>``; never auto-created (ADR-gated);
    ``create=True`` is rejected here.

Creation is SLURM-array safe: a complete 0600 file is published atomically via
``os.link`` (link fails if it exists -> exactly one writer wins, no reader ever sees a
partial/empty key). Rotation = a new ``surrogate_version`` (see `surrogates`), which
re-keys cleanly; this module does not overwrite a live secret in place.
"""
from __future__ import annotations

import hmac
import os
import secrets
import stat
import time
from hashlib import sha256
from pathlib import Path

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    phi_intermediate_run_dir,
    phi_root,
)
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import LinkageScope

_SECRET_BYTES = 32
_RUN_SECRET_NAME = "exhaust_hmac_secret"
_CACHE: dict[str, bytes] = {}


class ExhaustSecretError(RuntimeError):
    """A surrogate secret is missing, unreadable, or has unsafe permissions
    (fail-closed: emit must not proceed without a sound key)."""


def _run_secret_path(run_id: str, dr: Path | None) -> Path:
    return phi_intermediate_run_dir(run_id, dr) / _RUN_SECRET_NAME


def _scoped_secret_path(scope: str, identifier: str, dr: Path | None) -> Path:
    return phi_root(dr) / "exhaust_secrets" / scope / identifier


def _verify_permissions(path: Path) -> None:
    """Fail closed unless the secret is 0600-tight (no group/other access)."""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ExhaustSecretError(
            f"exhaust secret {path} has unsafe permissions {oct(mode)} "
            "(expected 0600, no group/other access)"
        )


def _read_secret(path: Path, *, retries: int = 0) -> bytes | None:
    """Read a complete secret, treating an empty/torn file as not-yet-present."""
    for attempt in range(retries + 1):
        if path.exists():
            _verify_permissions(path)
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                return bytes.fromhex(raw)
        if attempt < retries:
            time.sleep(0.01)
    return None


def _create_secret(path: Path) -> bytes:
    """Atomically publish a fresh 0600 secret under a 0700 parent (link-publish)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass  # best-effort tightening; the file mode is the hard guarantee
    secret = secrets.token_bytes(_SECRET_BYTES)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)  # defeat a permissive umask: the mode is the hard guarantee
        os.write(fd, secret.hex().encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.link(str(tmp), str(path))  # atomic publish of a complete, 0600 file
        _verify_permissions(path)  # the creator is held to the same fail-closed gate as readers
        return secret
    except FileExistsError:  # a racing writer won — read theirs (now complete)
        existing = _read_secret(path, retries=200)
        if existing is None:
            raise ExhaustSecretError(f"exhaust secret {path} present but unreadable") from None
        return existing
    finally:
        tmp.unlink(missing_ok=True)


def _resolve(path: Path, *, create: bool) -> bytes:
    cached = _CACHE.get(str(path))
    if cached is not None:
        if path.exists():
            _verify_permissions(path)  # keep the fail-closed perms promise live across the run
        return cached
    existing = _read_secret(path)
    if existing is not None:
        _CACHE[str(path)] = existing
        return existing
    if not create:
        raise ExhaustSecretError(
            f"exhaust secret {path} is missing and create=False (fail-closed)"
        )
    secret = _create_secret(path)
    _CACHE[str(path)] = secret
    return secret


def run_secret(run_id: str, dr: Path | None = None) -> bytes:
    """The run-scoped HMAC secret, auto-created on first use (one per run)."""
    return _resolve(_run_secret_path(run_id, dr), create=True)


def study_secret(study_id: str, dr: Path | None = None, *, create: bool = False) -> bytes:
    """The study-scoped secret. ``create`` is an explicit init act, never implicit."""
    return _resolve(_scoped_secret_path(LinkageScope.STUDY.value, study_id, dr), create=create)


def site_secret(site_id: str, dr: Path | None = None, *, create: bool = False) -> bytes:
    """The site-scoped secret. Never auto-created (ADR-gated); ``create=True`` rejected."""
    if create:
        raise ExhaustSecretError(
            "site-scope secrets are never auto-created (ADR-gated); provision out of band"
        )
    return _resolve(_scoped_secret_path(LinkageScope.SITE.value, site_id, dr), create=False)


def secret_for_scope(
    scope: str,
    *,
    run_id: str | None = None,
    study_id: str | None = None,
    site_id: str | None = None,
    dr: Path | None = None,
) -> bytes:
    """Resolve the secret for a `LinkageScope`, applying each scope's creation policy."""
    if scope == LinkageScope.RUN.value:
        if not run_id:
            raise ExhaustSecretError("run-scope secret requires run_id")
        return run_secret(run_id, dr)
    if scope == LinkageScope.STUDY.value:
        if not study_id:
            raise ExhaustSecretError("study-scope secret requires study_id")
        return study_secret(study_id, dr)
    if scope == LinkageScope.SITE.value:
        if not site_id:
            raise ExhaustSecretError("site-scope secret requires site_id")
        return site_secret(site_id, dr)
    raise ExhaustSecretError(f"unknown linkage scope: {scope!r}")


def secret_fingerprint(secret: bytes) -> str:
    """A non-reversible 12-hex digest identifying *which* secret minted a run's
    surrogates (stamped into the exhaust manifest; proves key continuity without
    revealing the key)."""
    return hmac.new(bytes(secret), b"fingerprint", sha256).hexdigest()[:12]
