"""The surrogate secret is PHI-side, 0600, and fail-closed.

A missing secret (when not creatable), or one with group/other-readable permissions,
must raise rather than let ``emit`` mint weak/empty-keyed surrogates.
"""
from __future__ import annotations

import os
import stat

import pytest

from jr_pipeline.runtime_infrastructure import (
    data_directory_layout_and_safe_writes as layout,
)
from jr_pipeline.runtime_infrastructure.exhaust import secret_lifecycle as sl


def setup_function(_):
    sl._CACHE.clear()


def test_run_secret_is_created_phi_side_with_0600(tmp_path):
    secret = sl.run_secret("20260617_000000", dr=tmp_path)
    assert isinstance(secret, bytes) and len(secret) == 32
    path = sl._run_secret_path("20260617_000000", tmp_path)
    assert path.is_file()
    assert layout.SENSITIVE_LABEL in path.parts  # under CONTAINS_PHI, never NO_PHI
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0  # no group/other access


def test_run_secret_is_stable_across_reads(tmp_path):
    first = sl.run_secret("20260617_000000", dr=tmp_path)
    sl._CACHE.clear()  # force a fresh on-disk read (simulates a second process)
    assert sl.run_secret("20260617_000000", dr=tmp_path) == first


def test_wrong_permissions_fail_closed(tmp_path):
    sl.run_secret("20260617_000000", dr=tmp_path)
    path = sl._run_secret_path("20260617_000000", tmp_path)
    os.chmod(path, 0o644)  # group/other readable
    sl._CACHE.clear()
    with pytest.raises(sl.ExhaustSecretError):
        sl.run_secret("20260617_000000", dr=tmp_path)


def test_missing_secret_with_no_create_fails_closed(tmp_path):
    with pytest.raises(sl.ExhaustSecretError):
        sl.study_secret("study_x", dr=tmp_path)  # create defaults False


def test_site_secret_is_never_auto_created(tmp_path):
    with pytest.raises(sl.ExhaustSecretError):
        sl.site_secret("site_x", dr=tmp_path, create=True)


def test_study_secret_creates_only_when_asked(tmp_path):
    secret = sl.study_secret("study_x", dr=tmp_path, create=True)
    assert len(secret) == 32
    sl._CACHE.clear()
    assert sl.study_secret("study_x", dr=tmp_path) == secret  # now readable


def test_secret_for_scope_dispatch_and_unknown(tmp_path):
    assert sl.secret_for_scope("run", run_id="20260617_000000", dr=tmp_path)
    with pytest.raises(sl.ExhaustSecretError):
        sl.secret_for_scope("galaxy", run_id="x", dr=tmp_path)


def test_secret_fingerprint_is_stable_12_hex():
    fp = sl.secret_fingerprint(b"\x01" * 32)
    assert fp == sl.secret_fingerprint(b"\x01" * 32)
    assert len(fp) == 12 and all(c in "0123456789abcdef" for c in fp)
    assert fp != sl.secret_fingerprint(b"\x02" * 32)
