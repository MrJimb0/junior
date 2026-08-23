"""The import-graph dead-module gate.

A module under ``src/jr_pipeline`` that is unreachable from the production roots and
not explicitly allowlisted is a failure — dead code, or code kept alive
only by tests. The allowlist documents known-deferred wiring with a one-line reason
and is kept honest by the stale-entry check (an allowlisted module that becomes
production-reachable must be removed).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_graph_checker import (  # noqa: E402  (sibling helper module, not a package)
    ALLOWLIST,
    classify_dead,
    production_dead_modules,
    stale_allowlist_entries,
)


def test_no_unallowlisted_production_dead_modules():
    dead = production_dead_modules()
    detail = "\n".join(f"  {m} ({classify_dead(m)})" for m in dead)
    assert not dead, (
        "production-dead src modules not in the allowlist — delete the code, wire "
        f"it to a production entry point, or allowlist with a reason:\n{detail}"
    )


def test_allowlist_entries_are_actually_dead():
    stale = stale_allowlist_entries()
    assert not stale, (
        "allowlist entries that are now production-reachable — remove them:\n  "
        + "\n  ".join(stale)
    )


def test_every_allowlist_entry_has_a_reason():
    missing = sorted(name for name, reason in ALLOWLIST.items() if not reason.strip())
    assert not missing, f"allowlist entries missing a one-line reason: {missing}"
