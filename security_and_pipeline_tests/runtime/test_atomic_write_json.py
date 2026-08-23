"""atomic_write_json is corruption-safe under concurrent writers.

py-filelock removes the lock file on release, which has a TOCTOU window that can
admit two writers into the critical section at once. atomic_write_json defends by
rendering into a PER-PROCESS-UNIQUE tmp file and using an atomic os.replace, so even
a double-admission cannot leave a torn/partial artifact (a reader always sees a
complete prior-or-new file), and no shared tmp file can be clobbered mid-write.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
)


def _tmp_leftovers(d):
    return [q.name for q in d.iterdir() if q.name.endswith(".tmp")]


def test_writes_then_overwrites_atomically(tmp_path):
    p = tmp_path / "nested" / "out.json"
    atomic_write_json(p, {"b": 1, "a": 2})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 2, "b": 1}
    atomic_write_json(p, {"a": 3})  # overwrite in place
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 3}
    # no temp file lingers after a successful write
    assert _tmp_leftovers(p.parent) == []


def test_concurrent_writers_never_leave_a_torn_file(tmp_path):
    p = tmp_path / "out.json"
    payloads = [{"writer": i, "filler": "x" * 256} for i in range(12)]

    def write(i):
        for _ in range(8):
            atomic_write_json(p, payloads[i])

    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(write, range(12)))

    # the file is always complete and equals exactly one writer's payload
    final = json.loads(p.read_text(encoding="utf-8"))
    assert final in payloads
    assert _tmp_leftovers(tmp_path) == []
