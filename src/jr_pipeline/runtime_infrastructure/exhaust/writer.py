"""Writer-owned ``emit()`` — the single NO_PHI exhaust write gate.

Every exhaust record is born here. ``emit`` is the one place that:

1. mints surrogates for the caller's raw PHI ids (``phi_keys``) with the scope's
   secret -- a call site never hand-rolls a surrogate or puts a raw id in the record;
2. pydantic-validates the final record against its schema -- a stray key,
   out-of-vocab enum, or missing field raises;
3. runs the schema-aware forbidden-content scan -- a raw id/date/text that
   slipped past 1-2 raises; and only then
4. appends the record to a **process-unique** JSONL shard:
   ``NO_PHI/<run>/exhaust/_shards/<record_type>/<run>.<host>.<pid>.<uuid>.jsonl``.

No two processes share a shard, so SLURM-array workers never contend on one file;
finalize compacts the shards into ``<record_type>.parquet``. A gate failure
raises ``ExhaustWriteError`` -- the record is never written. Callers may catch it only
to record a durable failure count; never to drop silently.
"""
from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    no_phi_exhaust_shards_dir,
)
from jr_pipeline.runtime_infrastructure.exhaust.forbidden_content import (
    scan_record_for_forbidden_content,
)
from jr_pipeline.runtime_infrastructure.exhaust.schema import validate_record
from jr_pipeline.runtime_infrastructure.exhaust.secret_lifecycle import secret_for_scope
from jr_pipeline.runtime_infrastructure.exhaust.surrogates import LinkageScope, make_surrogate

_PROCESS_TOKEN: str | None = None


class ExhaustWriteError(RuntimeError):
    """An exhaust record failed the write gate (schema or forbidden-content). The
    record is NOT written. Catch only to count a durable failure -- never to drop."""


def _process_token() -> str:
    """``<host>.<pid>.<uuid8>`` -- stable within a process so all of its records for a
    record type land in one shard; unique across processes so shards never collide."""
    global _PROCESS_TOKEN
    if _PROCESS_TOKEN is None:
        _PROCESS_TOKEN = f"{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    return _PROCESS_TOKEN


def emit(
    record_type: str,
    record: dict[str, Any],
    *,
    run_id: str,
    phi_keys: dict[str, Any] | None = None,
    linkage_domains: dict[str, str] | None = None,
    scope: str = LinkageScope.RUN.value,
    data_root: Path | None = None,
) -> Path:
    """Validate, scrub, and append one NO_PHI exhaust record; return the shard path.

    ``record`` must contain NO raw PHI identifiers. ``phi_keys`` maps each surrogate
    output-field name to its raw identifier; ``emit`` substitutes the surrogate so the
    raw value never enters ``record``. Raises ``ExhaustWriteError`` on any gate
    failure (and the secret layer raises ``ExhaustSecretError`` if the key is missing
    or unsafe -- both fail closed, nothing is written).

    ``linkage_domains`` optionally overrides the surrogate domain for specific
    output-fields. By default a surrogate is domain-separated by this record's
    ``record_type`` (so the same id does NOT join across record types -- the right
    default for patient identity). A field listed here is minted under the given shared
    domain instead, used for the evidence-chunk surrogate so a ``relevance_label`` /
    ``retriever_miss`` joins back to its ``selection_judgment`` row. Use ONLY for chunk
    linkage; patient/reviewer ids must stay per-record-type.
    """
    secret = secret_for_scope(
        scope,
        run_id=run_id,
        study_id=record.get("study_id"),
        site_id=record.get("site_id"),
        dr=data_root,
    )

    final = dict(record)
    for output_field, raw in (phi_keys or {}).items():
        domain = (linkage_domains or {}).get(output_field, record_type)
        final[output_field] = make_surrogate(
            secret, scope=scope, record_type=domain, output_field=output_field, raw=str(raw)
        )

    try:
        validated = validate_record(record_type, final)  # schema gate
    except ValidationError as exc:
        # The pydantic message embeds the offending INPUT value, which may be PHI; surface
        # only the failing field paths, and drop the cause so the raw value can't be logged.
        locs = "; ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
        raise ExhaustWriteError(
            f"exhaust record {record_type!r} failed schema validation at fields: {locs}"
        ) from None
    except Exception as exc:  # unknown record_type etc. (no PHI in the message)
        raise ExhaustWriteError(f"exhaust record {record_type!r} failed validation: {exc}") from exc

    violations = scan_record_for_forbidden_content(validated)  # PHI scan
    if violations:
        raise ExhaustWriteError(
            f"exhaust record {record_type!r} failed forbidden-content scan: " + "; ".join(violations)
        )

    shard = no_phi_exhaust_shards_dir(run_id, data_root) / record_type / f"{run_id}.{_process_token()}.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    with shard.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(validated, sort_keys=True, ensure_ascii=False) + "\n")
    return shard
