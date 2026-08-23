"""a small on-disk cache of language-model responses, so an identical model call
isn't paid for twice — but built so a silent model swap can never serve a stale answer.

The cache is a single SQLite database file. Each cached answer is filed under a
key built from the full request plus the provider's configuration and the exact
model's fingerprint — a short value identifying the precise model/endpoint that
produced it (ADR 0009 + ADR 0028). So if the institution's API gateway quietly
points an endpoint at a different model, the fingerprint no longer matches and we
treat it as a cache miss (re-asking) instead of returning a stale answer that a
different model actually produced.

SQLite's WAL mode (write-ahead logging) plus a 30-second busy timeout let two
worker processes share the same file safely on a local disk. The caller chooses
where the file lives (extract.resolve_llm_cache_path defaults it inside the run's
PHI directory tree).

The database file and its WAL/SHM companion files are locked to owner-only access
(file mode 0600), and the folder to owner-only (0700) (ADR 0027), because the
cached prompts and responses are PHI. On a cache hit the returned response is tagged
``usage.cache = "hit"`` so it is easy to see in logs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
    LLMResponse,
    ResolvedModel,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_json,
)

# The WAL/SHM companion files can hold fragments of the (PHI) prompts too, so they
# get the same owner-only 0600 file mode as the main database.
_WAL_SIDECAR_SUFFIXES = ("-wal", "-shm")
_SQLITE_BUSY_TIMEOUT_MS = 30000
_SQLITE_CONNECT_TIMEOUT_S = 30.0


@dataclass
class LLMCache:
    """SQLite-backed response cache. The file is owner-only (0600) because the cached
    prompts and responses contain patient text (PHI)."""

    path: Path
    _conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # ADR 0027: lock the folder to owner-only before the file is created in it.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
            self._conn = sqlite3.connect(
                str(self.path), isolation_level=None, timeout=_SQLITE_CONNECT_TIMEOUT_S
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "  key TEXT PRIMARY KEY,"
                "  endpoint_name TEXT NOT NULL,"
                "  fingerprint TEXT NOT NULL,"
                "  created_at REAL NOT NULL,"
                "  response_json TEXT NOT NULL"
                ")"
            )
            for candidate in (
                self.path,
                *(self.path.with_name(self.path.name + s) for s in _WAL_SIDECAR_SUFFIXES),
            ):
                if candidate.is_file():
                    try:
                        os.chmod(candidate, 0o600)
                    except OSError:
                        pass
        return self._conn

    @staticmethod
    def make_key(
        *,
        req: LLMRequest,
        provider_config: dict[str, Any],
    ) -> str:
        """build the lookup key for a request (the same request always yields the same
        key). The model fingerprint is deliberately NOT part of the key — it is checked
        separately at get() time so a fingerprint mismatch reads as a miss (ADR 0028)."""
        payload = {
            "endpoint_name": req.endpoint_name,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "response_format": req.response_format,
            "seed": req.seed,
            "provider_config": provider_config,
        }
        return hash_json(payload)

    def get(
        self,
        key: str,
        *,
        expected_fingerprint: str | None = None,
    ) -> LLMResponse | None:
        """return the cached response for this key, or None if there isn't one. If a
        fingerprint is supplied and the stored one differs, treat it as a miss — this is
        what catches the gateway silently pointing the endpoint at a different model."""
        conn = self._connect()
        cur = conn.execute(
            "SELECT response_json, fingerprint FROM llm_cache WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None
        stored_fp = row[1]
        if expected_fingerprint is not None and expected_fingerprint != stored_fp:
            return None
        data = json.loads(row[0])
        rm = data["resolved_model"]
        response = LLMResponse(
            content=data["content"],
            response_raw=data["response_raw"],
            resolved_model=ResolvedModel(
                endpoint_name=rm["endpoint_name"],
                api_model_id=rm["api_model_id"],
                api_version=rm.get("api_version"),
                deployment_name=rm.get("deployment_name"),
                fingerprint=rm["fingerprint"],
            ),
            usage={**(data.get("usage") or {}), "cache": "hit"},
            latency_s=0.0,
        )
        return response

    def put(self, key: str, resp: LLMResponse) -> None:
        """store resp under key; safe to re-run — it overwrites any prior entry."""
        conn = self._connect()
        payload = json.dumps(
            {
                "content": resp.content,
                "response_raw": resp.response_raw,
                "resolved_model": resp.resolved_model.to_dict(),
                "usage": dict(resp.usage or {}),
            },
            ensure_ascii=False,
        )
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache "
            "(key, endpoint_name, fingerprint, created_at, response_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                resp.resolved_model.endpoint_name,
                resp.resolved_model.fingerprint,
                time.time(),
                payload,
            ),
        )

    def close(self) -> None:
        """close the sqlite connection; safe to call twice."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
