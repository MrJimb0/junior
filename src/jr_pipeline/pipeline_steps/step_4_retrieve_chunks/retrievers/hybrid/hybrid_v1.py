"""Run several retrievers and blend their answers into one ranked list.

The default way to merge is reciprocal rank fusion (RRF) — combine ranked lists
by rank position, not by raw score: each chunk earns 1 / (k0 + rank) from every
member that returned it, summed across members. Because RRF looks only at where a
chunk landed in each member's list (1st, 2nd, ...) and ignores the raw score, it
works even when members produce scores on totally different scales (a BM25
keyword score vs. an embedding similarity).

If one member fails (for example its embedding index is missing) it contributes
an empty list and the other members still produce a result. If EVERY member
fails, the query raises an error rather than silently returning nothing.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    Candidate,
    PatientChunkStore,
    Retriever,
    RetrieverInfo,
)

_log = get_logger("hybrid")


def _sanitized_hash(message: str) -> str:
    """Turn a raw error message into a short fixed fingerprint (16 hex characters).
    Identical failures produce the same fingerprint so they can be counted/correlated,
    while the raw text — which could contain patient data (PHI) — is never logged."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]

_DEFAULT_K0 = 60  # value from the original RRF paper; keeps the very top ranks from dominating
_MIN_MEMBER_FETCH = 10
# Ask each member for more than the final k chunks ("over-fetch"), so that after
# members drop their own low-quality hits there are still enough left to fill the
# final merged top-k.
_MEMBER_FETCH_MULTIPLIER = 2

_VERSION = "v1"

# ── recipe config options ─────────────────────────────────────────────────────
# Set in your recipe YAML under retrieval:
#
#   kind:    hybrid
#   fusion:  reciprocal_rank   merge members by rank position (RRF) — works even
#                              when members use different score scales (keyword
#                              bm25 + meaning-based embedding); use weighted_sum
#                              only when all members share one score scale
#   k0:      60                RRF smoothing constant — larger values shrink the
#                              gap between a 1st-place and a lower-ranked hit;
#                              default is the value from the original RRF paper
#   members:                   list of retriever configs (any kind except hybrid)
#   weights:                   optional per-member weights, e.g. [1.0, 2.0] to
#                              double the contribution of the second member;
#                              only meaningful with fusion: weighted_sum
#
# Example:
#   retrieval:
#     kind: hybrid
#     k: 10
#     members:
#       - kind: bm25
#         query: "metastatic systemic treatment chemotherapy"
#       - kind: embedding
#         query: "metastatic systemic treatment chemotherapy"


@dataclass
class HybridRetriever:
    """Runs multiple retrievers and merges their results into one ranked list."""

    info: RetrieverInfo

    def __init__(
        self,
        *,
        members: list[Retriever],
        fusion: str = "reciprocal_rank",
        k0: int = _DEFAULT_K0,
        weights: list[float] | None = None,
    ):
        if not members:
            raise ValueError("HybridRetriever needs at least one member retriever")
        if fusion not in {"reciprocal_rank", "weighted_sum"}:
            raise ValueError(f"Unknown fusion strategy: {fusion!r}")
        if weights is not None and len(weights) != len(members):
            raise ValueError("weights must match members length")
        self._members = members
        self._fusion = fusion
        self._k0 = k0
        self._weights = weights or [1.0] * len(members)
        # PHI-safe summary of how each member did on the most recent query (counts and
        # error types only, no raw text) — recorded into the retrieval audit record.
        self.last_member_outcomes: list[dict] = []
        self.info = RetrieverInfo(
            kind="hybrid",
            version=_VERSION,
            config={
                "fusion": fusion,
                "k0": k0,
                "weights": self._weights,
                "members": [
                    {"kind": m.info.kind, "version": m.info.version, "config": m.info.config}
                    for m in members
                ],
            },
        )

    @property
    def score_normalization(self) -> str:
        return "reciprocal_rank" if self._fusion == "reciprocal_rank" else "raw"

    def query(
        self,
        corpus: PatientChunkStore,
        *,
        text: str,
        k: int,
    ) -> list[Candidate]:
        """Ask every member retriever for ``text`` and merge their ranked lists into one top-k."""
        per_member: list[list[Candidate]] = []
        outcomes: list[dict] = []
        member_k = max(k * _MEMBER_FETCH_MULTIPLIER, _MIN_MEMBER_FETCH)
        for m in self._members:
            try:
                results = m.query(corpus, text=text, k=member_k)
                per_member.append(results)
                outcomes.append({
                    "kind": m.info.kind, "n": len(results), "error_kind": None,
                    "sanitized_message_hash": None,
                })
            except Exception as exc:  # noqa: BLE001 — one bad member must not crash the rest
                # Record only safe fields: the raw message may contain patient data
                # (PHI), so it never reaches this PHI-free log — just a fixed fingerprint
                # of the message plus the error's type name.
                per_member.append([])
                outcomes.append({
                    "kind": m.info.kind, "n": 0, "error_kind": type(exc).__name__,
                    "sanitized_message_hash": _sanitized_hash(str(exc)),
                })
                _log.warning("hybrid_member_failed", extra_={
                    "member_kind": m.info.kind, "error_kind": type(exc).__name__,
                    "sanitized_message_hash": _sanitized_hash(str(exc)),
                })
        self.last_member_outcomes = outcomes
        if outcomes and all(o["error_kind"] is not None for o in outcomes):
            raise RuntimeError(
                "all hybrid member retrievers failed: "
                + ", ".join(f"{o['kind']}={o['error_kind']}" for o in outcomes)
            )

        fused: dict[str, dict[str, float | str | None]] = {}
        for mi, cands in enumerate(per_member):
            weight = self._weights[mi]
            member_kind = self._members[mi].info.kind
            for c in cands:
                contrib = self._contribution(c, weight)
                entry = fused.setdefault(c.chunk_id, {"score": 0.0, "retriever": member_kind})
                entry["score"] = float(entry["score"]) + contrib

        # Sort by fused score, highest first; break exact ties by chunk_id so the
        # ordering is identical every time the same query is run.
        ordered = sorted(fused.items(), key=lambda kv: (-float(kv[1]["score"]), kv[0]))
        out: list[Candidate] = []
        for i, (cid, entry) in enumerate(ordered[:k]):
            out.append(
                Candidate(
                    chunk_id=cid,
                    rank=i + 1,
                    score=float(entry["score"]),
                    retriever=str(entry.get("retriever") or ""),
                )
            )
        return out

    def _contribution(self, cand: Candidate, weight: float) -> float:
        if self._fusion == "reciprocal_rank":
            return weight * (1.0 / (self._k0 + cand.rank))
        return weight * cand.score
