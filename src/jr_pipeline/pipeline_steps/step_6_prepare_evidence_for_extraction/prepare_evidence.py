"""Assemble the evidence text for one patient and one field to extract.

Step 6 is a consolidator, not a gate. Step 5 is the gate

It does NOT trim or drop passages to fit a budget. The one guard is a sanity ceiling (``max_context_tokens``)

Token counts here are a rough estimate (characters / 4)

``prepare_evidence`` does no reading or writing of files — it just returns the
assembled packet. Saving it to disk happens later: Step 8 (organize_output) takes
the packet and writes the patient-identifiable evidence files (these live under
the CONTAINS_PHI tree, the part of the output that may hold protected health
information):
    CONTAINS_PHI/prepared_evidence_text/       the formatted evidence (clinical text)
    CONTAINS_PHI/evidence_selection_metadata/  what was assembled (IDs + token counts)

A separate, shareable version — stripped of all clinical text and patient
identifiers — is written to the NO_PHI tree by
``runtime_enforcing_safety_and_reproducibility.evidence_selection_trace``,
triggered from the retrieve_and_prompt step where the retriever and recipe
details are available.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore

# A generous pre-send sanity ceiling, in estimated tokens (chars / 4). Normal
# evidence (a handful of re-ranked chunks) is a few thousand tokens, well under
# this; the ceiling exists only to catch a pathological assembly (e.g. a giant
# table row) before it reaches the model. A recipe can tighten it to match a
# specific model's real context window via evidence.max_context_tokens.
DEFAULT_MAX_CONTEXT_TOKENS = 200_000


class EvidenceTooLargeError(ValueError):
    """Raised when the assembled evidence exceeds ``max_context_tokens`` — we fail
    loudly here rather than let the model silently truncate an over-long prompt."""


# --- evidence blocks: the data type + the source-labeled formatting ----------
#
# Each block carries a header showing where the evidence came from (source file
# type, record number, position within the note) so the language model can cite
# its sources and a human reviewer can trace it back to the original note.
# Example of a rendered block:
#
#     [Evidence 1 | chunk_id=… | doc_type=pathology_report | date=2024-01-02 | …]
#     Invasive ductal carcinoma, Nottingham Grade 2, ER+/PR+/HER2-…


@dataclass(frozen=True)
class EvidenceBlock:
    chunk_id: str
    rank: int
    score: float
    source_file: str
    record_index: int
    chunk_index: int
    text: str
    token_count: int
    doc_type: str | None = None
    document_date: str | None = None
    author: str | None = None
    linked_author: str | None = None
    title: str | None = None


def _fmt(value: object) -> str:
    """Show a metadata value, or an em-dash (—) placeholder when it's missing."""
    return str(value) if value not in (None, "") else "—"


def format_one_block(b: EvidenceBlock) -> str:
    """Render one evidence block (its source header followed by its text). The
    header carries the chunk_id (the passage's unique identifier) so the model can
    cite it — the prompts tell the model to "cite the chunk_id you used" — plus the
    title, document type, date, and author so the answer can be traced to its source
    and to who wrote it. This same function builds the final package and estimates the
    block's token cost, so both agree on exactly what the model will see."""
    header = (
        f"[Evidence {b.rank} | chunk_id={b.chunk_id} | title={_fmt(b.title)} | "
        f"doc_type={_fmt(b.doc_type)} | date={_fmt(b.document_date)} | "
        f"author={_fmt(b.author)} | linked_author={_fmt(b.linked_author)} | "
        f"{b.source_file} | record {b.record_index} | chunk {b.chunk_index} | score {b.score:.3f}]"
    )
    return f"{header}\n{b.text}"


def format_evidence_blocks(blocks: list[EvidenceBlock]) -> str:
    """Join the evidence blocks into one formatted string."""
    return "\n\n".join(format_one_block(b) for b in blocks)


def estimate_block_tokens(b: EvidenceBlock) -> int:
    """Estimate one rendered block's token cost (characters / 4, the ``[Evidence N
    | …]`` header included because that is part of what the model reads). A rough
    estimate for the sanity guard and the per-document-type breakdown — the real
    per-call usage comes from the model API and is recorded in the step-7 receipt."""
    return max(1, len(format_one_block(b)) // 4)


# --- prepare_evidence: the step-6 entry point --------------------------------


def prepare_evidence(
    *,
    corpus: PatientChunkStore,
    reranked_candidates: list[dict[str, Any]],
    variable: str,
    patient_id: str,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    direct_evidence_items: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Consolidate the chosen evidence for one patient and one field into one
    labeled package, and measure it.

    Includes everything step 5 selected.

    ``direct_evidence_items`` are passages whose text the caller has ALREADY
    resolved — e.g. whole rows from a builder-produced structured table the
    retrieve_and_prompt step normalized and read itself, rather than chunks looked
    up from ``corpus``. Each item is a dict ``{chunk_id, text, source_file?,
    doc_type?, document_date?, record_index?, chunk_index?, rank?, score?}``. They
    are placed AHEAD of the re-ranked free-text passages (this is high-value,
    recipe-named context) and flow through the same formatting and counting."""
    blocks = _build_direct_evidence_blocks(direct_evidence_items) + _build_evidence_blocks(
        corpus, reranked_candidates
    )

    tokens_by_doc_type: dict[str, int] = defaultdict(int)
    total_tokens = 0
    for b in blocks:
        cost = estimate_block_tokens(b)
        total_tokens += cost
        tokens_by_doc_type[b.doc_type or "unknown"] += cost

    if total_tokens > max_context_tokens:
        raise EvidenceTooLargeError(
            f"assembled evidence for {variable} (patient {patient_id}) is "
            f"~{total_tokens} tokens, over the {max_context_tokens}-token ceiling. "
            "Tighten the step-5 selection (filters / top_n) or raise "
            "evidence.max_context_tokens to your model's real context window."
        )

    return {
        "patient_id": patient_id,
        "variable": variable,
        "formatted_evidence_text": format_evidence_blocks(blocks),
        "block_count": len(blocks),
        # Estimated (chars/4) — for the sanity guard + burn awareness, not billing.
        "total_evidence_tokens": total_tokens,
        "evidence_tokens_by_doc_type": dict(tokens_by_doc_type),
        "max_context_tokens": max_context_tokens,
        "included_chunk_ids": [b.chunk_id for b in blocks],
        "blocks": [_block_record(b) for b in blocks],
    }


def _block_record(b: EvidenceBlock) -> dict[str, Any]:
    """Full patient-identifiable record for one block — everything Step 8 needs to
    write prepared_evidence_text/evidence_blocks.json."""
    return {
        "chunk_id": b.chunk_id,
        "rank": b.rank,
        "score": b.score,
        "source_file": b.source_file,
        "doc_type": b.doc_type,
        "document_date": b.document_date,
        "record_index": b.record_index,
        "chunk_index": b.chunk_index,
        "text": b.text,
        "token_count": b.token_count,
    }


def _build_direct_evidence_blocks(
    items: Sequence[dict[str, Any]],
) -> list[EvidenceBlock]:
    """Turn caller-resolved passages (text already in hand — e.g. builder-table
    rows) into EvidenceBlocks, no corpus lookup. Items whose text is empty are
    dropped, the same rule ``_build_evidence_blocks`` applies, so an empty block
    can't add a header with no evidence. ``rank`` defaults to the item's position
    and ``score`` to 1.0; the token estimate falls back to chars/4 when the caller
    doesn't supply one."""
    blocks: list[EvidenceBlock] = []
    for position, item in enumerate(items, start=1):
        text = item.get("text") or ""
        if not text.strip():
            continue
        token_count = int(item.get("token_count") or max(1, len(text) // 4))
        blocks.append(EvidenceBlock(
            chunk_id=item["chunk_id"],
            rank=int(item.get("rank", position)),
            score=float(item.get("score", 1.0)),
            source_file=item.get("source_file", ""),
            record_index=int(item.get("record_index", 0)),
            chunk_index=int(item.get("chunk_index", 0)),
            text=text,
            token_count=token_count,
            doc_type=item.get("doc_type"),
            document_date=item.get("document_date"),
            author=item.get("author"),
            linked_author=item.get("linked_author"),
            title=item.get("title"),
        ))
    return blocks


def _build_evidence_blocks(
    corpus: PatientChunkStore,
    reranked_candidates: list[dict[str, Any]],
) -> list[EvidenceBlock]:
    """Turn the re-ranked candidate passages into full EvidenceBlocks by looking up
    each one's text and metadata. Candidates whose text can't be found are dropped:
    an empty block with only a header would add no actual evidence."""
    blocks: list[EvidenceBlock] = []

    for c in reranked_candidates:
        chunk_id = c["chunk_id"]

        try:
            text = corpus.text_for(chunk_id)
        except (KeyError, IndexError, FileNotFoundError):
            text = ""
        if not (text and text.strip()):
            continue

        source_file = ""
        record_index = 0
        chunk_idx = 0
        token_count = 0
        doc_type = None
        document_date = None
        author = None
        linked_author = None
        title = None

        # The chart metadata for this chunk: its chunk_index row for an embedded
        # chunk, the structured parquet row for a ":TABLE" pseudo-chunk (a whole
        # table row treated as one piece of evidence). None for a chunk id the
        # corpus doesn't recognize. Header metadata is read the same way step 5
        # filtered on it, so evidence selected by author or date shows that author
        # and date rather than a blank header.
        row = corpus.metadata_for(chunk_id)
        if row is not None:
            source_file = row.get("source_file") or ""
            doc_type = row.get("doc_type")
            document_date = row.get("document_date")
            author = row.get("author")
            linked_author = row.get("linked_author")
            title = row.get("title")
        # Only an embedded chunk has a position within its source row; a table row
        # takes its record index from the chunk_id and is counted by its text.
        if chunk_id.endswith(":TABLE"):
            row_str = chunk_id[: -len(":TABLE")].rpartition(":")[2]
            record_index = int(row_str) if row_str.isdigit() else 0
        elif row is not None:
            record_index = row.get("row_id", 0)
            chunk_idx = row.get("chunk_idx", 0)
            token_count = row.get("token_count", len(text) // 4)

        if token_count == 0:
            token_count = len(text) // 4

        blocks.append(EvidenceBlock(
            chunk_id=chunk_id,
            rank=c.get("rank", 0),
            score=c.get("score", 0.0),
            source_file=source_file,
            record_index=record_index,
            chunk_index=chunk_idx,
            text=text,
            token_count=token_count,
            doc_type=doc_type,
            document_date=document_date,
            author=author,
            linked_author=linked_author,
            title=title,
        ))

    return blocks
