"""Pre-resolved (``direct_evidence_items``) passages merge into the step-6 package.

These are passages whose text the caller already has in hand — e.g. whole rows
from a builder-produced structured table the retrieve_and_prompt step normalized
and read itself — so they need no ``corpus`` lookup. They must: land in the
package, sit AHEAD of the reranked free text, keep their metadata, and be dropped
when empty. Step 6 includes everything (no budget packing), so there is no
crowd-out to test.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_6_prepare_evidence_for_extraction.prepare_evidence import (
    prepare_evidence,
)


class _FakeCorpus:
    """Minimal PatientChunkStore stand-in: every text chunk resolves to one text."""

    def __init__(self, text: str) -> None:
        self._text = text

    def text_for(self, chunk_id: str) -> str:  # noqa: ARG002
        return self._text

    def metadata_for(self, chunk_id: str):  # noqa: ARG002
        return None


def _direct_item(chunk_id: str, text: str, **extra):
    return {"chunk_id": chunk_id, "text": text, **extra}


def test_direct_item_lands_in_package_with_no_corpus_lookup():
    packet = prepare_evidence(
        corpus=_FakeCorpus("unused — no reranked candidates"),
        reranked_candidates=[],
        variable="v/step",
        patient_id="P",
        direct_evidence_items=[_direct_item("P:path_cap:0:TABLE", "diagnosis: invasive ductal carcinoma")],
    )
    assert packet["included_chunk_ids"] == ["P:path_cap:0:TABLE"]
    assert packet["block_count"] == 1
    assert "invasive ductal carcinoma" in packet["formatted_evidence_text"]


def test_direct_items_sit_ahead_of_reranked_text():
    packet = prepare_evidence(
        corpus=_FakeCorpus("free-text passage from a note"),
        reranked_candidates=[{"chunk_id": "P:notes:0:0", "rank": 1, "score": 0.9}],
        variable="v/step",
        patient_id="P",
        direct_evidence_items=[_direct_item("P:path_cap:0:TABLE", "specimen: left breast")],
    )
    # table row is first, then the reranked note chunk
    assert packet["included_chunk_ids"] == ["P:path_cap:0:TABLE", "P:notes:0:0"]


def test_empty_direct_item_is_dropped():
    packet = prepare_evidence(
        corpus=_FakeCorpus("note"),
        reranked_candidates=[],
        variable="v/step",
        patient_id="P",
        direct_evidence_items=[
            _direct_item("P:path_cap:0:TABLE", "   "),   # whitespace only -> dropped
            _direct_item("P:path_cap:1:TABLE", "real value"),
        ],
    )
    assert packet["included_chunk_ids"] == ["P:path_cap:1:TABLE"]


def test_direct_item_metadata_carried_into_block_record():
    packet = prepare_evidence(
        corpus=_FakeCorpus("note"),
        reranked_candidates=[],
        variable="v/step",
        patient_id="P",
        direct_evidence_items=[
            _direct_item(
                "P:path_cap:0:TABLE",
                "diagnosis: IDC",
                source_file="path_cap",
                doc_type="builder_table",
                document_date="2024-01-02",
                record_index=0,
            )
        ],
    )
    [rec] = packet["blocks"]
    assert rec["source_file"] == "path_cap"
    assert rec["doc_type"] == "builder_table"
    assert rec["document_date"] == "2024-01-02"


def test_direct_item_counts_into_doc_type_breakdown_with_token_fallback():
    packet = prepare_evidence(
        corpus=_FakeCorpus("unused"),
        reranked_candidates=[],
        variable="v/step",
        patient_id="P",
        # caller omits token_count -> chars/4 fallback in the block record
        direct_evidence_items=[_direct_item("P:path_cap:0:TABLE", "diagnosis: IDC", doc_type="builder_table")],
    )
    by_type = packet["evidence_tokens_by_doc_type"]
    assert "builder_table" in by_type
    assert by_type["builder_table"] == packet["total_evidence_tokens"]  # the only block
    assert packet["blocks"][0]["token_count"] > 0  # chars/4 fallback populated
