"""Author / title / linked_author from the chunk row reach the model in the evidence
header — "who wrote a note and when" must be visible to the model and the provenance,
not just stored. Date and doc_type were already shown; this guards the added fields."""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_6_prepare_evidence_for_extraction.prepare_evidence import (
    EvidenceBlock,
    format_one_block,
)


def test_header_carries_author_title_linked_author_and_date():
    block = EvidenceBlock(
        chunk_id="p1:clinical_note:3:0",
        rank=1,
        score=0.9,
        source_file="clinical_note.csv",
        record_index=3,
        chunk_index=0,
        text="Patient seen for follow-up.",
        token_count=6,
        doc_type="Progress Notes",
        document_date="2024-02-15",
        author="SMITH, JANE MD",
        linked_author="DOE, JOHN MD",
        title="Oncology Follow-up Note",
    )
    header = format_one_block(block).splitlines()[0]
    assert "title=Oncology Follow-up Note" in header
    assert "author=SMITH, JANE MD" in header
    assert "linked_author=DOE, JOHN MD" in header
    assert "date=2024-02-15" in header


def test_header_shows_dash_when_metadata_absent():
    block = EvidenceBlock(
        chunk_id="p1:labs:0:0", rank=1, score=0.5, source_file="labs.csv",
        record_index=0, chunk_index=0, text="WBC 5.1", token_count=3,
    )
    header = format_one_block(block).splitlines()[0]
    assert "author=—" in header
    assert "title=—" in header
