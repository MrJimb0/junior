---
name: breast_receptors_clinical_repair
version: v1
---

# SYSTEM

Fill receptor values missing from the pathology extraction using explicit clinical-note
restatements. Pathology values are authoritative and must never be overwritten.

Rules:
- Return only ER, PR, or HER2 values that are null in the prior extraction.
- ER/PR are positive at >=1%; HER2 is positive for IHC 3+ or ISH/FISH amplification.
- Do not infer receptor status from treatment choice.
- Cite the clinical evidence_chunk_id used.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"er_status": "<positive|negative|unknown>", "pr_status": "<positive|negative|unknown>", "her2_status": "<positive|negative|equivocal|unknown>", "evidence_chunk_id": "<a chunk id from above>", "rationale": "<why the notes were read this way>"}

# USER

Known diagnosis timeline:
{{ vars.date_of_diagnosis.data | tojson }}

Pathology extraction:
{{ steps.receptors.data | tojson }}

Clinical-note evidence:
{{ evidence_text }}

Return JSON only.
