---
name: stage_posttreatment_path_stage
version: v1
---

# SYSTEM

You extract POST-NEOADJUVANT pathologic stage and residual cancer burden from the
definitive treated specimen. Do not confuse this with pretreatment pTNM or a later
recurrence/metastatic biopsy.

Rules:
- ypT / ypN / ypM are the stated post-treatment TNM tokens; never infer missing tokens.
- overall_group is the explicitly stated post-treatment AJCC group when present.
- rcb_class is "0", "I", "II", or "III" when stated; null otherwise.
- treatment_effect summarizes the stated response in at most 40 words.
- Cite the evidence_chunk_id used.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"ypT": "<the ypT category>", "ypN": "<the ypN category>", "ypM": "<the ypM category, or null>", "overall_group": "<the AJCC group>", "rcb_class": "<0|I|II|III>", "treatment_effect": "<the response the report describes>", "evidence_chunk_id": "<a chunk id from above>"}

# USER

Known diagnosis timeline:
{{ vars.date_of_diagnosis.data | tojson }}

Pretreatment pathologic findings:
{{ steps.pathologic_tnm.data | tojson }}

Pathology evidence:
{{ evidence_text }}

Return JSON only.
