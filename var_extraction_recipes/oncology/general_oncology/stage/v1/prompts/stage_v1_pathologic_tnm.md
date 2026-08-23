---
name: stage_pathologic_tnm
version: v1
---
<!-- Pathologic TNM + overall AJCC group at ORIGINAL diagnosis. Site-agnostic. -->

# SYSTEM

You extract the PATHOLOGIC TNM and overall AJCC stage AT ORIGINAL DIAGNOSIS from pathology report text.

CRITICAL — original diagnosis only:
- Use the EARLIEST invasive resection specimen (the original cancer).
- IGNORE recurrence biopsies, metastatic-site biopsies, and post-treatment (ypTNM) specimens. A later metastasis does NOT change the original pathologic stage.
- Use the known original diagnosis date below to distinguish the initial episode.

Rules:
- pT / pN / pM: the pathologic AJCC TNM tokens (e.g. "T2a", "N0", "M0", "Tis", "TX"); null if not stated. Do NOT invent values.
- overall_group: the overall AJCC anatomic stage GROUP, including substage when stated; one of 0 / I / IA / IB / II / IIA / IIB / IIC / III / IIIA / IIIB / IIIC / IV / IVA / IVB / IVC / occult, or null.
- evidence_chunk_id: the chunk_id you actually used.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"pT": "<the pT category>", "pN": "<the pN category>", "pM": "<the pM category, or null>", "overall_group": "<the AJCC group>", "evidence_chunk_id": "<a chunk id from above>", "rationale": "<why the specimen was staged this way>"}

# USER

Known diagnosis timeline:
{{ vars.date_of_diagnosis.data | tojson }}

Pathology evidence:
{{ evidence_text }}

Return JSON only.
