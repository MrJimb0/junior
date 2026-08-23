---
name: dod_recent_activity
version: v1
---

# SYSTEM

Summarize the newest chart activity relevant to whether this patient remained alive,
transferred care, entered surveillance/remission, or stopped appearing after active
declining cancer care. Do not classify vital status in this pass.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"last_chart_activity_date": "<YYYY-MM-DD>", "ongoing_treatment_or_followup": false, "transfer_or_relocation": false, "surveillance_or_remission": false, "unexplained_silence_after_decline": false, "evidence_chunk_id": "<a chunk id from above>"}

# USER

Newest and activity-related clinical-note evidence:
{{ evidence_text }}

Return JSON only.
