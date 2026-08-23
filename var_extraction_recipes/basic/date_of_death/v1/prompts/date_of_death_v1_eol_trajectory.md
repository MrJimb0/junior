---
name: dod_eol_trajectory
version: v1
---

# SYSTEM

Summarize end-of-life trajectory evidence for this patient. Hospice, comfort care,
DNR/DNI, and terminal prognosis are trajectory signals but are not by themselves proof
of death.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"strong_signals": ["<a strong end-of-life signal from the passages>"], "moderate_signals": ["<a weaker signal>"], "latest_signal_date": "<YYYY-MM-DD>", "transfer_or_followup_elsewhere": false, "evidence_chunk_id": "<a chunk id from above>"}

# USER

End-of-life and recent clinical-note evidence:
{{ evidence_text }}

Return JSON only.
