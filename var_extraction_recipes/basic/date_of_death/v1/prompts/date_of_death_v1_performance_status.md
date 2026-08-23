---
name: dod_performance_status
version: v1
---

# SYSTEM

Summarize the patient's latest documented functional status. Extract explicit ECOG,
KPS, or PPS values and concrete decline signals. ECOG 3-4 or PPS <= 40 is a strong
end-of-life signal but not proof of death.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"latest_ecog": 0, "latest_kps": 100, "latest_pps": null, "decline_signals": ["<a decline the passages describe>"], "latest_status_date": "<YYYY-MM-DD>", "evidence_chunk_id": "<a chunk id from above>"}

# USER

Performance-status evidence:
{{ evidence_text }}

Return JSON only.
