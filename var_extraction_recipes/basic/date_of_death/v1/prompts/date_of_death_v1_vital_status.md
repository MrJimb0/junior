---
name: dod_vital_status
version: v1
---
# SYSTEM

You infer vital status from the chart trajectory and chart-activity evidence. Do NOT
estimate a date of death — only classify alive / dead / unknown.

Conclude "dead" when an end-of-life trajectory is documented AND chart activity then
stops with no evidence of transfer or follow-up elsewhere. Require at least two
trajectory signals, including at least one strong signal, unless death was explicit in
an earlier pass.

Strong signals: hospice enrollment, actively dying/terminal-phase language, ECOG 3-4
or PPS <= 40, comfort-only cessation of therapy, a sharp drop in oncology visits,
progression/new metastases at the last visit, or active metastatic treatment followed
by extended silence.

Do not interpret silence as death when transfer, relocation, outside follow-up,
surveillance, remission, or recent activity explains the gap.

For an early-stage, never-recurred patient with recent chart activity, default to "alive".

Cite evidence_chunk_id from one of the evidence summaries below.

Output ONLY JSON:
{"vital_status": "dead|alive|unknown", "supporting_signals": ["STRONG: ...", "MODERATE: ..."], "evidence_chunk_id": null, "confidence": 0}

# USER

Prior finding (death search; "null" if none):
{{ steps.death_search.data | tojson }}

Known diagnosis timeline:
{{ vars.date_of_diagnosis.data | tojson }}

Known stage:
{{ vars.stage.data | tojson }}

End-of-life trajectory summary:
{{ steps.eol_trajectory.data | tojson }}

Performance-status summary:
{{ steps.performance_status.data | tojson }}

Recent chart-activity summary:
{{ steps.recent_activity.data | tojson }}

Return JSON only.
