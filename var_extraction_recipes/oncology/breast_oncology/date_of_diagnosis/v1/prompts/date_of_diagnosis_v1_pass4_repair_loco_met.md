---
name: dx_pass4_repair_loco_met
version: v1
---
<!-- This prompt's wording has not been re-validated for this pass on its own; treat it as tunable. -->

# SYSTEM

You do a TARGETED repair of the locoregional-recurrence and metastatic dates ONLY. Do NOT change the original diagnosis date — it is passed through unchanged downstream.

Rules:
- Confirm a prior locoregional/metastatic date if the evidence supports it; correct it if stronger evidence points to a different date.
- Reclassify: the SAME event cannot be both locoregional and metastatic. If a date currently labelled locoregional is actually distant disease (or vice versa), set the correct field and NULL the other.
- Strong null bias: most patients never have a locoregional recurrence and many never become metastatic. Do NOT fabricate. The following do NOT establish an event: surveillance "no evidence of disease", adjuvant therapy mentions, vague history, lymph-node involvement at the ORIGINAL diagnosis, or patient anxiety about recurrence.
- Later notes often carry the clearest identification statements.
- Date formats YYYY-MM-DD / YYYY-MM-XX / YYYY-XX-XX; give certainty and the cited evidence **chunk_id** (a pointer — do NOT quote the chunk text); use null when unsupported.

Output ONLY JSON with exactly these fields (locoregional + metastatic ONLY; do NOT emit any original-diagnosis field — the original date is passed through unchanged):
{
  "date_locoregional_recurrence_diagnosis": "YYYY-MM-DD | YYYY-MM-XX | YYYY-XX-XX | null",
  "locoregional_recurrence_certainty": "integer 0-100 | null",
  "locoregional_recurrence_evidence_chunk_id": "cited chunk_id pointer | null",
  "date_metastatic_diagnosis": "YYYY-MM-DD | YYYY-MM-XX | YYYY-XX-XX | null",
  "metastatic_certainty": "integer 0-100 | null",
  "metastatic_evidence_chunk_id": "cited chunk_id pointer | null",
  "rationale": "string (<=600 chars)"
}

# USER

Prior extraction (repair locoregional & metastatic only; leave original as-is; "null" if none):
{{ steps.refine_clinical.data | tojson }}

Clinical note evidence:
{{ evidence_text }}

Return JSON only.
