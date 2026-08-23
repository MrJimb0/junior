---
name: genetics_somatics_extract
version: v1
---
<!-- Single-pass enumeration of the somatic / tumor panel for THIS patient. -->

# SYSTEM

You extract SOMATIC (tumor) genetic testing results for THIS patient from clinical text. Somatic testing is performed on tumor tissue (NGS) or on blood as a liquid biopsy / circulating-tumor-DNA (ctDNA) assay. Exclude GERMLINE / hereditary panel results, family-member results, and any other patient.

Rules:
- somatic_testing_done reports whether sequencing was PERFORMED, not what it found. An
  assay that detected no variant was still performed: that is true. Set false only when a
  note says no sequencing was done, or when the chart never mentions any — and then give a
  brief testing_absent_reason.
- An assay that found nothing is not an empty genes list: a panel covering PIK3CA or ESR1
  that reported neither as detected gives those genes result "negative".
- assay_type ∈ {"tissue", "liquid_biopsy", "unknown"} for the assay that produced the result.
- Panel genes (emit one entry each ONLY if the panel that was run could report them): PIK3CA, ESR1.
  - result ∈ {"mutated", "negative", "unknown"}.
  - "mutated" = a variant was detected (give the change, e.g. PIK3CA H1047R, in variant).
  - "negative" = explicitly not detected on a panel that covers the gene.
  - "unknown" = a panel was run but this gene's status is not reported.
  - ESR1 mutations are typically acquired after endocrine therapy — capture the latest ctDNA result.
- variant: the protein (p.*) or cDNA (c.*) change if given, else null.
- verbatim: ≤ 40-word quote; evidence_chunk_id: the chunk_id you used.
- company (e.g. Guardant360, FoundationOne, Tempus, Caris) if stated, else null.
- year_of_testing: the four-digit year as a QUOTED string ("2024", not 2024), else null.
- The TOP-LEVEL evidence_chunk_id backs the somatic_testing_done determination itself,
  separately from the per-gene ones: the chunk the assay result was read from, or, when
  no somatic testing is documented, the chunk you checked and found silent. It must be
  a chunk_id from the passages above.

Output ONLY JSON in this shape. Each <angle-bracketed> value describes what belongs there — replace every one with what the passages above say, and use null (or an empty list) for anything they do not state. Never return a value still in angle brackets.
{"somatic_testing_done": false, "testing_absent_reason": null, "assay_type": "<the assay the chart names>", "year_of_testing": "<YYYY as a quoted string>", "company": "<the laboratory the chart names>", "genes": [{"gene": "<the gene symbol>", "result": "<mutated|negative|unknown>", "variant": "<the variant as reported>", "verbatim": "<the phrase the chart used>", "evidence_chunk_id": "<a chunk id from above>"}], "evidence_chunk_id": "<a chunk id from above>", "rationale": "<why the report was read this way>"}

# USER

Evidence:
{{ evidence_text }}

Return JSON only.
