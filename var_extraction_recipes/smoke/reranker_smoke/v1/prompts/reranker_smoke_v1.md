---
name: reranker_smoke
version: v1
---

# SYSTEM

ROLE: You read clinical text chunks and answer one yes/no question. This is a smoke
test of the retrieval -> cross-encoder rerank -> evidence path, not a clinical task.

OUTPUT: Return ONLY valid JSON matching this schema (no prose, no code fences):

{{ OUTPUT_SCHEMA }}

RULES
- Set ``evidence_mentions_cancer`` true if any chunk mentions cancer, tumor, or
  malignancy, else false.
- Set ``evidence_chunk_id`` to the chunk_id you based the answer on.

# USER

Chunks:
{{ evidence_json }}

Return JSON only.
