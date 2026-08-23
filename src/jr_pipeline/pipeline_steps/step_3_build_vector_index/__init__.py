"""Build the patient's fast vector-search index from the embeddings.

Step 3 takes the per-patient embedding vectors (numeric summaries of each text
chunk's meaning, produced in Step 2) and loads them into an HNSW index
(hnswlib) — an approximate-nearest-neighbor structure that, given a query
vector, returns the closest chunks quickly instead of comparing against every
chunk one by one.
"""
