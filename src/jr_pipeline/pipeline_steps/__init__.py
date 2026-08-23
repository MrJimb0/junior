"""The seven data-flow pipeline steps, in execution order.

The pipeline turns a patient's raw chart files into structured data fields. It
finds the passages of the chart most relevant to each question, then asks a
language model to read those passages and fill in the answer.

A few terms used throughout these steps:
  - embedding: a numeric vector that captures a chunk's meaning, so chunks with
    similar meaning sit close together — this is what makes search by meaning
    (rather than exact keyword) possible.
  - index: a data structure that finds the closest chunks to a query fast.

    step_1_ingest_raw_files              raw CSV/Excel chart files → typed table files
    step_2_embed_chunks                  split chart text into chunks → turn each into a meaning vector (embedding)
    step_3_build_vector_index            embeddings → a fast nearest-neighbor index
    step_4_retrieve_chunks               a question + filters → the most relevant chunks, ranked
    step_5_rerank_chunks                 re-score the top chunks using several signals together
    step_6_prepare_evidence_for_extraction   selected chunks → text trimmed to fit the model's input limit
    step_7_extract_variables             evidence text → language model → structured answer (JSON)
"""
