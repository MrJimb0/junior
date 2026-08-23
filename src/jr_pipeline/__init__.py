"""jr_pipeline — auditable, per-patient clinical RAG extraction pipeline.

CLI (entry point):

    jr-pipeline ingest     raw CSV/xlsx → typed parquet
    jr-pipeline embed      chunk text → vectors
    jr-pipeline index      vectors → ANN index
    jr-pipeline retrieve   text query + filters → ranked chunks
    jr-pipeline extract    recipe-driven LLM variable extraction
                           (reranking + evidence prep run inside extract)

Source layout:

    runtime_infrastructure/
        Execution engine: CLI, config loading, file layout, logging, caching.

    pipeline_steps/
        The data-flow stages, numbered step_1_ through step_8_.

    runtime_enforcing_safety_and_reproducibility/
        Trust layer that fires on every pipeline invocation.
            phi/              PHI containment runtime checks
            reproducibility/  code-bundle sealing, manifests, scoped hashes
            schemas/          artifact format contracts (Pydantic + JSON Schema)
            content_fingerprinting.py    canonical sha256 primitives
            pipeline_progress_log.py     append-only state-machine log

    evaluating_pipeline_performance/
        Run analysis, metrics, retrieval/extraction comparisons.

Data layout (see data/README.md):

    data/CONTAINS_PHI/                   patient text, LLM messages, results
    data/NO_PHI__shareable/              embeddings, scores, code snapshots
"""

__version__ = "0.0.1"
