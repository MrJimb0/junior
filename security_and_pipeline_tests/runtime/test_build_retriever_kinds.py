"""Every retriever kind a recipe can name must construct from a plain config dict.

Construction-only: no corpus, no model load. This is the seam recipes cross by
string (`retrieval: kind: ...`), so a signature drift between build_retriever and
a retriever's __init__ breaks recipes at runtime — as a config dict, nothing the
type checker sees.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_4_retrieve_chunks.retrieve import build_retriever

_ENCODER_CFG = {"model_id": "fake-model", "max_tokens": 512}


def test_every_recipe_nameable_kind_constructs():
    cases = {
        "bm25": {"kind": "bm25", "source_file": "notes.csv"},
        "exact": {"kind": "exact", "case_insensitive": True},
        "embedding": {"kind": "embedding", "encoder": _ENCODER_CFG},
        "direct_parquet": {
            "kind": "direct_parquet",
            "table": "demographics",
            "filter": "status == 'active'",
        },
        "hybrid": {"kind": "hybrid", "source_file": "notes.csv", "encoder": _ENCODER_CFG},
    }
    for kind, cfg in cases.items():
        retriever = build_retriever(cfg, encoder_cfg_fallback=_ENCODER_CFG)
        assert retriever.info.kind == kind
