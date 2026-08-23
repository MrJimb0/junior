"""The shareable encoder identity must not depend on where the model sits on disk.

The NO_PHI exhaust exists to be pooled across institutions, so an encoder has to
fingerprint the same wherever it ran. Hashing the config verbatim broke that twice:
`junior run` resolves model_id to an absolute path before the config gets here, while
the per-stage commands hash the raw relative string — so one laptop produced two
different encoder identities for one settings file depending on which command was
typed — and the cohort one wrote a home directory into a value meant to be shared.

The retrieval layer already draws this line: verify_corpus_encoder_alignment excludes
model_id from its comparison because a path does not affect the vectors.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
    encoder_fingerprint,
)

_AN_ENCODER = {
    "model_id": "./models/embedding/BioClinical-ModernBERT-base",
    "pooling": "mean", "normalize": True, "max_tokens": 512, "dtype": "float16",
}


def test_the_same_encoder_at_a_different_path_is_the_same_encoder():
    on_a_laptop = dict(_AN_ENCODER)
    on_a_cluster = {**_AN_ENCODER, "model_id": "/shared/lab/shared/models/BioClinical-ModernBERT-base"}
    resolved_absolute = {**_AN_ENCODER, "model_id": "/Users/someone/Desktop/junior/models/embedding/BioClinical-ModernBERT-base"}

    identities = {encoder_fingerprint(cfg)
                  for cfg in (on_a_laptop, on_a_cluster, resolved_absolute)}

    assert len(identities) == 1, f"the same encoder fingerprinted {len(identities)} ways"


def test_a_setting_that_changes_the_vectors_changes_the_identity():
    """Excluding the path must not make the fingerprint blind to what it is for."""
    base = encoder_fingerprint(_AN_ENCODER)

    for field, other in (("pooling", "cls"), ("normalize", False),
                         ("max_tokens", 256), ("dtype", "float32")):
        assert encoder_fingerprint({**_AN_ENCODER, field: other}) != base, field


def test_a_different_model_is_a_different_identity():
    """Dropping the path is not enough on its own. If the model is not named at all,
    two unrelated encoders sharing pooling/max_tokens/dtype fingerprint the same, and
    pooled data from two models merges silently — worse than the path problem."""
    base = encoder_fingerprint(_AN_ENCODER)

    for other_model in ("./models/embedding/pubmedbert-base",
                        "NeuML/bioclinical-modernbert-base-embeddings",
                        "/shared/lab/models/some-other-encoder"):
        assert encoder_fingerprint({**_AN_ENCODER, "model_id": other_model}) != base, other_model


def test_a_hub_id_keeps_its_publisher():
    """Reducing "NeuML/bioclinical-modernbert-base-embeddings" to its last component
    would drop the org and let two publishers' models of the same name collide. A hub
    id is a name already, not a location."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        _model_identity,
    )

    assert _model_identity("NeuML/bioclinical-modernbert-base-embeddings") == (
        "NeuML/bioclinical-modernbert-base-embeddings")
    assert _model_identity("./models/embedding/thomas-sounack:BioClinical-ModernBERT-base23APR2026") == (
        "thomas-sounack:BioClinical-ModernBERT-base23APR2026")
    assert _model_identity("/shared/lab/shared/models/thomas-sounack:BioClinical-ModernBERT-base23APR2026") == (
        "thomas-sounack:BioClinical-ModernBERT-base23APR2026")


def test_a_machine_fact_is_not_part_of_the_identity():
    """device is where it ran, not what ran. Folding it in means a laptop and a cluster
    can never agree about the same encoder — the pooling failure again, one field over."""
    on_a_laptop = {**_AN_ENCODER, "device": "auto", "local_files_only": True}
    on_a_cluster = {**_AN_ENCODER, "device": "cuda", "local_files_only": False}

    assert encoder_fingerprint(on_a_laptop) == encoder_fingerprint(on_a_cluster)


def test_no_path_reaches_the_hash():
    """A home directory in a value meant to be shared is a disclosure as well as a
    correctness problem."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
        _model_identity,
    )

    named = _model_identity("/Users/someone/Desktop/junior/models/embedding/an-encoder")

    assert named == "an-encoder"
    assert "/" not in named


def test_both_call_sites_use_it():
    """Two callers computing this independently is how they diverged."""
    import inspect
    import re

    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import (
        retrieve_and_prompt_step,
    )
    from jr_pipeline.runtime_infrastructure import cohort_runner

    for module in (cohort_runner, retrieve_and_prompt_step):
        source = inspect.getsource(module)
        assert "encoder_fingerprint(" in source, module.__name__
        # A word boundary, because encoder_fingerprint( ends with fingerprint( — a
        # plain substring test passes whether or not the fix is there.
        bare = re.search(
            r"(?<![_\w])fingerprint\((ctx\.encoder_cfg|embed_cfg\.get\(\"encoder\"\))",
            source,
        )
        assert bare is None, f"{module.__name__} still hashes the encoder config verbatim"
