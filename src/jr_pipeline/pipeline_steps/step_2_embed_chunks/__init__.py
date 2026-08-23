"""Embed module — split source text into short chunks and turn each into a vector.

Step 2 reads the cleaned-up tables from ingest, breaks each text field into
chunks, and encodes every chunk into an embedding (a numeric vector capturing
its meaning) so later steps can search by similarity.
"""
from __future__ import annotations

from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import (
    _DEFAULT_DEVICE,
    _DEFAULT_DTYPE,
    _DEFAULT_LOCAL_FILES_ONLY,
    _DEFAULT_MAX_TOKENS,
    _DEFAULT_NORMALIZE,
    _DEFAULT_POOLING,
    Encoder,
    HFEncoder,
)

# In-memory cache of loaded encoders, keyed by the config values that determine
# an encoder's identity — the same config returns the same already-loaded model
# so retrieve steps don't reload the (large) model on every call.
# The model's weight content hash (model_sha256) is intentionally absent from the
# key: it is computed from disk after the model loads, not available from the
# config dict alone. So if someone overwrites the weights file in place but keeps
# the same model path, this cache will NOT notice within a running process — it
# only refreshes on a process restart.
_ENCODER_CACHE: dict[tuple, HFEncoder] = {}


def build_encoder(cfg: dict) -> Encoder:
    """Build (or return a cached) encoder for the given config."""
    key = (
        cfg["model_id"],
        cfg.get("pooling", _DEFAULT_POOLING),
        int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)),
        cfg.get("device", _DEFAULT_DEVICE),
        bool(cfg.get("normalize", _DEFAULT_NORMALIZE)),
        cfg.get("dtype", _DEFAULT_DTYPE),
        bool(cfg.get("local_files_only", _DEFAULT_LOCAL_FILES_ONLY)),
        # a config that pins weight hashes must not reuse an encoder built by a
        # config that didn't — the pins would silently go unverified.
        tuple(sorted((cfg.get("expected_file_sha256") or {}).items())),
    )
    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = HFEncoder(
            model_id=cfg["model_id"],
            pooling=cfg.get("pooling", _DEFAULT_POOLING),
            max_tokens=int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS)),
            device_preference=cfg.get("device", _DEFAULT_DEVICE),
            normalize=bool(cfg.get("normalize", _DEFAULT_NORMALIZE)),
            expected_file_sha256=cfg.get("expected_file_sha256") or None,
            dtype=cfg.get("dtype", _DEFAULT_DTYPE),
            local_files_only=bool(cfg.get("local_files_only", _DEFAULT_LOCAL_FILES_ONLY)),
        )
    return _ENCODER_CACHE[key]


__all__ = ["Encoder", "HFEncoder", "build_encoder"]
