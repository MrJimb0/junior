"""Retired config keys are rejected with a migration error, not silently inert.

`extends:` (base-chain composition) and `encoder.use_safetensors` are not valid keys. A
leftover one would otherwise be ignored — a child config would silently lose inherited keys,
and a .bin-only use_safetensors:false would be dropped and later fail with a generic error.
"""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.config_loading import (
    load_config,
    validate_embed_config,
)


def test_load_config_rejects_extends(tmp_path):
    cfg = tmp_path / "child.yaml"
    cfg.write_text("extends: base.yaml\nrun_id: r\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extends"):
        load_config(cfg)


def test_load_config_accepts_a_self_contained_config(tmp_path):
    cfg = tmp_path / "ok.yaml"
    cfg.write_text("run_id: r\n", encoding="utf-8")
    assert load_config(cfg) == {"run_id": "r"}


def test_validate_embed_config_rejects_use_safetensors():
    cfg = {"run_id": "r", "encoder": {"model_id": "./models/x", "use_safetensors": False}}
    with pytest.raises(ValueError, match="use_safetensors"):
        validate_embed_config(cfg)
