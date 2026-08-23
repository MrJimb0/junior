"""The local extraction LLM is OFFLINE by default —
both transformers ``from_pretrained`` calls get ``local_files_only=True`` so a PHI run
never silently pulls weights from the Hub; a missing local model dir fails LOUD instead
of being reinterpreted as a Hub repo id; the exact weight SHA is recorded for provenance;
and an endpoint can still explicitly opt into a download (dev/demo only).

Torch-free: transformers + torch are faked so no weights load."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (
    LocalHFProvider,
    _resolve_local_weight_sha,
)


def _fake_hf(monkeypatch, captured):
    class FakeTok:
        pad_token_id = 0
        eos_token_id = 0
        pad_token = "<eos>"
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, model_id, **kw):
            captured["tok"] = kw
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            captured["mdl"] = kw
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeTok
    transformers.AutoModelForCausalLM = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", MagicMock())
    monkeypatch.setattr(
        "jr_pipeline.runtime_infrastructure.torch_device_selection.select_device",
        lambda *a, **k: ("cpu", None),
    )


def test_offline_by_default(monkeypatch):
    captured: dict = {}
    _fake_hf(monkeypatch, captured)
    provider = LocalHFProvider(endpoint_name="e", model_id="Qwen/Qwen2.5-0.5B-Instruct")
    provider._ensure_loaded()
    assert captured["tok"]["local_files_only"] is True
    assert captured["mdl"]["local_files_only"] is True


def test_allow_hub_download_opt_in(monkeypatch):
    captured: dict = {}
    _fake_hf(monkeypatch, captured)
    provider = LocalHFProvider(
        endpoint_name="e", model_id="Qwen/Qwen2.5-0.5B-Instruct", allow_hub_download=True
    )
    provider._ensure_loaded()
    assert captured["tok"]["local_files_only"] is False
    assert captured["mdl"]["local_files_only"] is False


def test_missing_local_dir_fails_loud():
    # a path-shaped model_id that does not exist must raise BEFORE any load/download.
    provider = LocalHFProvider(endpoint_name="e", model_id="./definitely_not_a_real_model_dir")
    with pytest.raises(RuntimeError, match="not found"):
        provider._ensure_loaded()


def test_weight_sha_recorded_for_local_dir(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"fake weights")
    sha = _resolve_local_weight_sha(str(tmp_path))
    assert sha is not None and sha.startswith("sha256:")
    # a Hub id is not a local dir -> no weight file to hash
    assert _resolve_local_weight_sha("Qwen/Qwen2.5-0.5B-Instruct") is None
