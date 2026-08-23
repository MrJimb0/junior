"""The encoder never pulls weights from the network.

On a PHI box the model must load from local files only; a bare ``from_pretrained``
silently falls back to a HuggingFace Hub download (network egress from a PHI host).
HFEncoder must pass ``local_files_only=True`` (the default) to all three
``from_pretrained`` calls — tokenizer, config, model — and honour an explicit
override for non-PHI dev auto-download.
"""
from __future__ import annotations

import types

import pytest

# Both tests monkeypatch transformers by name, so the library has to be importable. It
# ships in the `torch` extra rather than the core install.
pytest.importorskip("transformers", reason="needs the torch extra: pip install -e '.[torch]'")

from jr_pipeline.pipeline_steps.step_2_embed_chunks.encoder import HFEncoder  # noqa: E402


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []


def _install_mocks(monkeypatch, recorder, hidden_size=8):
    def tok_from_pretrained(model_id, **kw):
        recorder.calls.append(("tokenizer", kw))
        return types.SimpleNamespace(pad_token_id=0, eos_token=None, cls_token=None)

    def cfg_from_pretrained(model_id, **kw):
        recorder.calls.append(("config", kw))
        return types.SimpleNamespace(reference_compile=True, hidden_size=hidden_size)

    def model_from_pretrained(model_id, config=None, **kw):
        recorder.calls.append(("model", kw))
        model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=hidden_size))
        model.to = lambda device: model
        model.eval = lambda: model
        return model

    monkeypatch.setattr("transformers.AutoTokenizer",
                        types.SimpleNamespace(from_pretrained=tok_from_pretrained))
    monkeypatch.setattr("transformers.AutoConfig",
                        types.SimpleNamespace(from_pretrained=cfg_from_pretrained))
    monkeypatch.setattr("transformers.AutoModel",
                        types.SimpleNamespace(from_pretrained=model_from_pretrained))


def test_all_three_from_pretrained_use_local_files_only_by_default(monkeypatch):
    recorder = _Recorder()
    _install_mocks(monkeypatch, recorder)
    HFEncoder(model_id="some/hub-id", device_preference="cpu")._ensure_loaded()
    assert [which for which, _ in recorder.calls] == ["tokenizer", "config", "model"]
    for which, kwargs in recorder.calls:
        assert kwargs.get("local_files_only") is True, f"{which} missing local_files_only=True"


def test_local_files_only_can_be_disabled_for_dev(monkeypatch):
    recorder = _Recorder()
    _install_mocks(monkeypatch, recorder)
    HFEncoder(model_id="some/hub-id", device_preference="cpu", local_files_only=False)._ensure_loaded()
    assert all(kwargs.get("local_files_only") is False for _, kwargs in recorder.calls)
