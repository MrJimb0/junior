"""Behavioral contract tests for the network LLM providers.

These tests stub ``requests.post`` and assert the actual posted payload + URL
+ response parsing, and that the SHIPPED deployment allowlists load and wire the
GPT-5 quirks (max_completion_tokens / skip temperature / api-version).
"""
from __future__ import annotations

from pathlib import Path

import requests

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.gemini_compatible_provider import (
    GeminiCompatProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_allowlist import (
    AllowlistError,
    load_allowlist,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
    DenylistViolation,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_lookup import (
    build_provider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.local_huggingface_provider import (
    LocalHFProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.openai_compatible_provider import (
    OpenAICompatProvider,
)

REPO = Path(__file__).resolve().parents[2]
HOST = "https://llm-gateway.example.org/azure-openai/deployments/gpt-4o/chat/completions"


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _capture_post(monkeypatch, payload):
    """Patch requests.post to record the call and return a canned response."""
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, allow_redirects=None):
        seen.update(
            url=url, json=json, headers=headers, timeout=timeout,
            allow_redirects=allow_redirects,
        )
        return _FakeResp(payload)

    monkeypatch.setattr(requests, "post", fake_post)
    return seen


_OPENAI_RESP = {
    "choices": [{"message": {"content": '{"date_of_birth": "1950-01-01"}'}}],
    "model": "gpt-4o-2024", "system_fingerprint": "fp_abc",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _req(**kw):
    base = dict(endpoint_name="x", messages=[{"role": "user", "content": "hi"}],
                temperature=0.0, max_tokens=512, response_format="json_object", seed=7)
    base.update(kw)
    return LLMRequest(**base)


def test_openai_default_payload_and_parse(monkeypatch):
    seen = _capture_post(monkeypatch, _OPENAI_RESP)
    p = OpenAICompatProvider(endpoint_name="e", url=HOST, default_model="gpt-4o")
    resp = p._raw_chat(_req())
    body = seen["json"]
    assert body["model"] == "gpt-4o"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.0           # default path sends temperature
    assert body["max_tokens"] == 512            # ...and max_tokens (not max_completion_tokens)
    assert "max_completion_tokens" not in body
    assert body["response_format"] == {"type": "json_object"}
    assert body["seed"] == 7
    assert seen["url"] == HOST                   # no api-version appended by default
    assert resp.content == '{"date_of_birth": "1950-01-01"}'
    assert resp.usage["total_tokens"] == 15
    assert "gpt-4o-2024" in resp.resolved_model.fingerprint


def test_openai_gpt5_extras_are_sent(monkeypatch):
    seen = _capture_post(monkeypatch, _OPENAI_RESP)
    p = OpenAICompatProvider(endpoint_name="gpt5", url=HOST, default_model="gpt-5",
                             api_version="2024-12-01-preview",
                             use_max_completion_tokens=True, skip_temperature=True)
    p._raw_chat(_req())
    body = seen["json"]
    assert body["max_completion_tokens"] == 512  # GPT-5 token param
    assert "max_tokens" not in body
    assert "temperature" not in body             # GPT-5 rejects temperature
    assert seen["url"].endswith("?api-version=2024-12-01-preview")


def test_gemini_payload_and_parse(monkeypatch):
    resp_payload = {
        "candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}],
        "modelVersion": "gemini-2.5",
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }
    seen = _capture_post(monkeypatch, resp_payload)
    p = GeminiCompatProvider(endpoint_name="g", url=HOST, default_model="gemini-2.5")
    resp = p._raw_chat(_req(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]))
    body = seen["json"]
    assert body["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert body["generationConfig"]["maxOutputTokens"] == 512
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert resp.content == '{"x": 1}'
    assert resp.usage["total_tokens"] == 5


def _a_hosted_gpt5_allowlist(tmp_path):
    """A BAA-attested GPT-5-style gateway entry, the shape a hosted deployment declares."""
    p = tmp_path / "allowlist.yaml"
    p.write_text(
        "allowed_endpoints:\n"
        "  - name: hosted_gpt5\n"
        "    url: https://llm-gateway.example.org/azure-openai/deployments/gpt-5/chat/completions\n"
        '    api_version: "2024-12-01-preview"\n'
        "    provider: openai_compat\n"
        "    attestation: baa\n"
        '    auth: "apim:$APIM_SUBSCRIPTION_KEY"\n'
        "    default_model: gpt-5\n"
        "    use_max_completion_tokens: true\n"
        "    skip_temperature: true\n",
        encoding="utf-8",
    )
    return p


def test_shipped_allowlists_load_and_wire_gpt5(tmp_path):
    # both bundled local allowlists must load.
    load_allowlist(REPO / "deployment/local/llm_allowlist.yaml")
    load_allowlist(REPO / "deployment/local/llm_allowlist_local3b.yaml")
    al = load_allowlist(_a_hosted_gpt5_allowlist(tmp_path))
    gpt5 = build_provider(al.get("hosted_gpt5"))
    assert isinstance(gpt5, OpenAICompatProvider)
    assert gpt5.use_max_completion_tokens is True   # extras wired through the factory
    assert gpt5.skip_temperature is True
    assert gpt5.api_version == "2024-12-01-preview"


def test_shipped_3b_allowlist_uses_half_precision():
    al = load_allowlist(REPO / "deployment/local/llm_allowlist_local3b.yaml")
    local = build_provider(al.get("local_qwen"))

    assert isinstance(local, LocalHFProvider)
    assert local.dtype == "float16"
    # A ceiling this machine can execute, not the 32768 the model advertises. Attention
    # memory grows with the square of the prompt: measured on 48 GiB, 7,909 tokens
    # peaked at 17 GiB and 15,816 at 62.9 GiB, past the machine. The previous 12000 sat
    # on the vertical part of that curve, which is what SIGKILLed a nine-patient run.
    assert local.max_prompt_tokens_cap == 8000


def test_every_local_endpoint_has_a_hardware_ceiling():
    """Unbounded is the dangerous state, and it does not fail — it takes the machine
    down. Metal refuses the allocation and the process dies with no Python error, so
    nothing catches it and nothing reports it."""
    # Every shipped allowlist, not just the laptop's. An endpoint that loads a model
    # into Junior's own process needs a ceiling measured for the hardware it runs on —
    # an A100 wants a far larger one than a laptop, and neither wants none. Endpoints
    # that talk to a server over HTTP (a frontier API, or Ollama on localhost) are
    # skipped: they allocate in someone else's process and answer with an error rather
    # than with a dead process.
    for allowlist in sorted(REPO.glob("deployment/*/llm_allowlist*.yaml")):
        al = load_allowlist(allowlist)
        for endpoint in al.endpoints():
            provider = build_provider(endpoint)
            if not isinstance(provider, LocalHFProvider):
                continue
            assert provider.max_prompt_tokens_cap, (
                f"{allowlist.name}: endpoint {endpoint.name!r} loads a model in-process "
                "with no "
                "max_prompt_tokens_cap, so an oversized prompt kills the process "
                "instead of being refused"
            )


def test_canonical_provider_kind_accepted_long_form_rejected(tmp_path):
    import pytest

    yml = tmp_path / "al.yaml"
    yml.write_text(
        "allowed_endpoints:\n"
        "  - name: ep\n"
        "    url: https://llm-gateway.example.org/x\n"
        "    provider: openai_compat\n"     # the canonical short kind
        "    attestation: baa\n",
        encoding="utf-8",
    )
    al = load_allowlist(yml)
    assert al.get("ep").provider == "openai_compat"
    assert isinstance(build_provider(al.get("ep")), OpenAICompatProvider)

    # the long "*_compatible" spelling is not a blessed kind; only the short form loads
    long_form = yml.with_name("long.yaml")
    long_form.write_text(
        "allowed_endpoints:\n"
        "  - name: ep\n"
        "    url: https://llm-gateway.example.org/x\n"
        "    provider: openai_compatible\n"
        "    attestation: baa\n",
        encoding="utf-8",
    )
    with pytest.raises(AllowlistError, match="unknown provider kind"):
        load_allowlist(long_form)


def test_unknown_extras_key_rejected_not_silently_ignored(tmp_path):
    import pytest

    yml = tmp_path / "al.yaml"
    yml.write_text(
        "allowed_endpoints:\n"
        "  - name: ep\n"
        "    url: https://llm-gateway.example.org/x\n"
        "    provider: openai_compat\n"
        "    attestation: baa\n"
        "    default_timeout_second: 300\n",  # typo: should be default_timeout_seconds
        encoding="utf-8",
    )
    with pytest.raises(AllowlistError, match="unknown extras key"):
        load_allowlist(yml)


def test_gemini_factory_honors_per_endpoint_timeout(tmp_path):
    yml = tmp_path / "al.yaml"
    yml.write_text(
        "allowed_endpoints:\n"
        "  - name: g\n"
        "    url: https://llm-gateway.example.org/gcp-vertex-ai/x\n"
        "    provider: gemini_compat\n"
        "    attestation: baa\n"
        "    default_timeout_seconds: 300\n",
        encoding="utf-8",
    )
    provider = build_provider(load_allowlist(yml).get("g"))
    assert isinstance(provider, GeminiCompatProvider)
    assert provider.timeout_s == 300.0  # per-endpoint override, not the 60s default


def test_denylist_still_blocks_public_host(tmp_path):
    yml = tmp_path / "al.yaml"
    yml.write_text(
        "allowed_endpoints:\n"
        "  - name: bad\n"
        "    url: https://api.openai.com/v1/chat/completions\n"
        "    provider: openai_compat\n"
        "    attestation: baa\n",
        encoding="utf-8",
    )
    import pytest
    with pytest.raises((DenylistViolation, AllowlistError)):
        load_allowlist(yml)


def test_a_local_model_says_it_runs_here_and_on_what(capsys):
    """Choosing a local model is choosing to spend this machine's memory. A remote
    endpoint answers an oversized prompt with an error; a local one dies with nothing in
    the log, so the ceiling that prevents that — and the hardware it was measured on —
    belongs on the screen before the run, not in a YAML comment."""
    from apps_and_interfaces.command_line_interface import (
        _describe_the_model_that_reads_the_chart,
    )

    _describe_the_model_that_reads_the_chart(
        {"allowlist_path": str(REPO / "deployment/local/llm_allowlist_local3b.yaml")}
    )
    shown = capsys.readouterr().out

    assert "on this machine" in shown
    assert "8000 prompt tokens" in shown, "the ceiling is not shown"
    assert "llm_allowlist_local3b.yaml" in shown, "no pointer to what to edit"


def test_the_shipped_configs_do_not_describe_anyone_s_hardware():
    """A memory figure measured on one laptop, shipped in the repo and printed as fact
    to whoever cloned it, is worse than no figure: it is wrong for every machine but
    one, and it reads as authoritative."""
    import yaml as _yaml

    for allowlist in sorted(REPO.glob("deployment/*/llm_allowlist*.yaml")):
        loaded = _yaml.safe_load(allowlist.read_text()) or {}
        for endpoint in loaded.get("allowed_endpoints") or []:
            assert "measured_on" not in endpoint, (
                f"{allowlist.name}: endpoint {endpoint.get('name')!r} ships a hardware "
                "measurement. That field is for an operator's own note about their own "
                "machine; the CLI reads installed RAM at runtime instead."
            )


def test_the_machine_size_is_read_not_configured():
    from apps_and_interfaces.command_line_interface import (
        _how_much_memory_this_machine_has,
    )

    detected = _how_much_memory_this_machine_has()
    assert detected == "" or detected.endswith(" GiB"), detected


def test_an_endpoint_with_no_ceiling_says_so_loudly(tmp_path, capsys):
    from apps_and_interfaces.command_line_interface import (
        _describe_the_model_that_reads_the_chart,
    )

    model = tmp_path / "a_model"
    model.mkdir()
    allowlist = tmp_path / "al.yaml"
    allowlist.write_text(
        "allowed_endpoints:\n"
        "  - name: uncapped\n"
        f"    url: {model}\n"
        "    provider: local_hf\n"
        "    attestation: self_hosted\n"
        f"    default_model: {model}\n",
        encoding="utf-8",
    )

    _describe_the_model_that_reads_the_chart({"allowlist_path": str(allowlist)})
    shown = capsys.readouterr().out

    assert "no ceiling set" in shown
    assert "kills the run" in shown


def test_a_remote_endpoint_is_not_described_as_local(capsys, tmp_path):
    """A hosted endpoint sends charts to an attested service. Telling an operator
    it runs on their machine would be false, and would hide the disclosure that
    matters about it."""
    from apps_and_interfaces.command_line_interface import (
        _describe_the_model_that_reads_the_chart,
    )

    _describe_the_model_that_reads_the_chart(
        {"allowlist_path": str(_a_hosted_gpt5_allowlist(tmp_path)),
         "model_override": "hosted_gpt5"}
    )
    shown = capsys.readouterr().out

    assert "on this machine" not in shown
    assert "prompt tokens" not in shown


def test_neither_url_provider_follows_a_redirect(monkeypatch):
    """The allowlist and the denylist are checked against the CONFIGURED url, and
    nothing re-checks a hop. So a 307 from an approved gateway would resend the body —
    the chart passages — to whatever host the response names, and the run would report
    clean. requests drops the Authorization header across hosts, which protects the
    credential and not the patient. Both providers must therefore refuse to follow."""
    seen = _capture_post(monkeypatch, _OPENAI_RESP)
    OpenAICompatProvider(endpoint_name="e", url=HOST, default_model="gpt-4o")._raw_chat(_req())
    assert seen["allow_redirects"] is False, "openai_compat followed a redirect"

    gemini_resp = {
        "candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}],
        "modelVersion": "gemini-2.5",
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5},
    }
    seen = _capture_post(monkeypatch, gemini_resp)
    GeminiCompatProvider(endpoint_name="g", url=HOST, default_model="gemini-2.5")._raw_chat(_req())
    assert seen["allow_redirects"] is False, "gemini_compat followed a redirect"
