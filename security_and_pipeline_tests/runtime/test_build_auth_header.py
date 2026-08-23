"""build_auth_header fails loudly when an env-referenced key is unset.

An allowlist auth spec like ``apim:$APIM_SUBSCRIPTION_KEY`` reads the key from the named
environment variable. An unset/empty variable must raise here — otherwise the request is
sent with an empty Ocp-Apim-Subscription-Key and 401s at call time with no hint.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    build_auth_header,
)


def test_unset_apim_env_var_raises_descriptive_error(monkeypatch):
    monkeypatch.delenv("APIM_SUBSCRIPTION_KEY", raising=False)
    with pytest.raises(ValueError, match=r"APIM_SUBSCRIPTION_KEY.*onc-gpt|onc-gpt.*APIM_SUBSCRIPTION_KEY"):
        build_auth_header("apim:$APIM_SUBSCRIPTION_KEY", "onc-gpt")


def test_set_apim_env_var_builds_header(monkeypatch):
    monkeypatch.setenv("APIM_SUBSCRIPTION_KEY", "secret123")
    assert build_auth_header("apim:$APIM_SUBSCRIPTION_KEY") == {
        "Ocp-Apim-Subscription-Key": "secret123"
    }


def test_literal_value_still_builds_header():
    assert build_auth_header("apim:literalkey") == {"Ocp-Apim-Subscription-Key": "literalkey"}


def test_empty_auth_yields_no_header():
    assert build_auth_header(None) == {}
    assert build_auth_header("") == {}
