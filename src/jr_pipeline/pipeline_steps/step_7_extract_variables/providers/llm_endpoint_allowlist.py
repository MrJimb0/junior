"""institutional endpoint allowlist loader.

The allowlist is the vetted list of LLM endpoints this pipeline is permitted
to call. The file lives outside this repo (box/nero, institution-private).
Each row carries a name + url + attestation (the legal basis for sending PHI:
``baa`` = a Business Associate Agreement is in place, or ``self_hosted`` =
runs on our own hardware) + provider kind + authentication mechanism. Every
url is checked against the hardcoded denylist (forbidden hosts) before the
entry becomes usable, so a typo in the allowlist can't quietly route PHI
(protected health information) to an unapproved host.

Recipes reference endpoints by name, never url. The saved-code record stores
names + attestations only (see runtime_enforcing_safety_and_reproducibility/
reproducibility/frozen_code_snapshot.py), never urls or credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
    check_url,
)

_VALID_ATTESTATIONS = {"baa", "self_hosted"}
_VALID_KINDS = {"openai_compat", "gemini_compat", "local_hf"}

# The top-level fields every entry may carry. Anything else is a provider-specific
# "extras" key and must be one the provider kind's factory actually reads (below) —
# otherwise a typo like `default_timeout_second` sweeps into extras and silently leaves
# the provider on its hardcoded defaults.
_RESERVED_KEYS = frozenset({"name", "url", "provider", "attestation", "auth", "default_model"})


def _anchor_local_model_dir(model_id: str) -> str:
    """Resolve a relative local-model directory against the checkout.

    A HuggingFace hub id ("Qwen/Qwen2.5-3B") is a name, not a path, and must be left
    alone — it is distinguished by not existing on disk relative to the checkout."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (  # noqa: E501
        resolve_repo_root,
    )

    path = Path(model_id).expanduser()
    if path.is_absolute():
        return str(path)
    anchored = (resolve_repo_root() / path).resolve()
    return str(anchored) if anchored.is_dir() else model_id
_KNOWN_EXTRAS_BY_KIND: dict[str, frozenset[str]] = {
    "openai_compat": frozenset(
        {"api_version", "use_max_completion_tokens", "skip_temperature", "default_timeout_seconds"}
    ),
    "gemini_compat": frozenset({"default_timeout_seconds"}),
    "local_hf": frozenset(
        {
            "device",
            "dtype",
            "max_new_tokens_cap",
            "max_prompt_tokens_cap",
            # What hardware the ceiling above was measured on, in the operator's words.
            # A cap is a number about a machine, and a number with no provenance is one
            # nobody can safely raise: whoever moves this model to an A100 needs to know
            # that 8000 was a laptop's answer, not the model's. Shown before extraction
            # runs, so it is read where it matters rather than living in a YAML comment.
            "measured_on",
            "allow_download",
        }
    ),
}


@dataclass(frozen=True)
class AllowedEndpoint:
    """one row of the institutional allowlist."""

    name: str
    url: str
    provider: str
    attestation: str
    auth: str | None = None
    default_model: str | None = None
    extras: dict[str, Any] | None = None


class AllowlistError(RuntimeError):
    """raised on invalid allowlist files or unresolved endpoint names."""


class Allowlist:
    """loaded endpoint allowlist keyed by name."""

    def __init__(self, entries: list[AllowedEndpoint]):
        self._by_name: dict[str, AllowedEndpoint] = {}
        for e in entries:
            if e.name in self._by_name:
                raise AllowlistError(f"Duplicate endpoint name: {e.name!r}")
            self._by_name[e.name] = e

    def endpoints(self) -> list[AllowedEndpoint]:
        """Every endpoint on the list, in the order the file names them.

        Resolving one by name is what a run does. Showing an operator which model is
        about to read their charts needs all of them, and that is a question about the
        list rather than about a name they already know."""
        return list(self._by_name.values())

    def get(self, name: str) -> AllowedEndpoint:
        if name not in self._by_name:
            raise AllowlistError(
                f"Endpoint {name!r} not in allowlist. "
                f"Available: {sorted(self._by_name)}"
            )
        return self._by_name[name]


def _validate_entry(raw: dict[str, Any]) -> AllowedEndpoint:
    for required in ("name", "url", "provider", "attestation"):
        if required not in raw:
            raise AllowlistError(f"allowlist entry missing {required!r}: {raw}")
    name = str(raw["name"])
    url = str(raw["url"])
    provider = str(raw["provider"])
    attestation = str(raw["attestation"])

    if provider not in _VALID_KINDS:
        raise AllowlistError(
            f"{name}: unknown provider kind {provider!r}; supported: {sorted(_VALID_KINDS)}"
        )
    if attestation not in _VALID_ATTESTATIONS:
        raise AllowlistError(
            f"{name}: unknown attestation {attestation!r}; supported: {sorted(_VALID_ATTESTATIONS)}"
        )

    extras = {k: v for k, v in raw.items() if k not in _RESERVED_KEYS}
    known_extras = _KNOWN_EXTRAS_BY_KIND.get(provider, frozenset())
    unknown_extras = sorted(set(extras) - known_extras)
    if unknown_extras:
        raise AllowlistError(
            f"{name}: unknown extras key(s) {unknown_extras} for provider {provider!r}; "
            f"valid extras: {sorted(known_extras)}"
        )

    # the local provider loads a HuggingFace model in-process, so its `url` field
    # is really a model id (a placeholder, not a network address) — every other
    # provider, which does reach the network, must clear the denylist first.
    if provider != "local_hf":
        check_url(url)
    else:
        # A relative model directory means "in the checkout", which is how the shipped
        # allowlists are written. Left bare it resolves against whatever directory the
        # command was run from, and a model that is not there fails as an empty answer
        # rather than an error — so anchor it while the mistake is still visible.
        url = _anchor_local_model_dir(url)
        default_model = raw.get("default_model")
        if default_model:
            raw = {**raw, "default_model": _anchor_local_model_dir(str(default_model))}

    return AllowedEndpoint(
        name=name,
        url=url,
        provider=provider,
        attestation=attestation,
        auth=raw.get("auth"),
        default_model=raw.get("default_model"),
        extras=extras or None,
    )


def load_allowlist(path: Path | str | None) -> Allowlist:
    """load and validate an allowlist yaml file; path=None yields an empty allowlist,
    used for fully offline runs (e.g. the local in-process model)."""
    if path is None:
        return Allowlist(entries=[])
    p = Path(path)
    if not p.is_file():
        raise AllowlistError(f"Allowlist file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    items = raw.get("allowed_endpoints") or []
    if not isinstance(items, list):
        raise AllowlistError("'allowed_endpoints' must be a list")
    entries = [_validate_entry(x) for x in items]
    return Allowlist(entries=entries)
