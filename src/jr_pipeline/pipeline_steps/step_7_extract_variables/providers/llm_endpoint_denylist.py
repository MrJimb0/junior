"""The fixed blocklist of public, consumer language-model API hosts (ADR 0006).

A denylist is a list of network hosts that are forbidden, no matter what. This
file names the public commercial chatbot/LLM (large language model) services —
OpenAI, Anthropic, public Google, etc. — that this pipeline must never send
data to, because doing so would push PHI (protected health information) out to
shared infrastructure we do not control.

The block is enforced when a model connector ("provider") is constructed, so no
caller can sneak past it by building a provider directly. Host matching ignores
upper/lower case and also catches subdomains via a leading-dot suffix, so
"foo.api.openai.com" is blocked just like "api.openai.com". Removing a host from
this list requires a security review. (ADR 0006 is the architecture decision
record that documents this rule.)
"""
from __future__ import annotations

from urllib.parse import urlparse

FORBIDDEN_HOSTS: frozenset[str] = frozenset({
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.cohere.com",
    "api.mistral.ai",
    "api.together.xyz",
    "api.together.ai",
    "openrouter.ai",
    "api.groq.com",
    "api.perplexity.ai",
    "api.deepseek.com",
    "api.fireworks.ai",
})


class DenylistViolation(RuntimeError):
    """raised when a provider tries to use a forbidden host."""


def _host_of(url: str) -> str:
    # The trailing dot is stripped because a fully qualified name ending in one resolves
    # to exactly the same host: "api.openai.com." and "api.openai.com" are one address,
    # and DNS-suffix-strict environments write the first form. Without this the blocklist
    # compares "api.openai.com." against a set holding "api.openai.com", matches neither
    # the exact entry nor the ".<entry>" subdomain rule, and lets the endpoint through.
    return (urlparse(url).hostname or "").lower().rstrip(".")


def host_is_forbidden(host: str) -> bool:
    """True if the host is on the blocklist, either exactly or as a subdomain."""
    host = host.lower()
    if host in FORBIDDEN_HOSTS:
        return True
    return any(host.endswith("." + f) for f in FORBIDDEN_HOSTS)


def check_url(url: str) -> None:
    """Refuse (raise DenylistViolation) if the URL's host is on the blocklist."""
    host = _host_of(url)
    if not host:
        raise DenylistViolation(
            f"Cannot determine host from endpoint URL: {url!r}"
        )
    if host_is_forbidden(host):
        raise DenylistViolation(
            f"Endpoint host {host!r} is on the hardcoded public-API denylist. "
            "Public consumer LLM APIs are refused by architecture. "
            "See docs/adr/0006-llm-endpoint-denylist-allowlist.md."
        )
