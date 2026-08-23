"""The chunk window reserves the encoder's special-token budget.

The chunker counts content tokens (add_special_tokens=False), but embed_batch
adds [CLS]/[SEP] and truncates to max_tokens — so a chunk of exactly max_tokens
content tokens silently loses its tail at embed time. build_chunker_with_reserved
_specials defaults an omitted span to the content budget (max_tokens minus the
special tokens) and rejects an explicit span larger than that budget.
"""
from __future__ import annotations

import re

import pytest

from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import (
    NoChunker,
    TokenWindowChunker,
    build_chunker_with_reserved_specials,
)


class _FakeEncoder:
    """Minimal encoder for chunker tests: whitespace content tokens with offsets,
    configurable max_tokens / num_special_tokens. No model, no torch."""

    model_id = "fake"

    def __init__(self, max_tokens: int, num_special_tokens: int):
        self.max_tokens = max_tokens
        self.num_special_tokens = num_special_tokens

    def tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def test_default_window_reserves_special_tokens():
    enc = _FakeEncoder(max_tokens=10, num_special_tokens=2)
    chunker = build_chunker_with_reserved_specials(
        {"chunker": {"kind": "token_window", "overlap": 0}}, enc
    )
    assert isinstance(chunker, TokenWindowChunker)
    assert chunker.window == 8  # 10 - 2 reserved for [CLS]/[SEP]


def test_chunks_never_exceed_content_budget_so_tail_is_not_truncated():
    enc = _FakeEncoder(max_tokens=10, num_special_tokens=2)
    chunker = build_chunker_with_reserved_specials(
        {"chunker": {"kind": "token_window", "overlap": 0}}, enc
    )
    text = " ".join(f"t{i}" for i in range(20))  # 20 content tokens
    chunks = chunker.chunk(text, tokenizer=enc)

    assert chunks
    # no chunk exceeds the content budget, so embed_batch never truncates one
    assert all(c.token_count <= 8 for c in chunks)
    # the whole text is still covered -> the tail is not dropped at the chunker
    assert chunks[-1].char_end == len(text)


def test_explicit_window_equal_to_max_tokens_raises():
    enc = _FakeEncoder(max_tokens=10, num_special_tokens=2)
    with pytest.raises(ValueError, match="content budget"):
        build_chunker_with_reserved_specials(
            {"chunker": {"kind": "token_window", "window": 10, "overlap": 0}}, enc
        )


def test_explicit_window_at_content_budget_is_allowed():
    enc = _FakeEncoder(max_tokens=10, num_special_tokens=2)
    chunker = build_chunker_with_reserved_specials(
        {"chunker": {"kind": "token_window", "window": 8, "overlap": 0}}, enc
    )
    assert chunker.window == 8


def test_none_chunker_defaults_max_tokens_to_content_budget():
    enc = _FakeEncoder(max_tokens=10, num_special_tokens=2)
    chunker = build_chunker_with_reserved_specials({"chunker": {"kind": "none"}}, enc)
    assert isinstance(chunker, NoChunker)
    assert chunker.max_tokens == 8
