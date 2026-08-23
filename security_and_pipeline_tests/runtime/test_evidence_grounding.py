"""The shared grounding helper: the chunk a pass CITED, or nothing.

grounding_chunk(pass_data, read_chunk_ids, cited_key="evidence_chunk_id") returns the
model's cited chunk id when it is one the pass was actually shown, and None otherwise. It
supports a custom cited_key for a pass that names its pointer differently.

It used to fall back to the top chunk the pass read whenever the model cited nothing, so
that even a "we looked and found nothing" answer named the region searched. That fallback
is what these tests now forbid. A borrowed pointer names a passage that WAS shown, so it
satisfies the grounding gate by construction: the value ships ok=true, with an empty
values_with_no_evidence, and a real quote beside it in the answer table that does not
support it. A reader who checks the quote finds it says nothing; a reader who trusts the
green columns never checks. A null pointer costs an honest failure and keeps the one
property this pipeline exists to provide.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "var_extraction_recipes" / "_shared_validation_rules" / "evidence_grounding.py"


def _load():
    spec = importlib.util.spec_from_file_location("evidence_grounding_test", HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_g = _load().grounding_chunk
READ = ["a:1:0", "b:2:0", "c:3:0"]


def test_model_citation_in_read_is_used():
    assert _g({"evidence_chunk_id": "b:2:0"}, READ) == "b:2:0"


def test_an_uncited_answer_gets_no_pointer():
    """Not the top chunk read. That chunk was shown, so it passes every automatic check
    while supporting nothing, which is worse than a blank a reviewer can see."""
    assert _g({"evidence_chunk_id": None}, READ) is None
    assert _g({}, READ) is None


def test_a_citation_the_pass_was_never_shown_gets_no_pointer():
    """Substituting the top chunk read would convert a caught fabrication into an
    uncaught one — the value would then look grounded to the gate."""
    assert _g({"evidence_chunk_id": "ghost:9:9"}, READ) is None


def test_empty_read_yields_none():
    assert _g({"evidence_chunk_id": "b:2:0"}, []) is None
    assert _g({}, []) is None


def test_non_list_read_is_treated_as_empty():
    assert _g({"evidence_chunk_id": "b:2:0"}, None) is None


def test_custom_cited_key():
    pd = {"status_at_diagnosis_evidence_chunk_id": "c:3:0", "evidence_chunk_id": "ghost:9:9"}
    assert _g(pd, READ, "status_at_diagnosis_evidence_chunk_id") == "c:3:0"
    # The default key here holds a chunk the pass was never shown, so: no pointer.
    assert _g(pd, READ) is None


def test_an_empty_string_citation_is_no_citation():
    assert _g({"evidence_chunk_id": ""}, READ) is None


def test_a_pass_that_answered_nothing_gets_no_pointer():
    assert _g(None, READ) is None
