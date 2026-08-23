"""Which chunk an answer should cite: the one the deciding pass actually cited.

Every answer this pipeline produces must point back to the specific piece of source text it
rests on, so a reviewer can check it. We call that piece of source text a "chunk" (a short
span of a note or one row of a structured table), and each chunk has a stable id
(``evidence_chunk_id``). A "pass" here means one extraction step that was shown some chunks
and asked a question.

This used to fall back to the first chunk the pass read when the model cited nothing, so
that even a "we looked and found nothing" answer named the chart region searched. The
reasoning was that a citation to a region read is better than no citation. It is not. A
fallback pointer names a passage that was genuinely shown, so it satisfies the grounding
gate by construction: the value ships with ok true, an empty values_with_no_evidence, and a
real quote beside it in the answer table that does not support it. A reader who checks the
quote finds it says nothing; a reader who trusts the green columns never checks. Nulling
the pointer instead costs the recipe an honest failure and buys back the one property the
pipeline exists to provide.

A missing citation is now asked about rather than filled in: the prompt steps re-ask
through ``evidence_grounding.ask_until_grounded``, and whatever is still uncited afterwards
is reported by step 8 as an unsupported claim.

Loaded by recipe python helpers via load_shared_validation_rule.
"""
from __future__ import annotations

from typing import Any


def grounding_chunk(
    pass_data: Any, read_chunk_ids: Any, cited_key: str = "evidence_chunk_id"
) -> str | None:
    """The chunk id a pass cited, when it is one the pass was actually shown.

    ``cited_key`` is usually ``evidence_chunk_id``, but a pass may name its pointer
    differently (e.g. ``status_at_diagnosis_evidence_chunk_id``). ``read_chunk_ids`` is what
    the pass was shown, and is used only to check the citation — never to supply one.

    Returns ``None`` when the pass cited nothing, or cited something it was never shown.
    The caller keeps the value's pointer null, and step 8 reports the value as the
    unsupported claim it is."""
    read = read_chunk_ids if isinstance(read_chunk_ids, list) else []
    cited = pass_data.get(cited_key) if isinstance(pass_data, dict) else None
    return cited if isinstance(cited, str) and cited and cited in read else None
