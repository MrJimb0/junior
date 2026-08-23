"""A model writing the four characters n-u-l-l gets a real null, not a string.

Seen in the wild, on a living patient: pass 1 answered
`date_of_death: "null"`. That string is not null to anything downstream. The recipe's
stop_if saw a value and ended the recipe, six of its seven steps were skipped — including
the python finalize that would have normalised it — and only the output schema caught
it, by rejecting "null" as neither a date nor null.

Fixing it in the stop_if was the wrong layer: it left date_of_birth broken (its helper
uses setdefault, which will not replace an existing "null"), and for a `vital_status:
"dead"` answer it turned a loud schema failure into a schema-valid, confidently wrong
result. The string has to stop existing at the point the response is parsed, which every
prompt step shares.
"""
from __future__ import annotations

import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.retrieve_and_prompt_step import (
    _best_effort_json,
)


@pytest.mark.parametrize("written,expected", [
    ('{"date_of_death": "null"}', None),
    ('{"date_of_death": "NULL"}', None),
    ('{"date_of_death": " null "}', None),
    ('{"date_of_death": null}', None),
    ('{"date_of_death": "2025-03-27"}', "2025-03-27"),
])
def test_the_literal_string_becomes_a_real_null(written, expected):
    parsed, error = _best_effort_json(written)

    assert error is None
    assert parsed["date_of_death"] == expected


def test_none_is_left_alone():
    """"none" is a real clinical answer — "prior therapy: none" — and coercing it would
    erase findings. Only the word that means "this is JSON null and I typed it wrong"
    is corrected."""
    parsed, _ = _best_effort_json('{"prior_therapy": "none", "sites": "None"}')

    assert parsed["prior_therapy"] == "none"
    assert parsed["sites"] == "None"


def test_an_empty_string_is_left_alone():
    """A model declining to answer is different from answering that there is nothing,
    and the schemas already tell them apart."""
    parsed, _ = _best_effort_json('{"note": ""}')

    assert parsed["note"] == ""


def test_it_reaches_nested_objects_and_lists():
    """Recipes return nested shapes — clinical_tnm is an object, primaries a list — and
    a bad token can land anywhere in one."""
    parsed, _ = _best_effort_json(
        '{"clinical_tnm": {"cT": "T1c", "evidence_chunk_id": "null"}, '
        '"primaries": [{"site": "null"}, {"site": "breast"}]}'
    )

    assert parsed["clinical_tnm"]["evidence_chunk_id"] is None
    assert parsed["primaries"][0]["site"] is None
    assert parsed["primaries"][1]["site"] == "breast"


def test_it_applies_however_the_json_was_wrapped():
    """Small models fence their JSON, prefix it with chatter, or drop the closing fence.
    Every one of those paths returns through the same normalisation."""
    for wrapped in (
        '```json\n{"date_of_death": "null"}\n```',
        'Here is the JSON:\n{"date_of_death": "null"}',
        '{"date_of_death": "null"} trailing chatter',
    ):
        parsed, _ = _best_effort_json(wrapped)
        assert parsed is not None and parsed["date_of_death"] is None, wrapped


def test_no_prompt_skeleton_still_leads_a_nullable_field_with_the_quoted_word():
    """The prompts are what teach it. A skeleton showing `"field": "null unless ..."`
    is a worked example of the mistake, in the shape the model is copying."""
    import re
    from pathlib import Path

    recipes = Path(__file__).resolve().parents[2] / "var_extraction_recipes"
    offenders = [
        f"{path.relative_to(recipes)}:{n}"
        for path in recipes.rglob("prompts/*.md")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'":\s*"null\b', line, re.IGNORECASE)
    ]

    assert not offenders, (
        "these prompt skeletons show a nullable field as a quoted string beginning "
        f"with the word null: {offenders}"
    )
