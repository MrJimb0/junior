"""A prompt's output example must not be returnable as a non-answer.

Every prompt ends by showing the model the JSON shape to produce. When that shape is
filled with nulls and empty lists, a small model reads it as the answer and returns it
verbatim. Measured on a real run against the 3B local model: 8 of 24 calls came back
byte-identical to their example, including a lines-of-therapy refine pass answering
{"lines": []} over the two lines the previous step had just found — and the run reported
that patient as having had no treatment.

An empty example is dangerous precisely because a copied one is invisible: it is
indistinguishable from a chart that genuinely says nothing. A filled example that gets
copied is obvious to anyone who looks at the output. So the examples carry illustrative
values, and this keeps them that way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from security_and_pipeline_tests.guardrails.recipe_files import finished_recipe_files

# A value that carries no information: copying it back answers nothing.
_SAYS_NOTHING = (None, "", [], {}, 0)


def _output_examples(text: str) -> list[dict]:
    """Every standalone JSON object literal in a prompt — the shape it shows the model."""
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj:
            found.append(obj)
    return found


def _prompt_files() -> list[Path]:
    """Every prompt of every recipe a run could use. Drafts are excluded the same way
    the other recipe guardrails exclude them."""
    return sorted(
        prompt
        for recipe in finished_recipe_files()
        for prompt in (recipe.parent / "prompts").glob("*.md")
    )


@pytest.mark.parametrize("prompt_path", _prompt_files(), ids=lambda p: p.name)
def test_the_output_example_shows_a_real_answer(prompt_path: Path):
    for example in _output_examples(prompt_path.read_text(encoding="utf-8")):
        informative = [
            key for key, value in example.items()
            # `false` is informative for a genuine yes/no field but says nothing as the
            # default of a whole example, so it only counts alongside other filled keys.
            if value not in _SAYS_NOTHING and value is not False
        ]
        assert informative, (
            f"this output example is entirely empty, so returning it verbatim is a "
            f"valid-looking non-answer that reads as 'the chart says nothing':\n"
            f"  {json.dumps(example)[:300]}"
        )
