"""[PHI]: no recipe prompt renders the raw patient_id into the LLM call.

The corpus is already scoped to one patient and no prompt reasons about the id value, so a
raw patient_id in a prompt is just an identifier leaked into the LLM call (and PHI-side
receipt) for no gain. A future recipe that genuinely needs an identifier must use a
surrogate and update this guard deliberately.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMPTS = sorted(
    p for p in (REPO / "var_extraction_recipes").rglob("*.md") if p.parent.name == "prompts"
)
_PATIENT_ID = re.compile(r"{{\s*patient_id\s*}}")


def test_no_prompt_renders_patient_id() -> None:
    offenders = [
        str(p.relative_to(REPO)) for p in PROMPTS
        if _PATIENT_ID.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"prompts still rendering raw patient_id: {offenders}"


def test_there_are_prompts_to_scan() -> None:
    """Non-tautology guard: the glob actually finds the recipe prompt files."""
    assert len(PROMPTS) >= 15
