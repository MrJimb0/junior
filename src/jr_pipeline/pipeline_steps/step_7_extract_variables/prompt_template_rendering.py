"""prompt loader and renderer for the extract step.

The extract step asks a language model to read a patient's evidence and return
a structured answer; this module turns a prompt file on disk into the actual
system + user text the model sees. Prompt files are markdown with optional yaml
front-matter, split into `# SYSTEM` / `# USER` sections. Jinja2 renders with
StrictUndefined so a missing context key fails loudly rather than silently
producing an empty string. The raw file's content hash is stored as
`template_hash` so the step's audit record can pin the exact prompt used.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, Template

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
)


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    path: Path
    system: str
    user: str
    front_matter: dict[str, Any]
    template_hash: str


_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^# +(SYSTEM|USER)\s*$", re.MULTILINE)


def load_prompt(path: Path) -> PromptTemplate:
    """read a prompt file, split it into system/user sections, and compute a
    content hash of the raw bytes so the exact prompt can be pinned later."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    fm: dict[str, Any] = {}
    body = raw
    m = _FRONT_RE.match(raw)
    if m:
        fm = yaml.safe_load(m.group(1)) or {}
        body = raw[m.end():]

    # no section headers? treat the whole body as the user prompt.
    sections: dict[str, list[str]] = {"SYSTEM": [], "USER": []}
    current: str | None = None
    for line in body.splitlines(keepends=True):
        hdr = _SECTION_RE.match(line.strip())
        if hdr:
            current = hdr.group(1).upper()
            continue
        if current is None:
            sections["USER"].append(line)
        else:
            sections[current].append(line)
    system = "".join(sections["SYSTEM"]).strip()
    user = "".join(sections["USER"]).strip()

    return PromptTemplate(
        name=fm.get("name") or path.stem,
        path=path,
        system=system,
        user=user,
        front_matter=fm,
        template_hash=hash_file(path),
    )


# autoescape (the templating library's html-escaping safety feature) is off
# because prompts go to a language model, not to a web page.
_env = Environment(undefined=StrictUndefined, autoescape=False)


def render(template: PromptTemplate, context: Mapping[str, Any]) -> tuple[str, str]:
    """render (system, user) against the step runner's template context."""
    sys_tmpl: Template = _env.from_string(template.system)
    usr_tmpl: Template = _env.from_string(template.user)
    return sys_tmpl.render(**context), usr_tmpl.render(**context)


def render_inline(text: str, context: Mapping[str, Any]) -> str:
    """Render one short template string (e.g. a recipe reranking filter value).

    Uses the same Jinja settings as the prompt renderer, so a recipe writes a
    filter threshold the same way it writes a prompt reference, e.g.
    ``{{ vars.date_of_diagnosis.data.date_of_diagnosis }}``. A string with no
    ``{{ ... }}`` is returned unchanged. Like ``render``, an undefined variable
    raises (StrictUndefined); the caller decides what to do with that.
    """
    return _env.from_string(text).render(**context)
