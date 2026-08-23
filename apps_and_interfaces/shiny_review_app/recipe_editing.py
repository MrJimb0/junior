"""Author and edit variable recipes — the job the CLI deliberately does not do.

`junior new-variable` was built and then reverted (81ac5f2) with the ruling that
authoring belongs in the app, not the CLI. This module is that ruling honoured: the
scaffold logic returns with the same guarantees it had there — every
``<variable>_<version>_`` cross-reference rewritten, every file renamed, and the copy
marked ``needs_editing: true`` so the loader refuses to run it until its author says
the content is theirs — and gains the half the CLI never had: editing the underlying
files with the pipeline's own loader as the gate on every save.

Nothing here writes outside the recipe tree, and no save lands unvalidated: an edit
is swapped in atomically, ``load_recipe`` is asked whether the recipe still loads,
and a failing edit is rolled back with the loader's message returned to the editor.
The one loader error tolerated on save is the draft refusal itself — a draft must
stay editable while it is a draft; ``finish_draft`` is where full validation has the
last word.
"""
from __future__ import annotations

import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RECIPES_ROOT = REPO / "var_extraction_recipes"

# The loader's draft refusal names the scaffold; matching on this phrase is how a
# draft-tolerant validation tells "unfinished on purpose" from "broken".
_DRAFT_REFUSAL_PHRASE = "scaffolded from another variable"

# What a variable may be called. It becomes a folder name, a file prefix, and a key
# in every result the variable ever produces.
_VARIABLE_NAME = re.compile(r"[a-z][a-z0-9_]*")

_EDITABLE_SUFFIXES = {".yaml", ".md", ".json", ".py"}


@dataclass
class RecipeVersion:
    variable: str
    version: str
    collection: str
    version_dir: Path
    is_draft: bool

    @property
    def label(self) -> str:
        shown = f"{self.collection}/{self.variable} ({self.version})"
        return f"{shown} · draft" if self.is_draft else shown


@dataclass
class RecipeFile:
    rel_path: str      # relative to the version dir
    kind: str          # "recipe" / "prompt" / "schema" / "helper" / "other"


def _recipe_yaml_in(version_dir: Path) -> Path | None:
    prefix = f"{version_dir.parent.name}_{version_dir.name}"
    candidate = version_dir / f"{prefix}_recipe.yaml"
    return candidate if candidate.is_file() else None


def is_draft(version_dir: Path) -> bool:
    """True while a scaffolded recipe still carries ``needs_editing: true``."""
    recipe_yaml = _recipe_yaml_in(Path(version_dir))
    if recipe_yaml is None:
        return False
    try:
        import yaml

        raw = yaml.safe_load(recipe_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        return False  # unreadable is broken, not drafted; the editor shows the error
    return bool(isinstance(raw, dict) and raw.get("needs_editing"))


def list_recipe_versions(recipes_root: Path | None = None) -> list[RecipeVersion]:
    """Every recipe version on disk, drafts included and labeled."""
    root = Path(recipes_root or RECIPES_ROOT)
    if not root.is_dir():
        return []
    found = []
    for recipe_yaml in sorted(root.rglob("*_recipe.yaml")):
        version_dir = recipe_yaml.parent
        variable_dir = version_dir.parent
        if recipe_yaml.name != f"{variable_dir.name}_{version_dir.name}_recipe.yaml":
            continue
        collection = variable_dir.parent.relative_to(root).as_posix()
        found.append(RecipeVersion(
            variable=variable_dir.name,
            version=version_dir.name,
            collection=collection,
            version_dir=version_dir,
            is_draft=is_draft(version_dir),
        ))
    return found


def recipe_files(version_dir: Path) -> list[RecipeFile]:
    """The version's editable files, the recipe yaml first."""
    version_dir = Path(version_dir)
    prefix = f"{version_dir.parent.name}_{version_dir.name}"

    def kind_of(path: Path) -> str:
        if path.name == f"{prefix}_recipe.yaml":
            return "recipe"
        if path.suffix == ".md":
            return "prompt"
        if path.suffix == ".json":
            return "schema"
        if path.suffix == ".py":
            return "helper"
        return "other"

    ranked = {"recipe": 0, "schema": 1, "prompt": 2, "helper": 3, "other": 4}
    files = [
        RecipeFile(rel_path=p.relative_to(version_dir).as_posix(), kind=kind_of(p))
        for p in sorted(version_dir.rglob("*"))
        if p.is_file() and p.suffix in _EDITABLE_SUFFIXES and "__pycache__" not in p.parts
    ]
    return sorted(files, key=lambda f: (ranked[f.kind], f.rel_path))


def _safe_target(version_dir: Path, rel_path: str) -> Path:
    """The file the editor may touch — inside the version dir, editable suffix, no
    traversal. Every read and write goes through here."""
    version_dir = Path(version_dir).resolve()
    target = (version_dir / rel_path).resolve()
    if version_dir not in target.parents and target != version_dir:
        raise ValueError(f"{rel_path!r} is not inside this recipe version")
    if target.suffix not in _EDITABLE_SUFFIXES:
        raise ValueError(f"{rel_path!r} is not an editable recipe file")
    return target


def read_file(version_dir: Path, rel_path: str) -> str:
    return _safe_target(version_dir, rel_path).read_text(encoding="utf-8")


def _load_error(version_dir: Path, *, tolerate_draft: bool) -> str:
    """What load_recipe says about this version dir; "" when it loads. The draft
    refusal is raised LAST in the loader, so tolerating it certifies everything
    else already passed."""
    from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
        load_recipe,
    )

    recipe_yaml = _recipe_yaml_in(Path(version_dir))
    if recipe_yaml is None:
        prefix = f"{Path(version_dir).parent.name}_{Path(version_dir).name}"
        return (f"no {prefix}_recipe.yaml in this folder — the recipe file's name must "
                "match its variable and version folders")
    try:
        load_recipe(recipe_yaml)
    except Exception as failure:  # noqa: BLE001 — every failure becomes the editor's message
        message = str(failure)
        if not (tolerate_draft and _DRAFT_REFUSAL_PHRASE in message):
            return message
    # The loader checks that the output schema EXISTS; it is parsed at extract time.
    # An editor that let a corrupt schema through would hand the failure to a run
    # minutes in, so every JSON in the version gets parsed here as well.
    import json

    for schema_file in sorted(Path(version_dir).glob("*.json")):
        try:
            json.loads(schema_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as bad_json:
            return f"{schema_file.name} is not valid JSON: {bad_json}"
    return ""


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_file(version_dir: Path, rel_path: str, new_text: str) -> str:
    """Swap the edit in, ask the loader, roll back on refusal.

    Returns "" when the save stood, else the loader's message — after the original
    content is back on disk. Validating the tree the recipe actually lives in (not a
    temp copy) is what lets prompts, schema, helper and shared-rule references
    resolve exactly as they will at run time."""
    target = _safe_target(version_dir, rel_path)
    original = target.read_text(encoding="utf-8") if target.exists() else None
    _atomic_write(target, new_text)
    error = _load_error(version_dir, tolerate_draft=True)
    if error:
        if original is None:
            target.unlink()
        else:
            _atomic_write(target, original)
        return error
    return ""


def finish_draft(version_dir: Path) -> str:
    """Remove ``needs_editing: true`` — the author saying the content is theirs now.

    Full validation has the last word: the line comes out, the loader is asked with
    NO draft tolerance, and a recipe that still does not load gets the line back and
    the message returned."""
    recipe_yaml = _recipe_yaml_in(Path(version_dir))
    if recipe_yaml is None:
        return "no recipe yaml to finish"
    original = recipe_yaml.read_text(encoding="utf-8")
    without_marker = re.sub(
        r"^# Delete this line once the prompts and schema below describe\n"
        r"^# this variable rather than the one it was copied from\.\n"
        r"^needs_editing: true\n",
        "", original, flags=re.M,
    )
    without_marker = re.sub(r"^needs_editing:.*\n", "", without_marker, flags=re.M)
    if without_marker == original:
        return "this recipe is not a draft — there is no needs_editing line to remove"
    _atomic_write(recipe_yaml, without_marker)
    error = _load_error(version_dir, tolerate_draft=False)
    if error:
        _atomic_write(recipe_yaml, original)
        return error
    return ""


def _copy_and_rewire(template_dir: Path, destination: Path,
                     new_name: str, new_version: str, *, mark_draft: bool) -> list[Path]:
    """The scaffold engine: copy a version dir, rewrite every
    ``<variable>_<version>_`` cross-reference, rename the files underneath, and
    (for a new variable) mark the copy a draft. Returns the renamed files.

    Adding a variable is a directory copy plus ten renames spread over five files —
    the yaml's name, its output_schema line, one prompt path per step, the python
    module reference, and every file carrying the old prefix in its name. Miss one
    and the failure is a path error at extract time, long after the mistake. None of
    it is a decision; it is transcription, so it lives here."""
    template_dir = Path(template_dir)
    old_prefix = f"{template_dir.parent.name}_{template_dir.name}"
    new_prefix = f"{new_name}_{new_version}"

    shutil.copytree(template_dir, destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            body = path.read_text(encoding="utf-8").replace(old_prefix, new_prefix)
            if path.name.endswith("_recipe.yaml"):
                body = re.sub(r"^name:.*$", f"name: {new_name}", body, count=1, flags=re.M)
                body = re.sub(r"^version:.*$", f"version: {new_version}", body,
                              count=1, flags=re.M)
                if mark_draft:
                    body = re.sub(
                        r"^(name: .*)$",
                        r"\1\n# Delete this line once the prompts and schema below describe"
                        r"\n# this variable rather than the one it was copied from."
                        r"\nneeds_editing: true",
                        body, count=1, flags=re.M,
                    )
            path.write_text(body, encoding="utf-8")
    renamed = []
    for path in sorted(destination.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() and old_prefix in path.name:
            moved = path.with_name(path.name.replace(old_prefix, new_prefix))
            path.rename(moved)
            renamed.append(moved)
    return renamed


def scaffold_new_recipe(name: str, template_variable: str,
                        recipes_root: Path | None = None) -> Path:
    """A new variable from one that already works, landing beside its template.

    The copy is a DRAFT: the wiring is right and the words are not — the prompts
    still ask the template's question — so it carries ``needs_editing: true`` and
    the loader refuses to run it until that line is gone."""
    root = Path(recipes_root or RECIPES_ROOT)
    if not _VARIABLE_NAME.fullmatch(name):
        raise ValueError(
            f"{name!r} is not a usable variable name. Lower case, digits and "
            "underscores, starting with a letter — it becomes a folder name, a file "
            "prefix, and a key in every result this variable ever produces."
        )
    templates = sorted(d for d in root.glob(f"**/{template_variable}/v*") if d.is_dir())
    if not templates:
        raise ValueError(f"no variable called {template_variable!r} to copy")
    template = templates[-1]                      # highest version on disk
    destination = template.parent.parent / name / template.name
    # Anywhere in the tree, not just beside the template: a variable name resolves
    # across every collection, so a same-named folder in another collection would
    # make name -> recipe resolution ambiguous for every run from then on.
    taken = sorted(d for d in root.glob(f"**/{name}") if d.is_dir())
    if taken:
        raise ValueError(
            f"{name!r} already exists at {taken[0]}. Pick another name, or edit "
            "the one that is there."
        )
    _copy_and_rewire(template, destination, name, template.name, mark_draft=True)
    return destination


def scaffold_new_version(variable: str, recipes_root: Path | None = None) -> Path:
    """The variable's next version, copied from its highest one.

    Not a draft: the content IS this variable's, and a version bump exists so a
    sealed run's v1 stays untouched while v2 evolves. The copy loads immediately;
    what changes is up to the edit that follows."""
    root = Path(recipes_root or RECIPES_ROOT)
    versions = sorted(d for d in root.glob(f"**/{variable}/v*") if d.is_dir())
    if not versions:
        raise ValueError(f"no variable called {variable!r}")
    newest = versions[-1]
    next_number = int(newest.name.lstrip("v")) + 1
    destination = newest.parent / f"v{next_number}"
    if destination.exists():
        raise ValueError(f"{destination} already exists")
    _copy_and_rewire(newest, destination, variable, f"v{next_number}", mark_draft=False)
    return destination
