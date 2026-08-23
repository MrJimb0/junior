"""Nothing in this repository may name a real site.

The deployment examples are a real deployment tree with every site fact replaced,
and the rest of the repository is held to the same standard: no institution name,
cluster name, internal gateway host, or lab path anywhere a reader can see.

The risk is not the first commit, it is the tenth: somebody debugging on a real cluster
edits the example instead of their own copy, a working path goes in, and it reads like
a template until you look. So the identifiers are asserted by name rather than trusted
to review.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = ["Local_SLURM_Cluster", "NeroGCP"]

# What "a site fact" means here: something that identifies WHOSE deployment this was.
# The word is not the point — a site's own name for a service is as identifying as its
# hostname, and a project path carrying a group name is worse than either.
#
# starr carries word boundaries because it is also an English fragment: "starring" and
# "starred" match it unbounded, and both turn up in ordinary prose. A pattern that fires
# on the word "starred" teaches everyone to read past this test's output.
#
# The second line is the private half: a storage remote, a lab folder, an internal
# project name, a transfer host, a support queue. These were missing, and that is how
# an entire deployment example shipped with a working Box path in it — the de-identifying
# pass only ever replaced a cluster name, so everything nobody thought to grep for
# survived. A site's own words for its own things are what identify it.
NAMES_A_SITE = re.compile(
    r"stanford|carina|akurian|securegpt|aihubapi|\bstarr\b|/projects/|/oak/|/scratch/"
    r"|\bsunet\b|[a-z0-9-]+\.stanfordhealthcare\.org"
    r"|medbox|box.?medicine|dickerson_lab|\bjr_pip\b|c2-transfer|\bsrcc\b|/share/pi/",
    re.IGNORECASE,
)


def _example_files(name: str) -> list[Path]:
    root = REPO / "deployment" / name
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in
                  {".sh", ".yaml", ".yml", ".py", ".md"})


@pytest.mark.parametrize("example", EXAMPLES)
def test_no_file_names_a_site(example: str):
    offending = []
    for path in _example_files(example):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if NAMES_A_SITE.search(line):
                offending.append(f"{path.relative_to(REPO)}:{number}: {line.strip()[:90]}")
    assert not offending, (
        "a deployment example names a real site — these ship publicly:\n  "
        + "\n  ".join(offending[:15])
    )


@pytest.mark.parametrize("example", EXAMPLES)
def test_the_allowlist_is_a_template_not_a_working_endpoint(example: str):
    """An allowlist that works out of the box is one nobody edits. Every endpoint must
    still carry a placeholder, because the file's whole job is to be a deliberate
    decision about where a patient's notes may be sent."""
    loaded = yaml.safe_load(
        (REPO / "deployment" / example / "llm_allowlist.yaml").read_text(encoding="utf-8"))
    endpoints = loaded["allowed_endpoints"]
    assert endpoints, "the example allowlist has no endpoints to show the shape of"
    unedited = [e["name"] for e in endpoints if "<" not in yaml.safe_dump(e)]
    assert not unedited, (
        f"these endpoints would work as shipped, so nobody has to think: {unedited}")


@pytest.mark.parametrize("example", EXAMPLES)
def test_it_still_carries_the_scripts_it_is_an_example_of(example: str):
    """De-identifying must not quietly become deleting. These are the pieces that make
    it a deployment rather than a folder of placeholders."""
    slurm = REPO / "deployment" / example / "slurm"
    missing = [n for n in ("seal.sh", "ingest.sh", "embed.sh", "index.sh", "RUNBOOK.md",
                           "embed_override.yaml") if not (slurm / n).exists()]
    assert not missing, f"{example} is missing {missing}"
    assert (REPO / "deployment" / example / "README.md").exists()

# The repo-wide sweep. Two kinds of file are exempt, each for a stated reason:
#   - examples/Test_Patient/ is a fabricated chart whose text deliberately reads like
#     a real one, hospital name included; de-identifying fiction helps nobody.
#   - the police files: the PHI egress guard and the site-fact tests carry the very
#     patterns they exist to catch, and two guardrail tests use catchable strings as
#     fixtures. Excluding them is what lets the pattern list live in code at all.
# The author's public contact address is attribution, not a site fact.
_EXEMPT_TREES = ("examples/Test_Patient",)
_POLICE_FILES = frozenset({
    "src/jr_pipeline/runtime_enforcing_safety_and_reproducibility/phi/phi_leak_prevention_checks.py",
    "security_and_pipeline_tests/guardrails/test_deployment_examples_name_no_site.py",
    "security_and_pipeline_tests/guardrails/test_export_egress_scan.py",
    "security_and_pipeline_tests/guardrails/test_exhaust_write_gate.py",
})
_ALLOWED_LITERALS = ("jcdicker@stanford.edu",)
# README.md says plainly which institution the author is at and which platforms he can
# help with, which is the point of it — a reader deciding whether this tool fits their
# site needs to know. That is a stated affiliation, not a leaked site fact: it carries no
# host, no path, no share, nothing anybody could act on. The rule this file exists to
# enforce is about working values surviving into the deployment templates, and those are
# still swept. junior.egg-info is the same prose again, copied into PKG-INFO by an
# editable install, so it is a build artifact rather than a shipped file.
_DELIBERATE_AFFILIATION = frozenset({"README.md"})
# models/ is gitignored and holds downloaded weights, which are somebody else's files.
# A tokenizer vocabulary is forty thousand English fragments and some of them are place
# names, so scanning it reports on the model rather than on this repository. Leaving it
# in means the suite goes red the moment an operator follows the setup instructions.
_SKIP_DIRS = {".git", ".venv", ".venv-arm64", "__pycache__", ".pytest_cache",
              ".ruff_cache", ".mypy_cache", "Junior Workbench.app", "data",
              "models", "your_projects", "logs"}
_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".icns", ".pdf", ".parquet",
                    ".npy", ".safetensors", ".bin", ".scpt", ".pyc", ".zip", ".whl"}


def _repo_files_to_scan() -> list[Path]:
    out = []
    for path in REPO.rglob("*"):
        rel = path.relative_to(REPO)
        parts = set(rel.parts)
        if parts & _SKIP_DIRS or not path.is_file():
            continue
        rel_str = str(rel)
        if any(rel_str.startswith(tree) for tree in _EXEMPT_TREES):
            continue
        if rel_str in _POLICE_FILES or path.suffix in _BINARY_SUFFIXES:
            continue
        if rel_str in _DELIBERATE_AFFILIATION or ".egg-info" in rel_str:
            continue
        out.append(path)
    return sorted(out)


def test_nothing_in_the_repository_names_a_site():
    offending = []
    for path in _repo_files_to_scan():
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            for literal in _ALLOWED_LITERALS:
                line = line.replace(literal, "")
            if NAMES_A_SITE.search(line):
                offending.append(f"{path.relative_to(REPO)}:{number}: {line.strip()[:90]}")
    assert not offending, (
        "a shipped file names a real site:\n  " + "\n  ".join(offending[:20])
    )

