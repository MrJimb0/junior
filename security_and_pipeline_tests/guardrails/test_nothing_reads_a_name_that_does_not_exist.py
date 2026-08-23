"""No code may read an attribute off an object whose class does not have it.

A helper that describes which language model is about to read a chart did
``allowlist.entries``. The ``Allowlist`` class takes ``entries`` as a constructor
argument, keeps it in ``self._by_name``, and exposes only ``get(name)`` — so the
attribute does not exist. The result was an AttributeError in the middle of a
nine-patient extract, on the operator's first real use.

It survived three waves of review, adversarial verification of every change, and 1171
passing tests, because all of those were reading the code or running it against
fixtures. The test that claimed to cover that screen passed a config with no
allow-list, so the helper returned two lines before the defect, and the branch every
real project takes was never executed once.

mypy reports it in under a second. It could not have caught it here even if someone had
run it: ``python_version`` was pinned below the interpreter the environment actually
runs, so mypy reported a single numpy stub, checked nothing else, and exited in a way
that reads like a pass. Both halves are guarded below — that the checker still checks
the whole tree, and that this class of defect is at zero.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The error codes that mean "this name is not there" — the ones that are an exception at
# runtime rather than a type that could be written more precisely. `misc` carries
# "None is not callable", which is the same defect wearing a different code. Codes about
# annotations being loose (arg-type, assignment, var-annotated) are deliberately not
# here: this guards against crashes, not against imprecision.
A_NAME_THAT_IS_NOT_THERE = ("attr-defined", "union-attr", "call-arg", "call-overload", "misc")


def _mypy() -> str:
    # Probes the module, because that is what is run below. Checking for an executable
    # on PATH instead would let a machine with mypy installed outside this environment
    # past the skip and into a run that produces nothing.
    installed = subprocess.run(
        [sys.executable, "-c", "import mypy"], capture_output=True, check=False
    )
    if installed.returncode != 0:
        pytest.skip("mypy is a dev dependency and is not installed in this environment")
    finished = subprocess.run(
        [sys.executable, "-m", "mypy"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    return finished.stdout + finished.stderr


@pytest.fixture(scope="module")
def reported() -> str:
    """mypy's output, having established that it actually checked the tree.

    The precondition belongs here rather than in a test of its own. Held separately, a
    checker that stopped before reading any code left the crash-class scan below with an
    empty string to search: no matches, green, nothing inspected — while the sibling test
    went red in a way that reads like a local environment problem. That is the same shape
    as the misconfiguration this file exists because of, so the two cannot be separable."""
    output = _mypy()
    counted = re.search(r"checked (\d+) source files?", output)
    assert counted, (
        "mypy did not report how many files it checked, which is what it does when it "
        f"stopped before checking any:\n{output[-1500:]}"
    )
    assert int(counted.group(1)) > 100, (
        f"mypy checked only {counted.group(1)} files — it is configured to check the "
        "roots below, so something is stopping it early."
    )
    return output


def test_every_root_of_the_tree_is_named_to_the_checker() -> None:
    """A count notices a root that stopped being checked. It cannot notice a root that
    was never named — and that is what happened: the CLI moved out of src/jr_pipeline
    into apps_and_interfaces/ and this setting said "src" for a week afterwards, so ten
    thousand lines of operator surface, the code most likely to break in somebody
    else's hands, silently left the check. Three real errors were sitting in it."""
    import tomllib

    settings = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    named = settings["tool"]["mypy"]["files"]
    missing = [root for root in ("src", "apps_and_interfaces") if root not in named]
    assert not missing, (
        f"these roots hold shipped code the type checker is never pointed at: {missing}"
    )


def test_the_type_checker_still_checks_the_whole_tree(reported: str) -> None:
    """Reaching this at all means the fixture's precondition held."""
    assert "checked" in reported


def test_nothing_reads_a_name_that_does_not_exist(reported: str) -> None:
    """The defect itself. Every one of these is an AttributeError or a TypeError the
    moment that line runs, waiting on whichever input reaches it first."""
    offending = [
        line for line in reported.splitlines()
        if any(f"[{code}]" in line for code in A_NAME_THAT_IS_NOT_THERE)
    ]
    assert not offending, (
        "these read a name the type checker knows is not there, so they raise as soon "
        "as that line runs:\n  " + "\n  ".join(offending)
    )
