"""Guardrail: detect version drift across the codebase.

Checks that version strings, package names, and model references
are consistent across pyproject.toml, __init__.py, quickstart.py,
and deployment configs. Drift means someone updated one file but
forgot the others.

Run as CI check:
    pytest tests/guardrails/test_version_drift.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(relpath: str) -> str:
    return (REPO_ROOT / relpath).read_text(encoding="utf-8")


class TestVersionDrift:
    """All version strings and package names agree across the repo."""

    def test_pyproject_and_init_version_match(self):
        pyproject = _read("pyproject.toml")
        init_py = _read("src/jr_pipeline/__init__.py")

        pyproject_version = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
        init_version = re.search(r'__version__\s*=\s*"([^"]+)"', init_py)

        assert pyproject_version, "No version in pyproject.toml"
        assert init_version, "No __version__ in __init__.py"
        assert pyproject_version.group(1) == init_version.group(1), (
            f"Version mismatch: pyproject.toml={pyproject_version.group(1)} "
            f"vs __init__.py={init_version.group(1)}"
        )

    def test_entry_point_matches_cli_location(self):
        """The pyproject.toml entry point must point at the actual CLI module."""
        pyproject = _read("pyproject.toml")
        entry_match = re.search(r'jr-pipeline\s*=\s*"([^"]+)"', pyproject)
        assert entry_match, "No jr-pipeline entry point in pyproject.toml"

        module_path = entry_match.group(1).split(":")[0]
        # pyproject declares two package roots: src/ (the engine) and the repo root
        # (apps_and_interfaces, the operator surfaces the entry point lives in).
        as_path = module_path.replace(".", "/")
        candidates = [
            root / relative
            for root in (REPO_ROOT / "src", REPO_ROOT)
            for relative in (Path(as_path + ".py"), Path(as_path) / "__init__.py")
        ]
        assert any(candidate.exists() for candidate in candidates), (
            f"Entry point {module_path} does not exist as a module"
        )

    def test_no_stale_old_package_names_in_src(self):
        """Check for leftover references to old package/module names."""
        stale_patterns = [
            "core_plumbing",
            "data_audit_trail",     # use runtime_enforcing_safety_and_reproducibility instead
            "debugging_tools",
            "phi_governance",
            "phi_data_stream",
        ]

        src_dir = REPO_ROOT / "src" / "jr_pipeline"
        findings = []
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text(encoding="utf-8")
            for pattern in stale_patterns:
                if f"jr_pipeline.{pattern}" in content:
                    findings.append(f"{py_file.relative_to(REPO_ROOT)}: "
                                    f"references 'jr_pipeline.{pattern}'")

        assert findings == [], (
            "Stale module references found:\n" + "\n".join(findings)
        )

    def test_sensitive_label_env_vars_consistent(self):
        """The layout module should use env-var-driven labels, not hardcoded."""
        layout = _read("src/jr_pipeline/runtime_infrastructure/data_directory_layout_and_safe_writes.py")
        assert "os.environ.get" in layout, (
            "Sensitive data labels should be configurable via env vars"
        )
        assert "CONTAINS_PHI" in layout, (
            "Default sensitive label should be CONTAINS_PHI"
        )

    def test_all_pipeline_steps_have_init(self):
        """Every pipeline module subdir must have an __init__.py."""
        modules_dir = REPO_ROOT / "src" / "jr_pipeline" / "pipeline_steps"
        if not modules_dir.exists():
            pytest.skip("pipeline_steps not found")

        missing = []
        for d in modules_dir.iterdir():
            if d.is_dir() and not d.name.startswith((".", "_")):
                if not (d / "__init__.py").exists():
                    missing.append(d.name)

        assert missing == [], (
            f"Pipeline modules missing __init__.py: {missing}"
        )

    def test_all_extraction_recipes_have_required_files(self):
        """Each recipe version must have recipe YAML + output schema."""
        recipes_dir = REPO_ROOT / "var_extraction_recipes"
        if not recipes_dir.exists():
            pytest.skip("var_extraction_recipes not found")

        issues = []
        # Version dirs may sit flat (<variable>/v<N>/) or nested under a collection
        # (<collection>/<variable>/v<N>/) — find them at any depth.
        for version_dir in recipes_dir.rglob("v*"):
            if not version_dir.is_dir() or not version_dir.name[1:].isdigit():
                continue
            var_dir = version_dir.parent
            if var_dir.name.startswith((".", "_")):
                continue
            prefix = f"{var_dir.name}_{version_dir.name}"
            recipe = version_dir / f"{prefix}_recipe.yaml"
            schema = version_dir / f"{prefix}_output_schema.json"
            if not recipe.exists():
                issues.append(f"{var_dir.name}/{version_dir.name}: missing {recipe.name}")
            if not schema.exists():
                issues.append(f"{var_dir.name}/{version_dir.name}: missing {schema.name}")

        assert issues == [], (
            "Recipe structure issues:\n" + "\n".join(issues)
        )
