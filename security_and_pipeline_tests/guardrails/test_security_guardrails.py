"""Guardrail: verify security boundaries are intact.

Checks that the public API denylist is active, that the sensitive data
layout enforces the PHI boundary, and that no code path bypasses the
provider registry (which enforces the denylist).

Run as CI check:
    pytest tests/guardrails/test_security_guardrails.py
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDenylist:
    """The public API denylist is active and non-empty."""

    def test_denylist_has_minimum_hosts(self):
        from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
            FORBIDDEN_HOSTS,
        )
        assert len(FORBIDDEN_HOSTS) >= 10, (
            f"Denylist only has {len(FORBIDDEN_HOSTS)} hosts — "
            f"expected at least 10 public API providers blocked"
        )

    def test_major_public_apis_are_blocked(self):
        from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
            FORBIDDEN_HOSTS,
        )
        must_block = ["api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com"]
        for host in must_block:
            assert host in FORBIDDEN_HOSTS, f"Critical: {host} is NOT on the denylist"

    def test_denylist_cannot_be_overridden_by_env_var(self):
        """The denylist is hardcoded, not configurable — removing a host requires a code change."""
        denylist_file = REPO_ROOT / "src/jr_pipeline/pipeline_steps/step_7_extract_variables/providers/llm_endpoint_denylist.py"
        content = denylist_file.read_text(encoding="utf-8")
        assert "os.environ" not in content, (
            "Denylist must be hardcoded, not env-var configurable"
        )


class TestDataLayoutSecurity:
    """The layout module enforces PHI containment by construction."""

    def test_all_patient_paths_route_to_phi_dir(self):
        """Every function that takes a patient_id must route to CONTAINS_PHI."""
        from jr_pipeline.runtime_infrastructure import (
            data_directory_layout_and_safe_writes as layout,
        )

        patient_functions = [
            name for name in dir(layout)
            if callable(getattr(layout, name))
            and "patient" in name.lower()
            and not name.startswith("_")
            and "patient_id" in getattr(getattr(layout, name), "__code__", type("", (), {"co_varnames": ()})()).co_varnames
            # validate_patient_id is a validator (returns the id), not a path builder.
            and name not in ("patient_data_dir", "validate_patient_id")
        ]

        import os
        os.environ.setdefault("JR_DATA_ROOT", "./data")

        for func_name in patient_functions:
            func = getattr(layout, func_name)
            try:
                if "run_id" in func.__code__.co_varnames:
                    result = func(run_id="test", patient_id="test_pt")
                else:
                    result = func(patient_id="test_pt")
            except TypeError:
                result = func("test_pt")

            path_str = str(result)
            assert "CONTAINS_PHI" in path_str or layout.SENSITIVE_LABEL in path_str, (
                f"{func_name}() routes to {path_str} — "
                f"expected CONTAINS_PHI for any patient-level function"
            )

    def test_no_phi_dir_has_no_patient_functions(self):
        """NO_PHI path helpers should not accept patient_id."""
        from jr_pipeline.runtime_infrastructure import (
            data_directory_layout_and_safe_writes as layout,
        )

        no_phi_functions = [
            name for name in dir(layout)
            if callable(getattr(layout, name))
            and "no_phi" in name.lower()
            and not name.startswith("_")
            and "deprecated" not in (getattr(getattr(layout, name), "__doc__", "") or "").lower()
        ]

        for func_name in no_phi_functions:
            func = getattr(layout, func_name)
            params = list(func.__code__.co_varnames[:func.__code__.co_argcount])
            assert "patient_id" not in params, (
                f"{func_name}() accepts patient_id — "
                f"NO_PHI functions must not handle per-patient data"
            )


class TestProviderSecurity:
    """LLM provider system enforces security boundaries."""

    def test_live_provider_checks_denylist(self):
        """A live network provider must call check_url from the denylist before any
        LLM call."""
        provider_file = REPO_ROOT / "src/jr_pipeline/pipeline_steps/step_7_extract_variables/providers/openai_compatible_provider.py"
        content = provider_file.read_text(encoding="utf-8")
        assert "check_url" in content, "the live provider must call check_url from the denylist"
        assert "llm_endpoint_denylist" in content, "the live provider must import the denylist module"
