"""One shared PHI content vocabulary used by tests + production.

The egress scanner and the phi-containment guard share a single set of PHI content
patterns in ``phi.phi_leak_prevention_checks.PHI_CONTENT_PATTERNS`` — the single scrub
vocabulary. This guard pins that single source.
"""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.phi.phi_leak_prevention_checks import (
    PHI_CONTENT_PATTERNS,
    scan_text_for_phi_content,
)


def test_vocabulary_catches_core_phi():
    assert scan_text_for_phi_content("123-45-6789")           # SSN
    assert scan_text_for_phi_content("MRN: 12345678")         # MRN
    assert scan_text_for_phi_content("Chief Complaint: x")    # clinical note text
    assert scan_text_for_phi_content("seen 2026-01-15")       # absolute date
    assert scan_text_for_phi_content("patient_name = Jane")   # name field


def test_vocabulary_allows_clean_text_and_month_stamp():
    # YYYY-MM administrative month stamp is NOT an absolute date and stays clean
    assert scan_text_for_phi_content("emitted_month = 2026-01") == []
    assert scan_text_for_phi_content("recall@5 = 0.83") == []


def test_production_egress_scanner_uses_the_shared_vocabulary():
    from jr_pipeline.evaluating_pipeline_performance import export_shareable_metadata as esm

    # the scanner sources the one shared list, never a private copy
    assert not hasattr(esm, "_FORBIDDEN_PATTERNS")
    assert esm.PHI_CONTENT_PATTERNS is PHI_CONTENT_PATTERNS
