"""Domain-separated HMAC surrogates are deterministic + irreversible.

The same raw id yields the same surrogate (linkage preserved) but a *different*
surrogate in a different field / record type / scope (domain separation), and the
output never contains the raw value.
"""
from __future__ import annotations

import pytest

from jr_pipeline.runtime_infrastructure.exhaust.surrogates import (
    LinkageScope,
    is_surrogate,
    make_surrogate,
)

SECRET = b"\x01" * 32
OTHER_SECRET = b"\x02" * 32


def _s(secret=SECRET, *, scope="run", record_type="selection_judgment", output_field="patient_surrogate", raw="Test_Patient"):
    return make_surrogate(secret, scope=scope, record_type=record_type, output_field=output_field, raw=raw)


def test_surrogate_is_deterministic_for_fixed_inputs():
    assert _s() == _s()


def test_surrogate_format_is_well_formed_and_id_free():
    s = _s(raw="Test_Patient")
    assert s.startswith("surr:v1:run:patient_surrogate:")
    assert is_surrogate(s)
    assert "Test_Patient" not in s  # irreversible: raw value absent


def test_domain_separation_across_field_record_type_scope_and_secret():
    base = _s()
    assert _s(output_field="chunk_surrogate") != base       # different field
    assert _s(record_type="extraction_outcome") != base     # different record type
    assert _s(scope="study") != base                        # different scope
    assert _s(secret=OTHER_SECRET) != base                  # different secret


def test_distinct_raw_values_give_distinct_surrogates():
    assert _s(raw="patient_a") != _s(raw="patient_b")


def test_is_surrogate_rejects_raw_and_malformed_values():
    assert not is_surrogate("Test_Patient:notes:0:0")  # a raw chunk id
    assert not is_surrogate("abc123def4567890")          # bare hex
    assert not is_surrogate("surr:v1:run:field")         # truncated
    assert not is_surrogate(None)


def test_is_surrogate_pins_the_exact_mint_length():
    # A hex-encoded raw id stuffed into an id slot must NOT be trusted as a surrogate
    # (the hex tail is pinned to the exact 24-char mint length).
    real = _s()
    assert is_surrogate(real)
    assert not is_surrogate("surr:v1:run:patient:" + "a" * 32)  # wrong length
    assert not is_surrogate("surr:v1:run:patient:" + "a" * 16)  # wrong length


def test_make_surrogate_rejects_ambiguous_or_unrecognizable_components():
    # delimiter injection / a ':' in the field would mint a surrogate is_surrogate rejects
    with pytest.raises(ValueError):
        make_surrogate(SECRET, scope="run", record_type="a|b", output_field="c", raw="x")
    with pytest.raises(ValueError):
        make_surrogate(SECRET, scope="run", record_type="r", output_field="a:b", raw="x")
    with pytest.raises(ValueError):
        make_surrogate(SECRET, scope="galaxy", record_type="r", output_field="f", raw="x")


def test_every_minted_surrogate_is_recognized():
    # mint and recognizer stay in lockstep across the real fields/record types used
    for rt in ("selection_judgment", "clinical_invariant", "extraction_outcome"):
        for field in ("patient_surrogate", "chunk_surrogate", "group"):
            assert is_surrogate(_s(record_type=rt, output_field=field))


def test_empty_secret_is_rejected():
    with pytest.raises(ValueError):
        make_surrogate(b"", scope="run", record_type="r", output_field="f", raw="x")


def test_scope_enum_values_match_the_wire_format():
    assert {s.value for s in LinkageScope} == {"run", "study", "site"}
