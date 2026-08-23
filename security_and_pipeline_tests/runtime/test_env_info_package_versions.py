"""The reproducibility snapshot records dependency versions under their DISTRIBUTION
names. PyYAML is the classic trap: it installs the ``yaml`` module but its
distribution is "PyYAML", so querying "yaml" silently records null."""
from __future__ import annotations

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility import (
    frozen_code_snapshot,
)


def test_env_info_resolves_pyyaml_version_under_distribution_name():
    packages = frozen_code_snapshot._env_info()["packages"]

    assert "PyYAML" in packages
    assert packages["PyYAML"] is not None  # resolves now that we query the distribution name
    assert "yaml" not in packages          # the old (import-name) key is gone
