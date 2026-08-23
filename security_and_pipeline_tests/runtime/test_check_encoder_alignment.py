"""check_encoder_alignment follows the pipeline's data-root + fingerprint-field rules.

It must resolve the data root via data_root() (honoring JR_DATA_ROOT) when the config has
no output_root, and compare exactly the encoder-owned VECTOR_AFFECTING_FINGERPRINT_FIELDS
rather than a hardcoded field tuple that would silently drift.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_CHECK = REPO / "deployment/Local_SLURM_Cluster/slurm/check_encoder_alignment.py"


def test_resolves_data_root_from_jr_data_root_when_no_output_root(tmp_path, monkeypatch):
    config = tmp_path / "consume.yaml"
    config.write_text("run_id: 20260101_120000_abcd\n", encoding="utf-8")
    data_tree = tmp_path / "datatree"
    env = {"JR_DATA_ROOT": str(data_tree), "PYTHONPATH": "src", "PATH": __import__("os").environ["PATH"]}
    proc = subprocess.run(
        [sys.executable, str(_CHECK), "--config", str(config), "--patient", "P1"],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    # no synced corpus -> it exits naming the sidecar path, which must sit under JR_DATA_ROOT
    assert str(data_tree) in (proc.stdout + proc.stderr)


def test_compared_fields_come_from_the_encoder_owned_list():
    source = _CHECK.read_text(encoding="utf-8")
    assert "VECTOR_AFFECTING_FINGERPRINT_FIELDS" in source
    # the old hardcoded tuple must be gone so the two lists can't drift apart
    assert '"model_sha256", "tokenizer_hash", "pooling"' not in source
