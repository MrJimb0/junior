"""print_slurm_commands cd's to the code root and ensures logs/ before each sbatch.

The stage scripts' #SBATCH --output=logs/ paths are relative to the submit directory, so
the printed commands must cd to $JR_CODE_ROOT and mkdir -p logs — otherwise a submission
from the wrong directory kills every array task with no captured stderr.
"""
from __future__ import annotations

from jr_pipeline.runtime_infrastructure.cohort_runner import (
    CohortSettings,
    print_slurm_commands,
)


def test_printed_sbatch_stages_cd_to_code_root_and_make_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path))
    inputs = tmp_path / "patients"
    (inputs / "P1").mkdir(parents=True)
    (inputs / "P2").mkdir()

    print_slurm_commands(CohortSettings(input_folder=inputs), "20260101_120000_abcd")
    out = capsys.readouterr().out

    guard = 'cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch'
    assert out.count(guard) == 3  # ingest, embed, index each ensure logs/ from the code root
    assert "export JR_CODE_ROOT=" in out
