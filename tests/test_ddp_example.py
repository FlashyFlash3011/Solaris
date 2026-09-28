"""End-to-end check that examples/train_fno_darcy.py really trains data-parallel.

Launches the example with ``torchrun --nproc_per_node=2`` on CPU (gloo).
"""

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


EXAMPLE = REPO / "examples" / "train_fno_darcy.py"


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed unavailable")
def test_train_fno_darcy_runs_data_parallel_on_two_processes(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        f"--master_port={_free_port()}",
        str(EXAMPLE),
        "--device",
        "cpu",
        "--epochs",
        "2",
        "--n_samples",
        "64",
        "--resolution",
        "16",
        "--hidden",
        "8",
        "--modes",
        "4",
        "--n_layers",
        "2",
        "--checkpoint_dir",
        str(tmp_path),
    ]
    run = subprocess.run(cmd, env=env, cwd=tmp_path, capture_output=True, text=True, timeout=180)
    out = run.stdout + run.stderr
    assert run.returncode == 0, out[-3000:]
    assert "world_size=2" in out
    # rank 0 alone logs epochs and writes the checkpoint
    assert len(re.findall(r"Epoch\s+2 \|", out)) == 1
    assert (tmp_path / "best_fno.pt").exists()
    # every rank ends with identical weights (DDP keeps replicas in sync)
    checksums = set(re.findall(r"param_checksum=(\S+)", out))
    assert len(checksums) == 1, checksums
