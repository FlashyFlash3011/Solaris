"""Sharded (FSDP + torch.distributed.checkpoint) save/load round trip on 2 CPU ranks."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, path: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from solaris.utils import load_checkpoint_distributed, save_checkpoint_distributed

        def make(seed):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 4)
            )
            model = FSDP(net, device_id=torch.device("cpu"), use_orig_params=True)
            return model, torch.optim.Adam(model.parameters(), lr=1e-2)

        model, opt = make(0)
        x = torch.randn(4, 8)
        model(x).sum().backward()
        opt.step()
        save_checkpoint_distributed(path, model, opt, epoch=3, loss=0.5)

        other, other_opt = make(123)  # different weights before loading
        state = load_checkpoint_distributed(path, other, other_opt)
        with torch.no_grad():
            torch.testing.assert_close(other(x), model(x))
        assert state["epoch"] == 3
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_fsdp_checkpoint_round_trip(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path / "ckpt")), nprocs=2, join=True)
