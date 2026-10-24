"""Multi-process (gloo, CPU) tests for the tensor-parallel spectral path.

Each test spawns 2 ranks.  The distributed computation on row shards must
reproduce the single-process result exactly (forward and gradients).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spectral_worker(rank: int, world: int, port: int, H: int, W: int, modes: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from solaris.nn import SpectralConv2d

        torch.manual_seed(0)
        ref = SpectralConv2d(3, 2, modes, modes)
        par = SpectralConv2d(3, 2, modes, modes, distributed=True, tp_group=dist.group.WORLD)
        par.load_state_dict(ref.state_dict())

        x = torch.randn(2, 3, H, W)
        target = torch.randn(2, 2, H, W)
        rows = slice(rank * H // world, (rank + 1) * H // world)

        y_ref = ref(x)
        (y_ref * target).sum().backward()

        x_local = x[:, :, rows].clone().requires_grad_(True)
        y_par = par(x_local)
        (y_par * target[:, :, rows]).sum().backward()

        torch.testing.assert_close(y_par, y_ref[:, :, rows], atol=1e-5, rtol=1e-5)
        for name in ("weights1", "weights2"):
            p = getattr(par, name)
            # a rank that owns none of the retained modes never touches the weights
            g = p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(g)  # each rank owns a slab of frequency columns
            torch.testing.assert_close(g, getattr(ref, name).grad, atol=1e-5, rtol=1e-5)
    finally:
        dist.destroy_process_group()


def _gloo_available() -> bool:
    return dist.is_available() and dist.is_gloo_available()


@pytest.mark.skipif(not _gloo_available(), reason="gloo backend unavailable")
@pytest.mark.parametrize(
    "H,W,modes",
    [
        (16, 16, 4),  # all kept modes on rank 0
        (16, 30, 8),  # kept modes straddle both ranks; W_freq=16 splits evenly
        (8, 13, 6),  # odd W, padded frequency axis
    ],
)
def test_spectral_conv2d_two_ranks_matches_single_process(H, W, modes):
    mp.spawn(_spectral_worker, args=(WORLD, _free_port(), H, W, modes), nprocs=WORLD, join=True)


def _create_group_worker(rank: int, world: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0")
    from solaris.distributed import DistributedManager

    DistributedManager.reset()
    manager = DistributedManager()
    manager.initialize(device_type="cpu")
    try:
        group = manager.create_group("only_rank0", ranks=[0])
        if rank == 0:
            assert isinstance(group, dist.ProcessGroup)
            assert manager.get_group("only_rank0") is group
        else:
            assert group is None
            with pytest.raises(KeyError):
                manager.get_group("only_rank0")
    finally:
        DistributedManager.reset()


@pytest.mark.skipif(not _gloo_available(), reason="gloo backend unavailable")
def test_create_group_returns_none_on_non_members():
    mp.spawn(_create_group_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True)
