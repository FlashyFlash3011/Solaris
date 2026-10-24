"""Tensor-parallel primitives on 2 CPU ranks (gloo).

Builds the two standard tensor-parallel linear layers from the primitives and
checks forward values and gradients against an ordinary single-process layer:

* column-parallel:  copy_to → local x @ W_rowsᵀ → gather(dim=-1)
* row-parallel:     scatter(dim=-1) → local x_cols @ W_colsᵀ → reduce
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


def _worker(rank: int, world: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from solaris.distributed import (
            copy_to_model_parallel_region,
            gather_from_model_parallel_region,
            reduce_from_model_parallel_region,
            scatter_to_model_parallel_region,
        )

        torch.manual_seed(0)
        x_full = torch.randn(4, 8)
        W = torch.randn(6, 8)  # out=6, in=8
        dy = torch.randn(4, 6)

        # reference
        x_ref = x_full.clone().requires_grad_(True)
        W_ref = W.clone().requires_grad_(True)
        y_ref = x_ref @ W_ref.T
        (y_ref * dy).sum().backward()

        # column-parallel: each rank owns output rows of W
        rows = slice(rank * 3, (rank + 1) * 3)
        x = x_full.clone().requires_grad_(True)
        W_local = W[rows].clone().requires_grad_(True)
        y = gather_from_model_parallel_region(copy_to_model_parallel_region(x) @ W_local.T, dim=-1)
        torch.testing.assert_close(y, y_ref)
        (y * dy).sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        torch.testing.assert_close(W_local.grad, W_ref.grad[rows])

        # row-parallel: each rank owns input columns of W
        cols = slice(rank * 4, (rank + 1) * 4)
        x2 = x_full.clone().requires_grad_(True)
        W2_local = W[:, cols].clone().requires_grad_(True)
        y2 = reduce_from_model_parallel_region(
            scatter_to_model_parallel_region(x2, dim=-1) @ W2_local.T
        )
        torch.testing.assert_close(y2, y_ref)
        (y2 * dy).sum().backward()
        torch.testing.assert_close(x2.grad, x_ref.grad)
        torch.testing.assert_close(W2_local.grad, W_ref.grad[:, cols])

        # default dim=0 keeps working (batch sharding)
        shard = scatter_to_model_parallel_region(x_full)
        torch.testing.assert_close(gather_from_model_parallel_region(shard), x_full)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_tensor_parallel_linear_layers_match_single_process():
    mp.spawn(_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True)
