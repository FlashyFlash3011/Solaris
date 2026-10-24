# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel autograd primitives.

Four differentiable collective operations for sharding model parameters across
a tensor-parallel process group.  Each is a ``torch.autograd.Function`` whose
forward and backward passes together form a consistent gradient flow:

+----------------------------+---------------------+---------------------+
| Function                   | Forward             | Backward            |
+============================+=====================+=====================+
| scatter_to_model_parallel  | split along ``dim`` | all-gather          |
| gather_from_model_parallel | all-gather          | split along ``dim`` |
| reduce_from_model_parallel | all-reduce          | identity            |
| copy_to_model_parallel     | identity            | all-reduce          |
+----------------------------+---------------------+---------------------+

Usage
-----
::

    from solaris.distributed.mappings import (
        copy_to_model_parallel_region,
        gather_from_model_parallel_region,
        reduce_from_model_parallel_region,
        scatter_to_model_parallel_region,
    )

    # Column-parallel linear: each rank holds some *output* rows of W
    y_local = copy_to_model_parallel_region(x, group=tp_group) @ W_rows.T
    y = gather_from_model_parallel_region(y_local, group=tp_group, dim=-1)

    # Row-parallel linear: each rank holds some *input* columns of W
    x_local = scatter_to_model_parallel_region(x, group=tp_group, dim=-1)
    y = reduce_from_model_parallel_region(x_local @ W_cols.T, group=tp_group)
"""

import torch
import torch.distributed as dist


def _get_world_size(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def _get_rank(group: dist.ProcessGroup | None) -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def _split(x: torch.Tensor, world: int, rank: int, dim: int) -> torch.Tensor:
    if x.shape[dim] % world:
        raise ValueError(
            f"dimension {dim} of size {x.shape[dim]} is not divisible by world size {world}"
        )
    return x.chunk(world, dim=dim)[rank].contiguous()


def _all_gather(x: torch.Tensor, group: dist.ProcessGroup | None, world: int, dim: int):
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Forward: keep this rank's slice along ``dim``.  Backward: all-gather."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:
        ctx.group, ctx.dim = group, dim
        world = _get_world_size(group)
        if world == 1:
            return x
        return _split(x, world, _get_rank(group), dim)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world = _get_world_size(ctx.group)
        if world == 1:
            return grad, None, None
        return _all_gather(grad, ctx.group, world, ctx.dim), None, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Forward: all-gather along ``dim``.  Backward: keep this rank's slice."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None, dim: int) -> torch.Tensor:
        ctx.group, ctx.dim = group, dim
        world = _get_world_size(group)
        if world == 1:
            return x
        return _all_gather(x, group, world, dim)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        world = _get_world_size(ctx.group)
        if world == 1:
            return grad, None, None
        return _split(grad, world, _get_rank(ctx.group), ctx.dim), None, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """Forward: all-reduce.  Backward: identity (pass-through)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        world = _get_world_size(group)
        if world == 1:
            return x
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Forward: identity.  Backward: all-reduce (sum gradients across ranks)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx.group
        world = _get_world_size(group)
        if world == 1:
            return grad, None
        out = grad.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out, None


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


def scatter_to_model_parallel_region(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """Split *x* along ``dim`` and keep only the local rank's slice.

    Use before a row-parallel linear layer (``dim=-1``: each rank holds a
    column shard of the weight and the matching slice of input features).
    ``x.shape[dim]`` must be divisible by the group size.
    """
    return _ScatterToModelParallelRegion.apply(x, group, dim)


def gather_from_model_parallel_region(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """All-gather *x* along ``dim`` across the tensor-parallel group.

    Use after a column-parallel linear layer (``dim=-1``) to reconstruct the
    full output features.
    """
    return _GatherFromModelParallelRegion.apply(x, group, dim)


def reduce_from_model_parallel_region(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-reduce *x* across the tensor-parallel group (sum).

    Use after a row-parallel linear layer to sum partial results.
    Gradient is passed through unchanged (identity backward).
    """
    return _ReduceFromModelParallelRegion.apply(x, group)


def copy_to_model_parallel_region(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Broadcast *x* to all TP ranks in the forward pass (identity).

    Use before a column-parallel linear layer so all ranks receive the same
    input.  Gradient is all-reduced backward.
    """
    return _CopyToModelParallelRegion.apply(x, group)
