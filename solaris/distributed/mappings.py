# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel autograd primitives.

Four differentiable collective operations for sharding model parameters across
a tensor-parallel process group.  Each is a ``torch.autograd.Function`` whose
forward and backward passes together form a consistent gradient flow:

+----------------------------+---------------------+---------------------+
| Function                   | Forward             | Backward            |
+============================+=====================+=====================+
| scatter_to_model_parallel  | split along dim 0   | all-gather          |
| gather_from_model_parallel | all-gather          | split along dim 0   |
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

    # Scatter input across TP ranks before a column-parallel linear layer
    x_local = scatter_to_model_parallel_region(x, group=tp_group)
    y_local = local_linear(x_local)
    # Gather outputs back across TP ranks
    y = gather_from_model_parallel_region(y_local, group=tp_group)
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


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Forward: scatter along dim 0.  Backward: all-gather."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        world = _get_world_size(group)
        if world == 1:
            return x
        rank = _get_rank(group)
        chunks = x.chunk(world, dim=0)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx.group
        world = _get_world_size(group)
        if world == 1:
            return grad, None
        gathered = [torch.empty_like(grad) for _ in range(world)]
        dist.all_gather(gathered, grad.contiguous(), group=group)
        return torch.cat(gathered, dim=0), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Forward: all-gather along dim 0.  Backward: scatter."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
        ctx.group = group
        world = _get_world_size(group)
        if world == 1:
            return x
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x.contiguous(), group=group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx.group
        world = _get_world_size(group)
        if world == 1:
            return grad, None
        rank = _get_rank(group)
        chunks = grad.chunk(world, dim=0)
        return chunks[rank].contiguous(), None


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
) -> torch.Tensor:
    """Split *x* along dim 0 and keep only the local rank's slice.

    Use before a row-parallel weight multiply where each rank holds a column
    shard of the weight matrix.
    """
    return _ScatterToModelParallelRegion.apply(x, group)


def gather_from_model_parallel_region(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """All-gather *x* along dim 0 across the tensor-parallel group.

    Use after a column-parallel linear layer to reconstruct the full output.
    """
    return _GatherFromModelParallelRegion.apply(x, group)


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
