# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed 2-D real FFT primitives.

:func:`distributed_rfft2` and :func:`distributed_irfft2` split the 2-D FFT
across a tensor-parallel process group so that large spatial grids can exceed
single-GPU memory.

Algorithm
---------
The 2-D FFT is separable, so it is computed as two batches of 1-D FFTs with a
global transpose (AllToAll) in between:

Forward (``distributed_rfft2``)
    1.  Each rank holds a row-slab of *x*: shape ``(B, C, H/world, W)``.
    2.  Local rFFT along *W*.
    3.  AllToAll: each rank ends up with all *H* rows of a slab of frequency
        columns: shape ``(B, C, H, ceil((W/2+1)/world))``.
    4.  Local FFT along *H*.

Inverse (``distributed_irfft2``)
    Exact reverse: iFFT along *H* → AllToAll → irFFT along *W*.

Gradients
---------
The local FFTs are ordinary ``torch.fft`` calls, so PyTorch supplies their
(non-trivial) adjoints.  Only the AllToAll transposes are custom autograd
functions; the adjoint of a transpose is the reverse transpose.

Single-rank fallback
--------------------
When ``world_size == 1`` (or ``dist`` is not initialised) both functions are
exactly ``torch.fft.rfft2`` / ``torch.fft.irfft2``.

Requirements
------------
``H`` must be divisible by the world size.  The frequency axis is zero-padded
to a multiple of the world size, so the last rank's slab may contain padding.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _is_distributed(group: dist.ProcessGroup | None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def _all_to_all(chunks: list[torch.Tensor], group: dist.ProcessGroup | None) -> list[torch.Tensor]:
    """Send ``chunks[i]`` to rank *i*; return the chunk received from each rank.

    Uses ``all_to_all_single`` (supported by NCCL/RCCL *and* gloo) on a stacked
    real view, so complex spectra work on every backend.
    """
    send = (
        torch.view_as_real(torch.stack(chunks)) if chunks[0].is_complex() else torch.stack(chunks)
    )
    send = send.contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    if chunks[0].is_complex():
        recv = torch.view_as_complex(recv)
    return list(recv.unbind(0))


def _rows_to_cols(x: torch.Tensor, group: dist.ProcessGroup | None, world: int) -> torch.Tensor:
    """``(B, C, H/world, Wf)`` row-sharded → ``(B, C, H, Wf_pad/world)`` column-sharded."""
    w_freq = x.shape[-1]
    pad = (world - w_freq % world) % world
    if pad:
        x = F.pad(x, (0, pad))
    return torch.cat(_all_to_all(list(x.chunk(world, dim=-1)), group), dim=2)


def _cols_to_rows(
    x: torch.Tensor, group: dist.ProcessGroup | None, world: int, w_freq: int
) -> torch.Tensor:
    """``(B, C, H, Wf_pad/world)`` column-sharded → ``(B, C, H/world, Wf)`` row-sharded."""
    return torch.cat(_all_to_all(list(x.chunk(world, dim=2)), group), dim=-1)[..., :w_freq]


class _RowsToCols(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, world):
        ctx.group, ctx.world, ctx.w_freq = group, world, x.shape[-1]
        return _rows_to_cols(x, group, world)

    @staticmethod
    def backward(ctx, grad):
        return _cols_to_rows(grad, ctx.group, ctx.world, ctx.w_freq), None, None


class _ColsToRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, world, w_freq):
        ctx.group, ctx.world, ctx.w_local = group, world, x.shape[-1]
        return _cols_to_rows(x, group, world, w_freq)

    @staticmethod
    def backward(ctx, grad):
        g = _rows_to_cols(grad, ctx.group, ctx.world)
        return g[..., : ctx.w_local], None, None, None


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def distributed_rfft2(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    norm: str = "ortho",
) -> torch.Tensor:
    """Distributed ``rfft2`` over the last two dims.

    Input is row-sharded ``(B, C, H/world, W)``; output is column-sharded
    ``(B, C, H, ceil((W//2+1)/world))``.  Falls back to ``torch.fft.rfft2``
    when world_size == 1.
    """
    if not _is_distributed(group):
        return torch.fft.rfft2(x, norm=norm)
    world = dist.get_world_size(group)
    x_f = torch.fft.rfft(x, dim=-1, norm=norm)
    x_col = _RowsToCols.apply(x_f, group, world)
    return torch.fft.fft(x_col, dim=-2, norm=norm)


def distributed_irfft2(
    x: torch.Tensor,
    output_size: tuple[int, int],
    group: dist.ProcessGroup | None = None,
    norm: str = "ortho",
) -> torch.Tensor:
    """Distributed ``irfft2`` — inverse of :func:`distributed_rfft2`.

    Input is column-sharded ``(B, C, H, ceil((W//2+1)/world))``; output is
    row-sharded ``(B, C, H/world, W)``.  ``output_size`` is the *global*
    ``(H, W)``.  Falls back to ``torch.fft.irfft2`` when world_size == 1.
    """
    H, W = output_size
    if not _is_distributed(group):
        return torch.fft.irfft2(x, s=(H, W), norm=norm)
    world = dist.get_world_size(group)
    g = torch.fft.ifft(x, n=H, dim=-2, norm=norm)
    g = _ColsToRows.apply(g, group, world, W // 2 + 1)
    return torch.fft.irfft(g, n=W, dim=-1, norm=norm)
