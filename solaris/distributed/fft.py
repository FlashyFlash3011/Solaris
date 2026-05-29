# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Distributed 2-D real FFT primitives.

:class:`DistributedRFFT2` and :class:`DistributedIRFFT2` split the 2-D FFT
across a tensor-parallel process group so that large spatial grids can exceed
single-GPU memory.

Algorithm
---------
The 2-D FFT is separable: FFT2(x) = FFT_rows(FFT_cols(x)).  We exploit this
to distribute the work:

Forward  (DistributedRFFT2)
    1.  Each rank holds a row-slab of *x*: shape ``(B, C, H/W_tp, W)``.
    2.  Local rFFT along the *W* dimension (columns).
    3.  AllToAll transpose: each rank ends up with a column-slab of the
        frequency domain: shape ``(B, C, H, (W/2+1)/W_tp)``.
    4.  Local FFT along the *H* dimension (rows).

Backward (DistributedIRFFT2)
    Exact reverse of forward: iFFT rows → AllToAll → irFFT cols.

Single-rank fallback
--------------------
When ``world_size == 1`` (or ``dist`` is not initialised) both classes fall
back to ``torch.fft.rfft2`` / ``torch.fft.irfft2`` with no communication
overhead.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _is_distributed(group: dist.ProcessGroup | None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def _alltoall_rows_to_cols(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    world: int,
) -> torch.Tensor:
    """AllToAll: redistribute from row-sharded to column-sharded.

    Input  shape: ``(B, C, H/world, W_freq)``  — each rank has H/world rows.
    Output shape: ``(B, C, H,       W_freq/world)`` — each rank has W_freq/world cols.

    The tensor is padded along W_freq if needed so it is divisible by *world*.
    """
    B, C, H_local, W_freq = x.shape
    # Pad W_freq to be divisible by world
    pad = (world - W_freq % world) % world
    if pad:
        x = F.pad(x, (0, pad))
        W_freq_padded = W_freq + pad
    else:
        W_freq_padded = W_freq

    # Split along W_freq into `world` chunks, one per destination rank
    # Each chunk: (B, C, H_local, W_freq_padded//world)
    chunks = list(x.chunk(world, dim=-1))  # list of world tensors

    out_chunks = [torch.empty_like(c) for c in chunks]
    dist.all_to_all(out_chunks, chunks, group=group)

    # Concatenate along H to restore full H
    result = torch.cat(out_chunks, dim=2)  # (B, C, H, W_freq_padded//world)

    # Trim padding from W_freq dim
    w_local = (W_freq_padded) // world
    return result[:, :, :, :w_local]


def _alltoall_cols_to_rows(
    x: torch.Tensor,
    group: dist.ProcessGroup | None,
    world: int,
    H_local: int,
    W_freq_full: int,
) -> torch.Tensor:
    """AllToAll: redistribute from column-sharded back to row-sharded.

    Inverse of :func:`_alltoall_rows_to_cols`.
    """
    B, C, H, W_local = x.shape
    pad = (world - W_freq_full % world) % world
    W_freq_padded = W_freq_full + pad

    # Pad W_local if necessary (last rank may have slightly fewer cols)
    target_w_local = W_freq_padded // world
    if W_local < target_w_local:
        x = F.pad(x, (0, target_w_local - W_local))

    # Split along H into `world` chunks
    chunks = list(x.chunk(world, dim=2))
    out_chunks = [torch.empty_like(c) for c in chunks]
    dist.all_to_all(out_chunks, chunks, group=group)

    result = torch.cat(out_chunks, dim=-1)  # (B, C, H_local, W_freq_padded)
    return result[:, :, :, :W_freq_full]


class DistributedRFFT2(torch.autograd.Function):
    """Distributed 2-D real-to-complex FFT.

    Equivalent to ``torch.fft.rfft2(x, norm="ortho")`` but distributes the
    computation across the tensor-parallel process group.

    Parameters (passed via ``apply``)
    ------------------------------------
    x : torch.Tensor  shape ``(B, C, H, W)``
        Row-sharded input: each rank holds ``H / world_size`` rows.
    group : dist.ProcessGroup or None
    norm : str  ``"ortho"`` (default) or ``"forward"`` / ``"backward"``
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        group: dist.ProcessGroup | None,
        norm: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.norm = norm
        ctx.input_shape = x.shape

        if not _is_distributed(group):
            return torch.fft.rfft2(x, norm=norm)

        world = dist.get_world_size(group)
        B, C, H_local, W = x.shape
        ctx.H_local = H_local
        ctx.W = W

        # Step 1: local rFFT along columns (W dim)
        x_f = torch.fft.rfft(x, dim=-1, norm=norm)  # (B, C, H_local, W//2+1)
        W_freq = x_f.shape[-1]
        ctx.W_freq = W_freq

        # Step 2: AllToAll — row → column sharding
        x_f = x_f.contiguous()
        x_col = _alltoall_rows_to_cols(x_f, group, world)  # (B, C, H, W_freq//world)

        # Step 3: local FFT along rows (H dim)
        out = torch.fft.fft(x_col, dim=-2, norm=norm)  # (B, C, H, W_freq//world)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx.group
        norm = ctx.norm

        if not _is_distributed(group):
            # grad is complex; use irfft2 to match forward
            B, C, H, W = ctx.input_shape
            return torch.fft.irfft2(grad, s=(H, W), norm=norm), None, None

        world = dist.get_world_size(group)
        H_local = ctx.H_local
        W_freq = ctx.W_freq

        # Reverse step 3: iFFT along rows
        g = torch.fft.ifft(grad, dim=-2, norm=norm)

        # Reverse step 2: AllToAll — column → row sharding
        g = _alltoall_cols_to_rows(g.contiguous(), group, world, H_local, W_freq)

        # Reverse step 1: irFFT along columns
        g_out = torch.fft.irfft(g, n=ctx.W, dim=-1, norm=norm)
        return g_out, None, None


class DistributedIRFFT2(torch.autograd.Function):
    """Distributed 2-D complex-to-real inverse FFT.

    Equivalent to ``torch.fft.irfft2(x, s=(H, W), norm="ortho")`` but
    distributed across the tensor-parallel group.

    The input is column-sharded (output of :class:`DistributedRFFT2`).
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        output_size: tuple[int, int],
        group: dist.ProcessGroup | None,
        norm: str,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.norm = norm
        ctx.input_shape = x.shape
        H, W = output_size
        ctx.H = H
        ctx.W = W

        if not _is_distributed(group):
            return torch.fft.irfft2(x, s=(H, W), norm=norm)

        world = dist.get_world_size(group)
        W_freq = W // 2 + 1
        H_local = H // world

        # Step 1: local iFFT along rows
        g = torch.fft.ifft(x, n=H, dim=-2, norm=norm)  # (B, C, H, W_freq//world)

        # Step 2: AllToAll — column → row sharding
        g = _alltoall_cols_to_rows(g.contiguous(), group, world, H_local, W_freq)

        # Step 3: local irFFT along columns
        out = torch.fft.irfft(g, n=W, dim=-1, norm=norm)  # (B, C, H_local, W)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        group = ctx.group
        norm = ctx.norm

        if not _is_distributed(group):
            return torch.fft.rfft2(grad, norm=norm), None, None, None

        world = dist.get_world_size(group)
        W_freq = ctx.W // 2 + 1

        # Reverse step 3: rFFT along columns
        g = torch.fft.rfft(grad, dim=-1, norm=norm)

        # Reverse step 2: AllToAll
        g = _alltoall_rows_to_cols(g.contiguous(), group, world)

        # Reverse step 1: FFT along rows
        g_out = torch.fft.fft(g, dim=-2, norm=norm)
        return g_out[:, :, :, :W_freq], None, None, None


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------


def distributed_rfft2(
    x: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    norm: str = "ortho",
) -> torch.Tensor:
    """Distributed ``rfft2``.  Falls back to local rfft2 when world_size==1."""
    return DistributedRFFT2.apply(x, group, norm)


def distributed_irfft2(
    x: torch.Tensor,
    output_size: tuple[int, int],
    group: dist.ProcessGroup | None = None,
    norm: str = "ortho",
) -> torch.Tensor:
    """Distributed ``irfft2``.  Falls back to local irfft2 when world_size==1."""
    return DistributedIRFFT2.apply(x, output_size, group, norm)
