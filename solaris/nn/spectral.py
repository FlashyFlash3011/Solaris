# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Spectral convolution layers (core building block of Fourier Neural Operators).

All layers:

* clamp the requested modes to what the input grid can represent, so a model
  built with ``modes=12`` still runs on a coarse 16×16 grid;
* only use the *active* slice of their weights (``modes_x`` etc. may be lowered
  after construction, e.g. by :meth:`solaris.models.FNO.set_modes`);
* compute the FFT in fp32 even under ``torch.autocast`` — cuFFT/hipFFT only
  support fp16 for power-of-two sizes and CPU FFT has no fp16 kernel.
"""

import torch
import torch.distributed as dist
import torch.nn as nn


def _fft_input(x: torch.Tensor) -> torch.Tensor:
    """Promote half-precision inputs to float32 (keep float64 as is)."""
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _resolve_tp_group(layer: nn.Module) -> dist.ProcessGroup | None:
    if layer._tp_group is not None:
        return layer._tp_group
    try:
        from solaris.distributed import DistributedManager

        return DistributedManager().get_group("tensor_parallel")
    except (KeyError, RuntimeError):
        return None


def _is_multi_rank(group: dist.ProcessGroup | None) -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


class SpectralConv1d(nn.Module):
    """1-D Fourier-space convolution layer.

    Multiplies the *modes_x* lowest Fourier modes by a learnable complex weight
    tensor, then transforms back to the physical domain.

    Parameters
    ----------
    in_channels, out_channels : int
        Number of input / output channels.
    modes_x : int
        Number of Fourier modes to keep along the spatial axis.
    """

    def __init__(self, in_channels: int, out_channels: int, modes_x: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, dtype=torch.cfloat)
        )

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, nx = x.shape
        mx = min(self.modes_x, nx // 2 + 1)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfft(_fft_input(x), norm="ortho")
            out_ft = torch.zeros(
                bsz, self.out_channels, nx // 2 + 1, dtype=x_ft.dtype, device=x.device
            )
            w = self.weights[:, :, :mx].to(x_ft.dtype)
            out_ft[:, :, :mx] = self._mul(x_ft[:, :, :mx], w)
            out = torch.fft.irfft(out_ft, n=nx, norm="ortho")
        return out.to(x.dtype)


class SpectralConv2d(nn.Module):
    """2-D Fourier-space convolution layer.

    Parameters
    ----------
    in_channels, out_channels : int
    modes_x, modes_y : int
        Fourier modes to retain per spatial dimension.
    distributed : bool
        When ``True`` and a tensor-parallel group with more than one rank is
        active, the input is expected to be *row-sharded* — each rank holds
        ``H / world_size`` rows of a ``(B, C, H, W)`` field — and the FFT is
        computed with :func:`~solaris.distributed.fft.distributed_rfft2`.  The
        output is row-sharded the same way.  With a single rank this falls
        back to the local computation.
    tp_group : dist.ProcessGroup, optional
        Explicit tensor-parallel process group.  If ``None`` and
        ``distributed=True``, the group is fetched from
        :class:`~solaris.distributed.DistributedManager` at forward time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        distributed: bool = False,
        tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.distributed = distributed
        self._tp_group = tp_group
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat)
        )

    def _tp_group_resolved(self) -> dist.ProcessGroup | None:
        return _resolve_tp_group(self) if self.distributed else None

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distributed:
            group = self._tp_group_resolved()
            if _is_multi_rank(group):
                return self._forward_distributed(x, group)

        bsz, _, nx, ny = x.shape
        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2 + 1)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfft2(_fft_input(x), norm="ortho")
            out_ft = torch.zeros(
                bsz, self.out_channels, nx, ny // 2 + 1, dtype=x_ft.dtype, device=x.device
            )
            if mx > 0:  # a 1-row grid has no modes to mix ([-0:] would select everything)
                w1 = self.weights1[:, :, :mx, :my].to(x_ft.dtype)
                w2 = self.weights2[:, :, :mx, :my].to(x_ft.dtype)
                out_ft[:, :, :mx, :my] = self._mul(x_ft[:, :, :mx, :my], w1)
                out_ft[:, :, -mx:, :my] = self._mul(x_ft[:, :, -mx:, :my], w2)
            out = torch.fft.irfft2(out_ft, s=(nx, ny), norm="ortho")
        return out.to(x.dtype)

    def _forward_distributed(self, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        from solaris.distributed.fft import distributed_irfft2, distributed_rfft2

        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        bsz, _, nx_local, ny = x.shape
        nx = nx_local * world
        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2 + 1)

        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            # (B, C, H, W_freq_local): full rows, this rank's slab of frequency columns
            x_ft = distributed_rfft2(_fft_input(x), group=group, norm="ortho")
            w_local = x_ft.shape[-1]
            out_ft = torch.zeros(
                bsz, self.out_channels, nx, w_local, dtype=x_ft.dtype, device=x.device
            )
            # Global frequency columns owned by this rank that fall inside [0, my)
            col0 = rank * w_local
            n_cols = max(0, min(my, col0 + w_local) - col0)
            if n_cols > 0 and mx > 0:
                w1 = self.weights1[:, :, :mx, col0 : col0 + n_cols].to(x_ft.dtype)
                w2 = self.weights2[:, :, :mx, col0 : col0 + n_cols].to(x_ft.dtype)
                out_ft[:, :, :mx, :n_cols] = self._mul(x_ft[:, :, :mx, :n_cols], w1)
                out_ft[:, :, -mx:, :n_cols] = self._mul(x_ft[:, :, -mx:, :n_cols], w2)
            else:
                # This rank owns no retained modes.  Keep the output attached to
                # the input graph so its backward still joins the AllToAll that
                # the other ranks are waiting on.
                out_ft = out_ft + 0 * x_ft.sum()
            out = distributed_irfft2(out_ft, output_size=(nx, ny), group=group, norm="ortho")
        return out.to(x.dtype)


class SpectralConv3d(nn.Module):
    """3-D Fourier-space convolution layer.

    Parameters
    ----------
    in_channels, out_channels : int
    modes_x, modes_y, modes_z : int
    distributed : bool
        Reserved for tensor-parallel execution.  Multi-rank 3-D spectral
        convolution is not implemented yet: with more than one rank in the
        tensor-parallel group the forward pass raises ``NotImplementedError``;
        with a single rank it falls back to the local computation.
    tp_group : dist.ProcessGroup, optional
        Explicit tensor-parallel group.  Auto-resolved from
        :class:`~solaris.distributed.DistributedManager` when ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
        modes_z: int,
        distributed: bool = False,
        tp_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.distributed = distributed
        self._tp_group = tp_group
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_x, modes_y, modes_z)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    def _tp_group_resolved(self) -> dist.ProcessGroup | None:
        return _resolve_tp_group(self) if self.distributed else None

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.distributed and _is_multi_rank(self._tp_group_resolved()):
            raise NotImplementedError(
                "Multi-rank SpectralConv3d is not implemented; use distributed=False "
                "or a tensor-parallel group of size 1."
            )

        bsz, _, nx, ny, nz = x.shape
        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2)
        mz = min(self.modes_z, nz // 2 + 1)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_ft = torch.fft.rfftn(_fft_input(x), dim=[-3, -2, -1], norm="ortho")
            out_ft = torch.zeros(
                bsz, self.out_channels, nx, ny, nz // 2 + 1, dtype=x_ft.dtype, device=x.device
            )

            def w(p: torch.Tensor) -> torch.Tensor:
                return p[:, :, :mx, :my, :mz].to(x_ft.dtype)

            if mx > 0 and my > 0:  # [-0:] would select the whole axis
                out_ft[:, :, :mx, :my, :mz] = self._mul(x_ft[:, :, :mx, :my, :mz], w(self.weights1))
                out_ft[:, :, -mx:, :my, :mz] = self._mul(
                    x_ft[:, :, -mx:, :my, :mz], w(self.weights2)
                )
                out_ft[:, :, :mx, -my:, :mz] = self._mul(
                    x_ft[:, :, :mx, -my:, :mz], w(self.weights3)
                )
                out_ft[:, :, -mx:, -my:, :mz] = self._mul(
                    x_ft[:, :, -mx:, -my:, :mz], w(self.weights4)
                )
            out = torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[-3, -2, -1], norm="ortho")
        return out.to(x.dtype)
