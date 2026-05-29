# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Spectral convolution layers (core building block of Fourier Neural Operators)."""

import torch
import torch.distributed as dist
import torch.nn as nn


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
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(
            bsz, self.out_channels, nx // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, : self.modes_x] = self._mul(x_ft[:, :, : self.modes_x], self.weights)
        return torch.fft.irfft(out_ft, n=nx, norm="ortho")


class SpectralConv2d(nn.Module):
    """2-D Fourier-space convolution layer.

    Parameters
    ----------
    in_channels, out_channels : int
    modes_x, modes_y : int
        Fourier modes to retain per spatial dimension.
    distributed : bool
        When ``True``, use :func:`~solaris.distributed.fft.distributed_rfft2`
        so the computation is sharded across the tensor-parallel process group.
        Requires :meth:`~solaris.distributed.DistributedManager.initialize_mesh`
        to have been called with a ``"tensor_parallel"`` dimension.
        Defaults to ``False`` (standard local FFT).
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
        if self._tp_group is not None:
            return self._tp_group
        if self.distributed:
            try:
                from solaris.distributed import DistributedManager

                return DistributedManager().get_group("tensor_parallel")
            except (KeyError, RuntimeError):
                return None
        return None

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, nx, ny = x.shape
        mx, my = self.modes_x, self.modes_y

        if self.distributed:
            from solaris.distributed.fft import distributed_irfft2, distributed_rfft2

            group = self._tp_group_resolved()
            x_ft = distributed_rfft2(x, group=group, norm="ortho")
            out_ft = torch.zeros(
                bsz, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device
            )
            out_ft[:, :, :mx, :my] = self._mul(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
            out_ft[:, :, -mx:, :my] = self._mul(
                x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my]
            )
            return distributed_irfft2(out_ft, output_size=(nx, ny), group=group, norm="ortho")

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            bsz, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :mx, :my] = self._mul(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
        out_ft[:, :, -mx:, :my] = self._mul(x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(nx, ny), norm="ortho")


class SpectralConv3d(nn.Module):
    """3-D Fourier-space convolution layer.

    Parameters
    ----------
    in_channels, out_channels : int
    modes_x, modes_y, modes_z : int
    distributed : bool
        When ``True``, the (x, y) plane FFT is distributed via
        :func:`~solaris.distributed.fft.distributed_rfft2`.  The z-dimension
        FFT remains local.  Defaults to ``False``.
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
        self.weights1 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale
            * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat)
        )

    def _tp_group_resolved(self) -> dist.ProcessGroup | None:
        if self._tp_group is not None:
            return self._tp_group
        if self.distributed:
            try:
                from solaris.distributed import DistributedManager

                return DistributedManager().get_group("tensor_parallel")
            except (KeyError, RuntimeError):
                return None
        return None

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, nx, ny, nz = x.shape
        mx, my, mz = self.modes_x, self.modes_y, self.modes_z

        if self.distributed:
            from solaris.distributed.fft import distributed_irfft2, distributed_rfft2

            group = self._tp_group_resolved()
            # Distribute the (x, y) plane FFT; z-dim stays local
            # Reshape: (B, C, nx, ny, nz) → (B*nz, C, nx, ny) for 2-D distributed FFT
            x_2d = x.permute(0, 1, 4, 2, 3).reshape(bsz * nz, -1, nx, ny)
            x_ft_2d = distributed_rfft2(x_2d, group=group, norm="ortho")
            x_ft_2d = x_ft_2d.reshape(bsz, -1, nz, nx, ny // 2 + 1).permute(0, 1, 3, 4, 2)
            # Local FFT along z
            x_ft = torch.fft.rfft(x_ft_2d, dim=-1, norm="ortho")

            out_ft = torch.zeros(
                bsz,
                self.out_channels,
                nx,
                ny // 2 + 1,
                nz // 2 + 1,
                dtype=torch.cfloat,
                device=x.device,
            )
            out_ft[:, :, :mx, :my, :mz] = self._mul(x_ft[:, :, :mx, :my, :mz], self.weights1)
            out_ft[:, :, -mx:, :my, :mz] = self._mul(x_ft[:, :, -mx:, :my, :mz], self.weights2)
            out_ft[:, :, :mx, -my:, :mz] = self._mul(x_ft[:, :, :mx, -my:, :mz], self.weights3)
            out_ft[:, :, -mx:, -my:, :mz] = self._mul(x_ft[:, :, -mx:, -my:, :mz], self.weights4)

            # Inverse: local irFFT along z, then distributed irfft2 over (x, y)
            out = torch.fft.irfft(out_ft, n=nz, dim=-1, norm="ortho")
            out_2d = out.permute(0, 1, 4, 2, 3).reshape(bsz * nz, -1, nx, ny // 2 + 1)
            out_2d = distributed_irfft2(out_2d, output_size=(nx, ny), group=group, norm="ortho")
            return out_2d.reshape(bsz, -1, nz, nx, ny).permute(0, 1, 3, 4, 2)

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        out_ft = torch.zeros(
            bsz, self.out_channels, nx, ny, nz // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :mx, :my, :mz] = self._mul(x_ft[:, :, :mx, :my, :mz], self.weights1)
        out_ft[:, :, -mx:, :my, :mz] = self._mul(x_ft[:, :, -mx:, :my, :mz], self.weights2)
        out_ft[:, :, :mx, -my:, :mz] = self._mul(x_ft[:, :, :mx, -my:, :mz], self.weights3)
        out_ft[:, :, -mx:, -my:, :mz] = self._mul(x_ft[:, :, -mx:, -my:, :mz], self.weights4)
        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[-3, -2, -1], norm="ortho")
