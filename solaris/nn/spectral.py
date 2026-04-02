# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Spectral convolution layers (core building block of Fourier Neural Operators)."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        out_ft = torch.zeros(bsz, self.out_channels, nx // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, : self.modes_x] = self._mul(x_ft[:, :, : self.modes_x], self.weights)
        return torch.fft.irfft(out_ft, n=nx, norm="ortho")


class SpectralConv2d(nn.Module):
    """2-D Fourier-space convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.in_channels = in_channels
        self.out_channels = out_channels
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, dtype=torch.cfloat))

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, nx, ny = x.shape
        mx, my = self.modes_x, self.modes_y
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(bsz, self.out_channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :my] = self._mul(x_ft[:, :, :mx, :my], self.weights1[:, :, :mx, :my])
        out_ft[:, :, -mx:, :my] = self._mul(x_ft[:, :, -mx:, :my], self.weights2[:, :, :mx, :my])
        return torch.fft.irfft2(out_ft, s=(nx, ny), norm="ortho")


class SpectralConv3d(nn.Module):
    """3-D Fourier-space convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, modes_x: int, modes_y: int, modes_z: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))

    def _mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, nx, ny, nz = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="ortho")
        out_ft = torch.zeros(bsz, self.out_channels, nx, ny, nz // 2 + 1, dtype=torch.cfloat, device=x.device)
        mx, my, mz = self.modes_x, self.modes_y, self.modes_z
        out_ft[:, :, :mx, :my, :mz] = self._mul(x_ft[:, :, :mx, :my, :mz], self.weights1)
        out_ft[:, :, -mx:, :my, :mz] = self._mul(x_ft[:, :, -mx:, :my, :mz], self.weights2)
        out_ft[:, :, :mx, -my:, :mz] = self._mul(x_ft[:, :, :mx, -my:, :mz], self.weights3)
        out_ft[:, :, -mx:, -my:, :mz] = self._mul(x_ft[:, :, -mx:, -my:, :mz], self.weights4)
        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=[-3, -2, -1], norm="ortho")
