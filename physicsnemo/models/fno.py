# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fourier Neural Operator (FNO) — Li et al., 2021."""

from typing import List, Optional

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.spectral import SpectralConv1d, SpectralConv2d, SpectralConv3d


class FNOBlock1d(nn.Module):
    def __init__(self, channels: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(channels, channels, modes)
        self.bypass = nn.Conv1d(channels, channels, 1)
        self.norm = nn.InstanceNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class FNOBlock2d(nn.Module):
    def __init__(self, channels: int, modes_x: int, modes_y: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_x, modes_y)
        self.bypass = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class FNOBlock3d(nn.Module):
    def __init__(self, channels: int, modes_x: int, modes_y: int, modes_z: int) -> None:
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes_x, modes_y, modes_z)
        self.bypass = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spectral(x) + self.bypass(x)))


class FNO(Module):
    """Fourier Neural Operator supporting 1-D, 2-D, and 3-D inputs.

    Parameters
    ----------
    in_channels : int
        Number of input channels / variables.
    out_channels : int
        Number of output channels / variables.
    hidden_channels : int
        Internal channel width.
    n_layers : int
        Number of Fourier blocks.
    modes : int or list[int]
        Fourier modes per spatial dimension. A single int is broadcast.
    dim : int
        Spatial dimensionality: 1, 2, or 3.
    """

    _meta = ModelMetaData(
        name="FNO",
        nvp_tags=["pde", "operator-learning", "fourier"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int | List[int] = 12,
        dim: int = 2,
    ) -> None:
        super().__init__(meta=self._meta)
        assert dim in (1, 2, 3), "dim must be 1, 2, or 3"
        if isinstance(modes, int):
            modes = [modes] * dim
        self.dim = dim
        self.lift = nn.Conv1d(in_channels, hidden_channels, 1) if dim == 1 else (
            nn.Conv2d(in_channels, hidden_channels, 1) if dim == 2 else
            nn.Conv3d(in_channels, hidden_channels, 1)
        )
        if dim == 1:
            self.blocks = nn.ModuleList([FNOBlock1d(hidden_channels, modes[0]) for _ in range(n_layers)])
        elif dim == 2:
            self.blocks = nn.ModuleList([FNOBlock2d(hidden_channels, modes[0], modes[1]) for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([FNOBlock3d(hidden_channels, modes[0], modes[1], modes[2]) for _ in range(n_layers)])
        conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dim]
        self.proj = nn.Sequential(
            conv_cls(hidden_channels, hidden_channels * 4, 1),
            nn.GELU(),
            conv_cls(hidden_channels * 4, out_channels, 1),
        )
        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            dim=dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
