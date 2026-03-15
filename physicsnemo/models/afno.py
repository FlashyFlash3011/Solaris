# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adaptive Fourier Neural Operator (AFNO) — Guibas et al., 2022."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


class AFNOMixingBlock(nn.Module):
    """Frequency-domain token mixing block used in AFNO."""

    def __init__(self, hidden_size: int, num_blocks: int = 8, sparsity_threshold: float = 0.01) -> None:
        super().__init__()
        assert hidden_size % num_blocks == 0, "hidden_size must be divisible by num_blocks"
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.sparsity_threshold = sparsity_threshold

        self.w1 = nn.Parameter(torch.empty(2, num_blocks, self.block_size, self.block_size))
        self.w2 = nn.Parameter(torch.empty(2, num_blocks, self.block_size, self.block_size))
        self.b1 = nn.Parameter(torch.zeros(2, num_blocks, self.block_size))
        self.b2 = nn.Parameter(torch.zeros(2, num_blocks, self.block_size))
        nn.init.xavier_uniform_(self.w1[0])
        nn.init.xavier_uniform_(self.w1[1])
        nn.init.xavier_uniform_(self.w2[0])
        nn.init.xavier_uniform_(self.w2[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) for 2-D input
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")  # (B, H, W//2+1, C)
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = (
            torch.einsum("...bi,bio->...bo", x.real, self.w1[0]) -
            torch.einsum("...bi,bio->...bo", x.imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag = (
            torch.einsum("...bi,bio->...bo", x.imag, self.w1[0]) +
            torch.einsum("...bi,bio->...bo", x.real, self.w1[1]) +
            self.b1[1]
        )
        o1 = F.softshrink(torch.stack([o1_real, o1_imag], dim=-1), lambd=self.sparsity_threshold)
        o1 = torch.view_as_complex(o1)

        o2_real = (
            torch.einsum("...bi,bio->...bo", o1.real, self.w2[0]) -
            torch.einsum("...bi,bio->...bo", o1.imag, self.w2[1]) +
            self.b2[0]
        )
        o2_imag = (
            torch.einsum("...bi,bio->...bo", o1.imag, self.w2[0]) +
            torch.einsum("...bi,bio->...bo", o1.real, self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        return x.to(dtype) + bias


class AFNOBlock(nn.Module):
    """Single AFNO transformer block (mixing + MLP)."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, num_blocks: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.filter = AFNOMixingBlock(hidden_size, num_blocks)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.filter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AFNO(Module):
    """Adaptive Fourier Neural Operator for 2-D spatial fields.

    Expects input of shape ``(batch, channels, height, width)`` and returns
    output of shape ``(batch, out_channels, height, width)``.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    img_size : tuple[int, int]
        Spatial resolution (H, W).
    patch_size : int
        Patch size for initial embedding.
    hidden_size : int
        Token embedding dimension.
    n_layers : int
        Number of AFNO blocks.
    mlp_ratio : float
        MLP expansion factor.
    num_blocks : int
        Number of frequency-domain mixing blocks.
    """

    _meta = ModelMetaData(
        name="AFNO",
        nvp_tags=["weather", "climate", "fourier", "transformer"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int] = (64, 64),
        patch_size: int = 4,
        hidden_size: int = 256,
        n_layers: int = 4,
        mlp_ratio: float = 4.0,
        num_blocks: int = 8,
    ) -> None:
        super().__init__(meta=self._meta)
        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0
        self.patch_size = patch_size
        self.img_size = img_size
        nh, nw = H // patch_size, W // patch_size
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, nh * nw, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([AFNOBlock(hidden_size, mlp_ratio, num_blocks) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.ConvTranspose2d(hidden_size, out_channels, patch_size, stride=patch_size)
        self._capture_init_args(
            in_channels=in_channels, out_channels=out_channels, img_size=img_size,
            patch_size=patch_size, hidden_size=hidden_size, n_layers=n_layers,
            mlp_ratio=mlp_ratio, num_blocks=num_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.patch_embed(x)         # (B, hidden, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, hidden)
        x = x + self.pos_embed
        nh, nw = H // self.patch_size, W // self.patch_size
        x = x.reshape(B, nh, nw, -1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.reshape(B, nh, nw, -1).permute(0, 3, 1, 2)  # (B, hidden, H/p, W/p)
        return self.head(x)
