# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wavelet Neural Operator (WNO) — Tripura & Chakraborty, 2022."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module


class WaveletConv2d(nn.Module):
    """2-D Haar wavelet convolution layer.

    Decomposes the input into LL/LH/HL/HH subbands using stride-2 convolutions
    with Haar filters, applies learnable weights per subband, and reconstructs
    via the inverse DWT.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    levels : int
        Number of DWT decomposition levels (applied only to LL at each step).
    """

    def __init__(self, in_channels: int, out_channels: int, levels: int = 2) -> None:
        super().__init__()
        self.levels = levels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Haar 1-D filters
        lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)   # (2,)
        hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)  # (2,)

        # Build 2-D separable analysis kernels via outer product → (1, 1, 2, 2) each
        ll = torch.outer(lo, lo).reshape(1, 1, 2, 2)
        lh = torch.outer(hi, lo).reshape(1, 1, 2, 2)
        hl = torch.outer(lo, hi).reshape(1, 1, 2, 2)
        hh = torch.outer(hi, hi).reshape(1, 1, 2, 2)
        self.register_buffer("analysis_filters", torch.cat([ll, lh, hl, hh], dim=0))  # (4,1,2,2)

        # Learnable mixing weights for the 4 subbands
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(out_channels, in_channels, 1, 1)) for _ in range(4)
        ])
        for w in self.weights:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        self.bypass = nn.Conv2d(in_channels, out_channels, 1)

    def _dwt(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Single-level 2-D Haar DWT. Returns (LL, LH, HL, HH)."""
        B, C, H, W = x.shape
        # Group-wise convolution: apply each of the 4 filters to all channels
        # Reshape to (B*C, 1, H, W)
        xr = x.reshape(B * C, 1, H, W)
        f = self.analysis_filters  # (4, 1, 2, 2)
        out = F.conv2d(xr, f, stride=2, padding=0)  # (B*C, 4, H//2, W//2)
        out = out.reshape(B, C, 4, H // 2, W // 2)
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def _idwt(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """Single-level 2-D Haar iDWT."""
        B, C, H, W = ll.shape
        # Stack subbands (B, C*4, H, W) and upsample via transpose conv
        stacked = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, H, W)
        f = self.analysis_filters  # reuse (orthonormal → analysis == synthesis)
        out = F.conv_transpose2d(stacked, f, stride=2, padding=0)  # (B*C, 1, 2H, 2W)
        return out.reshape(B, C, H * 2, W * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        bypass = self.bypass(x)

        # Multi-level DWT
        ll = x
        detail_stack: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(self.levels):
            ll, lh, hl, hh = self._dwt(ll)
            detail_stack.append((lh, hl, hh))

        # Apply learnable weights to deepest LL subband
        ll_out = F.conv2d(ll, self.weights[0].expand(-1, -1, 1, 1))

        # Inverse DWT
        for lh, hl, hh in reversed(detail_stack):
            lh_out = F.conv2d(lh, self.weights[1].expand(-1, -1, 1, 1))
            hl_out = F.conv2d(hl, self.weights[2].expand(-1, -1, 1, 1))
            hh_out = F.conv2d(hh, self.weights[3].expand(-1, -1, 1, 1))
            ll_out = self._idwt(ll_out, lh_out, hl_out, hh_out)

        # Trim/pad to original size (DWT requires even dims)
        ll_out = ll_out[..., :H, :W]
        return ll_out + bypass


class WNO(Module):
    """Wavelet Neural Operator.

    Replaces the Fourier transform of FNO with a 2-D Haar discrete wavelet
    transform (DWT).  Wavelet bases are better suited to solutions with
    local discontinuities or sharp fronts.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    hidden_channels : int
    n_layers : int
    levels : int
        DWT decomposition levels per WNO layer.
    padding : int
        Zero-padding added before the WNO blocks to handle non-power-of-2
        spatial sizes, removed afterwards.
    """

    _meta = ModelMetaData(
        name="WNO",
        nvp_tags=["pde", "operator-learning", "wavelet"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        levels: int = 2,
        padding: int = 8,
    ) -> None:
        super().__init__(meta=self._meta)
        self.padding = padding
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                WaveletConv2d(hidden_channels, hidden_channels, levels=levels),
                nn.InstanceNorm2d(hidden_channels),
                nn.GELU(),
            )
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, out_channels, 1),
        )
        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            levels=levels,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)

        Returns
        -------
        torch.Tensor  shape (B, out_channels, H, W)
        """
        H, W = x.shape[-2], x.shape[-1]
        x = self.lift(x)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        for block in self.blocks:
            x = block(x)
        if self.padding > 0:
            x = x[..., :H, :W]
        return self.proj(x)
