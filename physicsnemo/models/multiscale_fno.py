# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-Scale Fourier Neural Operator with cross-scale attention.

Standard FNO treats all retained Fourier modes equally via a single set of
learnable weights.  This architecture decomposes the spectrum into distinct
frequency *bands* and assigns a dedicated spectral operator to each:

    Low-frequency head  → large-scale structures  (pressure gradients, slow trends)
    Mid-frequency head  → intermediate features   (fronts, eddies, diffusion fronts)
    High-frequency head → fine-scale detail       (sharp boundaries, thin layers)

After each layer, a learned *cross-scale attention* mechanism allows the bands
to interact.  Coarse-scale context can guide where fine details matter; sharp
edges can sharpen coarse predictions.  This directly addresses the known FNO
weakness of under-representing high-frequency features.

Architecture per layer
----------------------
    input x  ──┬── BandSpectralConv (low)  ──┐
               ├── BandSpectralConv (mid)  ──┤ CrossScaleAttention → aggregate → x'
               └── BandSpectralConv (high) ──┘

The cross-scale attention is a lightweight gated pooling — O(C²) parameters,
not O(H·W·C²) — so it scales to large spatial grids.
"""

from typing import List

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


class BandSpectralConv2d(nn.Module):
    """Spectral convolution restricted to a wavenumber band [k_min, k_max).

    Unlike ``SpectralConv2d`` (which always takes the *lowest* N modes),
    this layer operates on a specific band in wavenumber magnitude space so
    that each MultiScaleFNO head truly specialises on a distinct frequency range.

    Parameters
    ----------
    in_channels, out_channels : int
    k_min, k_max : float
        Inclusive lower / exclusive upper bound on the wavenumber magnitude.
    max_modes : int
        Maximum number of modes kept per axis (caps memory usage).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_min: float,
        k_max: float,
        max_modes: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_min = k_min
        self.k_max = k_max
        self.max_modes = max_modes

        scale = 1.0 / (in_channels * out_channels)
        m = max_modes
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, m, m, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, m, m, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        kx = torch.fft.fftfreq(H, device=x.device).reshape(H, 1)
        ky = torch.fft.rfftfreq(W, device=x.device).reshape(1, W // 2 + 1)
        k_mag = (kx ** 2 + ky ** 2).sqrt()

        k_max_val = self.k_max if self.k_max < float("inf") else k_mag.max().item() + 1.0
        band_mask = (k_mag >= self.k_min) & (k_mag < k_max_val)  # (H, W//2+1)

        m = self.max_modes
        out_ft = torch.zeros_like(x_ft)

        # Positive-kx quadrant
        r1_mask = band_mask[:m, :m].unsqueeze(0).unsqueeze(0)  # (1,1,m,m)
        region1 = x_ft[:, :, :m, :m] * r1_mask
        out_ft[:, :, :m, :m] = out_ft[:, :, :m, :m] + torch.einsum(
            "bixy,ioxy->boxy", region1, self.weights1
        )

        # Negative-kx quadrant
        r2_mask = band_mask[-m:, :m].unsqueeze(0).unsqueeze(0)
        region2 = x_ft[:, :, -m:, :m] * r2_mask
        out_ft[:, :, -m:, :m] = out_ft[:, :, -m:, :m] + torch.einsum(
            "bixy,ioxy->boxy", region2, self.weights2
        )

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class CrossScaleAttention(nn.Module):
    """Learned cross-scale gating that lets frequency bands exchange information.

    Computes a single gating scalar from the pooled cross-scale query/key dot
    product and uses it to modulate the value tensor.  This is intentionally
    lightweight — the novelty is in the *multi-band specialisation*, not in
    the attention mechanism itself.

    Parameters
    ----------
    channels : int
        Feature channels per scale.
    n_scales : int
        Number of frequency scales.
    """

    def __init__(self, channels: int, n_scales: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.n_scales = n_scales
        total = channels * n_scales

        self.q_proj = nn.Conv2d(total, channels, 1)
        self.k_proj = nn.Conv2d(total, channels, 1)
        self.v_proj = nn.Conv2d(total, total, 1)
        self.out_proj = nn.Conv2d(total, total, 1)
        self.norm = nn.GroupNorm(min(8, channels), total)
        self.scale = channels ** -0.5

    def forward(self, scale_feats: List[torch.Tensor]) -> List[torch.Tensor]:
        combined = torch.cat(scale_feats, dim=1)  # (B, C*n_scales, H, W)

        q = self.q_proj(combined)  # (B, C, H, W)
        k = self.k_proj(combined)
        v = self.v_proj(combined)  # (B, C*n_scales, H, W)

        # Global pooled attention gate — O(C²) cost
        q_pool = q.mean(dim=[-2, -1], keepdim=True)
        k_pool = k.mean(dim=[-2, -1], keepdim=True)
        gate = torch.sigmoid((q_pool * k_pool).sum(dim=1, keepdim=True) * self.scale)

        out = self.out_proj(v * gate)
        out = self.norm(out + combined)  # residual
        return list(out.chunk(self.n_scales, dim=1))


class MultiScaleFNOBlock(nn.Module):
    """One layer of MultiScaleFNO: parallel band convolutions + cross-scale attention."""

    def __init__(
        self,
        channels: int,
        n_scales: int,
        band_edges: List[float],
        max_modes: int,
    ) -> None:
        super().__init__()
        self.n_scales = n_scales

        self.band_convs = nn.ModuleList(
            [
                BandSpectralConv2d(
                    channels, channels, band_edges[i], band_edges[i + 1], max_modes
                )
                for i in range(n_scales)
            ]
        )
        self.bypasses = nn.ModuleList(
            [nn.Conv2d(channels, channels, 1) for _ in range(n_scales)]
        )
        self.cross_attn = CrossScaleAttention(channels, n_scales)
        self.norms = nn.ModuleList(
            [nn.InstanceNorm2d(channels) for _ in range(n_scales)]
        )
        self.act = nn.GELU()
        self.aggregate = nn.Conv2d(channels * n_scales, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each frequency band processes the same input independently
        scale_feats = [
            self.act(self.norms[i](self.band_convs[i](x) + self.bypasses[i](x)))
            for i in range(self.n_scales)
        ]

        # Cross-scale attention: let scales inform each other
        scale_feats = self.cross_attn(scale_feats)

        # Aggregate all bands back into a single feature map (with residual)
        combined = torch.cat(scale_feats, dim=1)
        return self.aggregate(combined) + x


class MultiScaleFNO(Module):
    """Multi-Scale Fourier Neural Operator with cross-scale attention.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    hidden_channels : int
    n_layers : int
    n_scales : int
        Number of frequency bands.  Default 3 (low / mid / high).
    max_modes : int
        Fourier modes kept per band per axis.
    """

    _meta = ModelMetaData(
        name="MultiScaleFNO",
        nvp_tags=["pde", "operator-learning", "fourier", "multi-scale", "attention"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        n_scales: int = 3,
        max_modes: int = 16,
    ) -> None:
        super().__init__(meta=self._meta)
        self.n_scales = n_scales

        # Evenly partition [0, 0.5] into n_scales bands; last band is open-ended
        edges = [i / n_scales * 0.5 for i in range(n_scales + 1)]
        edges[-1] = float("inf")

        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                MultiScaleFNOBlock(hidden_channels, n_scales, edges, max_modes)
                for _ in range(n_layers)
            ]
        )
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
            n_scales=n_scales,
            max_modes=max_modes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
