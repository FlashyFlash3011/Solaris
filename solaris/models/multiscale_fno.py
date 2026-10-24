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

import torch
import torch.nn as nn

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module
from solaris.nn.norm import UNSET, NormArg, make_norm, resolve_norm
from solaris.nn.spectral import _pad_to, _stack_signed, complex_einsum


class BandSpectralConv2d(nn.Module):
    """Spectral convolution restricted to a band of Fourier modes.

    The layer keeps the usual FNO block of the lowest ``max_modes`` modes per
    axis (both signs of k_x), and of those only uses the modes whose *mode
    index radius* ``r = sqrt(i_x² + i_y²)`` lies in ``[k_min, k_max)``.

    Bands are defined in mode-index units (the n-th Fourier mode), not as a
    fraction of the grid's Nyquist frequency, so a band selects the same
    physical scales — and the same weights — at every resolution.

    Parameters
    ----------
    in_channels, out_channels : int
    k_min, k_max : float
        Inclusive lower / exclusive upper bound on the mode-index radius.
        ``k_max = inf`` makes the band open-ended (it takes the corners of the
        retained block).
    max_modes : int
        Modes kept per axis (the same block a ``SpectralConv2d`` would keep).
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

    def _block(self, H: int, W: int) -> tuple[int, int]:
        """Size of the retained block, clamped to what the grid can represent."""
        return min(self.max_modes, H // 2), min(self.max_modes, W // 2 + 1)

    def band_mask(self, H: int, W: int, device: torch.device | None = None) -> torch.Tensor:
        """Boolean mask over the ``(H, W//2+1)`` rfft2 spectrum of this band's modes."""
        ix = torch.fft.fftfreq(H, d=1.0 / H, device=device).reshape(H, 1)  # signed index
        iy = torch.arange(W // 2 + 1, device=device, dtype=ix.dtype).reshape(1, -1)
        r = (ix**2 + iy**2).sqrt()
        in_band = (r >= self.k_min) & (r < self.k_max)

        mx, my = self._block(H, W)
        retained = torch.zeros(H, W // 2 + 1, dtype=torch.bool, device=device)
        if mx > 0:
            retained[:mx, :my] = True
            retained[-mx:, :my] = True
        return in_band & retained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp16 FFTs are unsupported on CPU and for non-power-of-2 GPU sizes
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
            return self._forward_fp32(xf).to(x.dtype)

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        mx, my = self._block(H, W)
        if mx == 0:
            out_ft = x_ft.new_zeros(B, self.out_channels, H, W // 2 + 1)
            return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")

        mask = self.band_mask(H, W, device=x.device).to(x_ft.real.dtype)
        w1 = self.weights1[:, :, :mx, :my]
        w2 = self.weights2[:, :, :mx, :my]

        # Positive-k_x rows, then negative-k_x rows of the retained block
        low = complex_einsum("bixy,ioxy->boxy", x_ft[:, :, :mx, :my] * mask[:mx, :my], w1)
        high = complex_einsum("bixy,ioxy->boxy", x_ft[:, :, -mx:, :my] * mask[-mx:, :my], w2)
        out_ft = _pad_to(_stack_signed(low, high, -2, H), -1, W // 2 + 1)
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
        self.scale = channels**-0.5

    def forward(self, scale_feats: list[torch.Tensor]) -> list[torch.Tensor]:
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
        band_edges: list[float],
        max_modes: int,
        norm: str | None = "instance",
    ) -> None:
        super().__init__()
        self.n_scales = n_scales

        self.band_convs = nn.ModuleList(
            [
                BandSpectralConv2d(channels, channels, band_edges[i], band_edges[i + 1], max_modes)
                for i in range(n_scales)
            ]
        )
        self.bypasses = nn.ModuleList([nn.Conv2d(channels, channels, 1) for _ in range(n_scales)])
        self.cross_attn = CrossScaleAttention(channels, n_scales)
        self.norms = nn.ModuleList([make_norm(norm, channels, 2) for _ in range(n_scales)])
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
        Fourier modes kept per axis.  The retained block is split into
        ``n_scales`` rings of equal width in mode-index radius, so each band
        covers the same physical scales at every resolution.
    norm : None or "instance"
        Normalisation inside the blocks; see :class:`~solaris.models.FNO`.
    """

    _meta = ModelMetaData(
        name="MultiScaleFNO",
        tags=["pde", "operator-learning", "fourier", "multi-scale", "attention"],
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
        norm: NormArg = UNSET,
    ) -> None:
        super().__init__(meta=self._meta)
        norm = resolve_norm(norm, "MultiScaleFNO")
        self.n_scales = n_scales

        # Equal-width rings in mode-index radius over [0, max_modes); the last
        # band is open-ended so it also takes the corners of the retained block.
        edges = [i * max_modes / n_scales for i in range(n_scales + 1)]
        edges[-1] = float("inf")

        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [
                MultiScaleFNOBlock(hidden_channels, n_scales, edges, max_modes, norm)
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
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
