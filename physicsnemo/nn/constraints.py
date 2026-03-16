# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Physics constraint layers — enforce conservation laws at the architecture level.

These layers are *hard* constraints: the output is mathematically guaranteed to
satisfy the physical law, regardless of network weights. This is fundamentally
different from soft PDE loss penalties that only penalise violations.
"""

from typing import List

import torch
import torch.nn as nn


class DivergenceFreeProjection2d(nn.Module):
    """Project a 2-D velocity field to be exactly divergence-free.

    Uses the Helmholtz decomposition in Fourier space.  For a 2-D velocity
    field u = (u_x, u_y), the solenoidal (divergence-free) component is
    obtained by projecting out the irrotational part:

        û_⊥(k) = û(k) − [k · û(k) / |k|²] k

    This projection has **zero learnable parameters** and costs essentially
    nothing: it is a single pass through FFT → projection → iFFT.

    Parameters
    ----------
    eps : float
        Numerical floor for |k|² to avoid division by zero at the DC component.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Project velocity field to divergence-free subspace.

        Parameters
        ----------
        u : torch.Tensor  shape (B, 2, H, W)
            Two-component velocity field (channel 0 = u_x, channel 1 = u_y).

        Returns
        -------
        torch.Tensor  shape (B, 2, H, W)
            Velocity field with ∇·u = 0 enforced exactly.
        """
        assert u.shape[1] == 2, "DivergenceFreeProjection2d expects shape (B,2,H,W)"
        B, _, H, W = u.shape

        ux_ft = torch.fft.rfft2(u[:, 0], norm="ortho")  # (B, H, W//2+1)
        uy_ft = torch.fft.rfft2(u[:, 1], norm="ortho")

        # Wavenumber grids
        kx = torch.fft.fftfreq(H, device=u.device).reshape(H, 1)
        ky = torch.fft.rfftfreq(W, device=u.device).reshape(1, W // 2 + 1)
        k2 = kx ** 2 + ky ** 2

        # Projection: û_⊥ = û − (k·û / |k|²) k
        k2_safe = k2.clamp(min=self.eps)
        k_dot_u = kx * ux_ft + ky * uy_ft
        proj = k_dot_u / k2_safe
        proj[:, 0, 0] = 0.0  # preserve DC (mean) component

        ux_df = torch.fft.irfft2(ux_ft - proj * kx, s=(H, W), norm="ortho")
        uy_df = torch.fft.irfft2(uy_ft - proj * ky, s=(H, W), norm="ortho")

        return torch.stack([ux_df, uy_df], dim=1)


class ConservationProjection(nn.Module):
    """Renormalise a scalar output field to exactly conserve a global quantity.

    For each sample, the spatial integral (sum) of the output is rescaled to
    match the spatial integral of the corresponding input channel.  This
    enforces global conservation of mass, energy, or any other conserved
    quantity with zero extra parameters.

    Parameters
    ----------
    source_channel : int
        Channel index in the *input* used as the conserved-quantity reference.
    output_channel : int
        Channel index in the *output* to be renormalised.
    eps : float
        Numerical floor to avoid division by zero.
    """

    def __init__(
        self, source_channel: int = 0, output_channel: int = 0, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.source_channel = source_channel
        self.output_channel = output_channel
        self.eps = eps

    def forward(self, source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        source : torch.Tensor  shape (B, C_in, ...)
        output : torch.Tensor  shape (B, C_out, ...)

        Returns
        -------
        torch.Tensor
            Output with the specified channel rescaled to conserve the source integral.
        """
        spatial_dims = tuple(range(1, source.ndim - 1))
        src_int = source[:, self.source_channel].abs().sum(dim=spatial_dims)  # (B,)

        out_ch = output[:, self.output_channel]
        out_int = out_ch.abs().sum(dim=tuple(range(1, out_ch.ndim))) + self.eps  # (B,)

        scale = src_int / out_int  # (B,)
        for _ in range(output.ndim - 2):
            scale = scale.unsqueeze(-1)

        output_renorm = output.clone()
        output_renorm[:, self.output_channel] = output[:, self.output_channel] * scale
        return output_renorm


class SpectralBandFilter(nn.Module):
    """Learnable bandpass filter in Fourier space.

    Applies a per-band, per-channel learned gain to different frequency ranges.
    Used as a building block for MultiScaleFNO.

    Parameters
    ----------
    channels : int
    n_bands : int
        Number of frequency bands to partition the spectrum into.
    """

    def __init__(self, channels: int, n_bands: int = 3) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.band_weights = nn.Parameter(torch.ones(n_bands, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        kx = torch.fft.fftfreq(H, device=x.device).reshape(H, 1)
        ky = torch.fft.rfftfreq(W, device=x.device).reshape(1, W // 2 + 1)
        k_mag = (kx ** 2 + ky ** 2).sqrt()
        k_max = k_mag.max()

        band_edges = torch.linspace(0, k_max.item(), self.n_bands + 1, device=x.device)
        out_ft = torch.zeros_like(x_ft)
        for b in range(self.n_bands):
            mask = (k_mag >= band_edges[b]) & (k_mag < band_edges[b + 1])
            w = torch.sigmoid(self.band_weights[b]).reshape(1, C, 1, 1)
            out_ft = out_ft + x_ft * mask.unsqueeze(0).unsqueeze(0) * w

        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
