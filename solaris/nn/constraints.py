# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Physics constraint layers — enforce conservation laws at the architecture level.

These layers are *hard* constraints: the output is mathematically guaranteed to
satisfy the physical law, regardless of network weights. This is fundamentally
different from soft PDE loss penalties that only penalise violations.
"""

import torch
import torch.nn as nn


def _helmholtz_split(u: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (divergence-free, curl-free) parts of a periodic 2-D field.

    Computed in fp32 (or fp64) with autocast disabled — fp16 FFTs are not
    supported on CPU nor for non-power-of-two sizes on GPU.
    """
    assert u.ndim == 4 and u.shape[1] == 2, "expected a (B, 2, H, W) vector field"
    _, _, H, W = u.shape
    with torch.amp.autocast(device_type=u.device.type, enabled=False):
        uf = u if u.dtype in (torch.float32, torch.float64) else u.float()
        ux_ft = torch.fft.rfft2(uf[:, 0], norm="ortho")  # (B, H, W//2+1)
        uy_ft = torch.fft.rfft2(uf[:, 1], norm="ortho")

        kx = torch.fft.fftfreq(H, device=u.device, dtype=uf.dtype).reshape(H, 1)
        ky = torch.fft.rfftfreq(W, device=u.device, dtype=uf.dtype).reshape(1, W // 2 + 1)
        k2 = (kx**2 + ky**2).clamp(min=eps)

        # Longitudinal (curl-free) coefficient: (k·û / |k|²); the mean is left
        # in the divergence-free part.
        proj = (kx * ux_ft + ky * uy_ft) / k2
        proj[:, 0, 0] = 0.0

        cf = torch.stack(
            [
                torch.fft.irfft2(proj * kx, s=(H, W), norm="ortho"),
                torch.fft.irfft2(proj * ky, s=(H, W), norm="ortho"),
            ],
            dim=1,
        )
        df = uf - cf
    return df.to(u.dtype), cf.to(u.dtype)


class DivergenceFreeProjection2d(nn.Module):
    """Project a 2-D velocity field to be exactly divergence-free.

    Uses the Helmholtz decomposition in Fourier space (periodic domain).  For
    a 2-D velocity field u = (u_x, u_y), the solenoidal (divergence-free)
    component is obtained by projecting out the irrotational part:

        û_⊥(k) = û(k) − [k · û(k) / |k|²] k

    This projection has **zero learnable parameters** and costs a single
    FFT → projection → iFFT pass.

    Parameters
    ----------
    eps : float
        Numerical floor for |k|² to avoid division by zero at the DC component.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Project ``u`` of shape ``(B, 2, H, W)`` onto the divergence-free subspace."""
        return _helmholtz_split(u, self.eps)[0]


class ConservationProjection(nn.Module):
    """Correct an output field so it exactly conserves a global quantity.

    For each sample, output channel ``output_channel`` is adjusted so that its
    spatial sum equals the spatial sum of input channel ``source_channel``
    (i.e. the integral on a uniform grid, up to the constant cell area).

    Two correction modes are available:

    ``"additive"`` (default)
        Shift the field by a constant: ``out + (Σsrc − Σout) / N``.  Exact for
        fields of any sign and the minimum-L2 correction.
    ``"multiplicative"``
        Rescale the field: ``out · Σsrc / Σout``.  Preserves positivity, so it
        suits strictly positive quantities (density, concentration, water
        depth).  Undefined when ``Σout`` is ~0; ``eps`` guards the division.

    Important: the constraint is only physically meaningful when both
    channels are in the *same physical units*.  Applying it to independently
    z-score–normalised channels conserves a meaningless quantity — enforce it
    on de-normalised fields instead.

    Parameters
    ----------
    source_channel : int
        Channel index in the *input* used as the conserved-quantity reference.
    output_channel : int
        Channel index in the *output* to be corrected.
    mode : str
        ``"additive"`` or ``"multiplicative"``.
    eps : float
        Numerical floor for the multiplicative denominator.
    """

    def __init__(
        self,
        source_channel: int = 0,
        output_channel: int = 0,
        mode: str = "additive",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if mode not in ("additive", "multiplicative"):
            raise ValueError(f"mode must be 'additive' or 'multiplicative', got {mode!r}")
        self.source_channel = source_channel
        self.output_channel = output_channel
        self.mode = mode
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
            ``output`` with the selected channel corrected to conserve the source integral.
        """
        src = source[:, self.source_channel]
        out = output[:, self.output_channel]
        spatial = tuple(range(1, out.ndim))
        src_int = src.sum(dim=tuple(range(1, src.ndim)), keepdim=True).reshape(
            -1, *([1] * (out.ndim - 1))
        )
        out_int = out.sum(dim=spatial, keepdim=True)

        if self.mode == "additive":
            n = out[0].numel()
            corrected = out + (src_int - out_int) / n
        else:
            sign = torch.where(out_int >= 0, 1.0, -1.0).to(out.dtype)
            corrected = out * (src_int / (out_int + sign * self.eps))

        channels = list(output.unbind(dim=1))
        channels[self.output_channel] = corrected
        return torch.stack(channels, dim=1)


class SpectralBandFilter(nn.Module):
    """Learnable bandpass filter in Fourier space.

    Splits the 2-D spectrum into ``n_bands`` equal-width rings in |k| and
    applies a learned per-band, per-channel gain in (0, 1).  Every mode
    belongs to exactly one band (the top band includes the maximum |k|).

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
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xf = x if x.dtype in (torch.float32, torch.float64) else x.float()
            x_ft = torch.fft.rfft2(xf, norm="ortho")

            kx = torch.fft.fftfreq(H, device=x.device).reshape(H, 1)
            ky = torch.fft.rfftfreq(W, device=x.device).reshape(1, W // 2 + 1)
            k_mag = (kx**2 + ky**2).sqrt()
            # Band index per mode; clamp puts k == k_max into the top band.
            band = (k_mag / k_mag.max() * self.n_bands).long().clamp(max=self.n_bands - 1)

            gains = torch.sigmoid(self.band_weights).to(xf.dtype)  # (n_bands, C)
            gain_map = gains[band].permute(2, 0, 1)  # (C, H, W//2+1)
            out = torch.fft.irfft2(x_ft * gain_map, s=(H, W), norm="ortho")
        return out.to(x.dtype)


class CurlFreeProjection2d(nn.Module):
    """Project a 2-D vector field to be exactly curl-free (irrotational).

    The curl-free component is the complement of the divergence-free component
    in the Helmholtz decomposition: u = u_⊥ + u_∥, so

        u_∥(k) = [k · û(k) / |k|²] k

    Together, DivergenceFreeProjection2d + CurlFreeProjection2d = identity.

    Parameters
    ----------
    eps : float
        Numerical floor for |k|² to avoid division by zero at the DC component.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _helmholtz_split(u, self.eps)[1]


class NeumannBCLayer(nn.Module):
    """Enforce zero-flux (Neumann) boundary conditions on selected spatial dims.

    Sets boundary slices equal to their immediate interior neighbours,
    which imposes ∂u/∂n = 0 on those edges with zero learnable parameters.

    Parameters
    ----------
    dims : tuple[int]
        Spatial dimension indices (relative to the full tensor) on which to
        enforce the BC.  Typically ``(2, 3)`` for (B, C, H, W) tensors.
    """

    def __init__(self, dims: tuple = (2, 3)) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for d in self.dims:
            idx_first = [slice(None)] * x.ndim
            idx_second = [slice(None)] * x.ndim
            idx_last = [slice(None)] * x.ndim
            idx_second_last = [slice(None)] * x.ndim

            idx_first[d] = 0
            idx_second[d] = 1
            idx_last[d] = -1
            idx_second_last[d] = -2

            x[tuple(idx_first)] = x[tuple(idx_second)]
            x[tuple(idx_last)] = x[tuple(idx_second_last)]
        return x


class DirichletBCLayer(nn.Module):
    """Enforce Dirichlet boundary conditions by overwriting border pixels.

    The interior of the field is left unchanged; only the outermost pixels
    on the H and W dimensions are set to ``boundary_values``.

    Parameters
    ----------
    spatial_shape : tuple[int, int]
        (H, W) of the field.
    channels : int
        Number of channels C.
    boundary_values : torch.Tensor, optional
        Shape (1, C, H, W). Defaults to all-zeros.
    """

    def __init__(
        self,
        spatial_shape: tuple,
        channels: int,
        boundary_values: torch.Tensor = None,
    ) -> None:
        super().__init__()
        H, W = spatial_shape
        if boundary_values is None:
            boundary_values = torch.zeros(1, channels, H, W)
        self.register_buffer("boundary_values", boundary_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        bv = self.boundary_values
        x[:, :, 0, :] = bv[:, :, 0, :]
        x[:, :, -1, :] = bv[:, :, -1, :]
        x[:, :, :, 0] = bv[:, :, :, 0]
        x[:, :, :, -1] = bv[:, :, :, -1]
        return x
