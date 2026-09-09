# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from solaris.models import ConstrainedFNO
from solaris.nn import (
    ConservationProjection,
    CurlFreeProjection2d,
    DirichletBCLayer,
    DivergenceFreeProjection2d,
    NeumannBCLayer,
    SpectralBandFilter,
)


@pytest.mark.parametrize("B,H,W", [(2, 32, 32), (1, 16, 16)])
def test_curl_free_projection_shape(B, H, W):
    layer = CurlFreeProjection2d()
    u = torch.randn(B, 2, H, W)
    out = layer(u)
    assert out.shape == u.shape


@pytest.mark.parametrize("B,H,W", [(2, 32, 32), (1, 16, 16)])
def test_helmholtz_completeness(B, H, W):
    """DivergenceFree + CurlFree should sum to identity."""
    df = DivergenceFreeProjection2d()
    cf = CurlFreeProjection2d()
    u = torch.randn(B, 2, H, W)
    reconstructed = df(u) + cf(u)
    assert torch.allclose(u, reconstructed, atol=1e-5), (
        f"Max deviation: {(u - reconstructed).abs().max().item()}"
    )


@pytest.mark.parametrize("B,C,H,W", [(2, 3, 16, 16), (1, 1, 32, 32)])
def test_neumann_bc_zero_flux(B, C, H, W):
    layer = NeumannBCLayer(dims=(2, 3))
    x = torch.randn(B, C, H, W)
    out = layer(x)
    assert out.shape == x.shape
    # Top row == second row
    assert torch.allclose(out[:, :, 0, :], out[:, :, 1, :])
    # Bottom row == second-to-last row
    assert torch.allclose(out[:, :, -1, :], out[:, :, -2, :])
    # Left col == second col
    assert torch.allclose(out[:, :, :, 0], out[:, :, :, 1])
    # Right col == second-to-last col
    assert torch.allclose(out[:, :, :, -1], out[:, :, :, -2])


@pytest.mark.parametrize("B,C,H,W", [(2, 1, 16, 16), (1, 2, 32, 32)])
def test_dirichlet_bc_boundary_values(B, C, H, W):
    bv = torch.ones(1, C, H, W) * 5.0
    layer = DirichletBCLayer(spatial_shape=(H, W), channels=C, boundary_values=bv)
    x = torch.randn(B, C, H, W)
    out = layer(x)
    assert out.shape == x.shape
    # Border pixels should equal boundary_values
    assert torch.allclose(out[:, :, 0, :], bv[:, :, 0, :].expand(B, -1, -1))
    assert torch.allclose(out[:, :, -1, :], bv[:, :, -1, :].expand(B, -1, -1))
    assert torch.allclose(out[:, :, :, 0], bv[:, :, :, 0].expand(B, -1, -1))
    assert torch.allclose(out[:, :, :, -1], bv[:, :, :, -1].expand(B, -1, -1))


def test_dirichlet_bc_default_zero_boundary():
    layer = DirichletBCLayer(spatial_shape=(8, 8), channels=1)
    x = torch.ones(2, 1, 8, 8)
    out = layer(x)
    assert torch.allclose(out[:, :, 0, :], torch.zeros(2, 1, 8))
    assert torch.allclose(out[:, :, -1, :], torch.zeros(2, 1, 8))


# --- ConservationProjection ---------------------------------------------------


def test_conservation_projection_matches_signed_integral():
    torch.manual_seed(0)
    src = torch.randn(4, 1, 16, 16)
    out = torch.randn(4, 1, 16, 16)
    res = ConservationProjection()(src, out)
    torch.testing.assert_close(res.sum(dim=(1, 2, 3)), src.sum(dim=(1, 2, 3)))


def test_conservation_projection_multiplicative_keeps_positivity():
    src = torch.rand(3, 1, 8, 8) + 0.1
    out = torch.rand(3, 1, 8, 8) + 0.1
    res = ConservationProjection(mode="multiplicative")(src, out)
    torch.testing.assert_close(res.sum(dim=(1, 2, 3)), src.sum(dim=(1, 2, 3)))
    assert (res > 0).all()


def test_conservation_projection_leaves_other_channels_untouched():
    src = torch.randn(2, 2, 8, 8)
    out = torch.randn(2, 3, 8, 8)
    res = ConservationProjection(source_channel=1, output_channel=2)(src, out)
    torch.testing.assert_close(res[:, :2], out[:, :2])
    torch.testing.assert_close(res[:, 2].sum(dim=(1, 2)), src[:, 1].sum(dim=(1, 2)))


def test_constrained_fno_conservative_conserves():
    model = ConstrainedFNO(1, 1, 8, 1, 4, constraint="conservative")
    x = torch.randn(3, 1, 16, 16)
    torch.testing.assert_close(model(x).sum(dim=(1, 2, 3)), x.sum(dim=(1, 2, 3)))


# --- SpectralBandFilter --------------------------------------------------------


def test_spectral_band_filter_covers_highest_frequency():
    f = SpectralBandFilter(1, n_bands=3)
    with torch.no_grad():
        f.band_weights.fill_(50.0)  # sigmoid(50) == 1 → filter should be identity
    x = torch.randn(1, 1, 16, 16)
    torch.testing.assert_close(f(x), x, atol=1e-5, rtol=1e-5)
