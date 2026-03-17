# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from solaris.nn import (
    DivergenceFreeProjection2d,
    CurlFreeProjection2d,
    NeumannBCLayer,
    DirichletBCLayer,
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
