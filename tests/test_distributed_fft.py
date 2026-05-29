# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DistributedRFFT2 / DistributedIRFFT2 (single-process fallback)."""

import torch

from solaris.distributed.fft import distributed_irfft2, distributed_rfft2


def test_rfft2_matches_local_single_rank():
    """distributed_rfft2 must match torch.fft.rfft2 when world_size=1."""
    x = torch.randn(2, 4, 32, 32)
    expected = torch.fft.rfft2(x, norm="ortho")
    got = distributed_rfft2(x, group=None, norm="ortho")
    torch.testing.assert_close(got, expected)


def test_irfft2_matches_local_single_rank():
    x = torch.randn(2, 4, 32, 32)
    x_f = torch.fft.rfft2(x, norm="ortho")
    expected = torch.fft.irfft2(x_f, s=(32, 32), norm="ortho")
    got = distributed_irfft2(x_f, output_size=(32, 32), group=None, norm="ortho")
    torch.testing.assert_close(got, expected)


def test_rfft2_irfft2_roundtrip():
    x = torch.randn(2, 3, 16, 16)
    x_f = distributed_rfft2(x)
    x_rec = distributed_irfft2(x_f, output_size=(16, 16))
    torch.testing.assert_close(x_rec, x, atol=1e-5, rtol=1e-5)


def test_rfft2_backward():
    x = torch.randn(2, 2, 16, 16, requires_grad=True)
    out = distributed_rfft2(x)
    out.abs().sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_irfft2_backward():
    x = torch.randn(2, 2, 16, 9, dtype=torch.cfloat, requires_grad=True)
    out = distributed_irfft2(x, output_size=(16, 16))
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_rfft2_output_shape():
    B, C, H, W = 3, 2, 32, 64
    x = torch.randn(B, C, H, W)
    out = distributed_rfft2(x)
    assert out.shape == (B, C, H, W // 2 + 1)


def test_rfft2_different_norms():
    x = torch.randn(1, 1, 8, 8)
    for norm in ("ortho", "forward", "backward"):
        out = distributed_rfft2(x, norm=norm)
        expected = torch.fft.rfft2(x, norm=norm)
        torch.testing.assert_close(out, expected)
