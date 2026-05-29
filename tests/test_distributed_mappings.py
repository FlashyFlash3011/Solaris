# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for tensor-parallel mapping primitives (single-process, world_size=1)."""

import torch

from solaris.distributed.mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
)


def _make(shape, requires_grad=True):
    return torch.randn(*shape, requires_grad=requires_grad)


# --- Forward pass: world_size=1 means identity for all primitives ---


def test_scatter_identity_single_rank():
    x = _make((8, 4))
    out = scatter_to_model_parallel_region(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_gather_identity_single_rank():
    x = _make((8, 4))
    out = gather_from_model_parallel_region(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_reduce_identity_single_rank():
    x = _make((8, 4))
    out = reduce_from_model_parallel_region(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


def test_copy_identity_single_rank():
    x = _make((8, 4))
    out = copy_to_model_parallel_region(x)
    assert out.shape == x.shape
    assert torch.equal(out, x)


# --- Backward pass: gradients flow correctly ---


def test_scatter_backward():
    x = _make((8, 4))
    out = scatter_to_model_parallel_region(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_gather_backward():
    x = _make((4, 4))
    out = gather_from_model_parallel_region(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_reduce_backward():
    x = _make((8, 4))
    out = reduce_from_model_parallel_region(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_copy_backward():
    x = _make((8, 4))
    out = copy_to_model_parallel_region(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_scatter_gather_roundtrip():
    """scatter → gather is identity for world_size=1."""
    x = _make((8, 4), requires_grad=False)
    out = gather_from_model_parallel_region(scatter_to_model_parallel_region(x))
    torch.testing.assert_close(out, x)
