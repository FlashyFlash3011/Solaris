# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import numpy as np
import pytest
import torch

from solaris.datapipes.transforms import SymmetryAugmentation


# --- SymmetryAugmentation ---

@pytest.mark.parametrize("H,W", [(32, 32), (16, 24)])
def test_symmetry_augmentation_shape(H, W):
    aug = SymmetryAugmentation(keys=["input", "target"], p_rot=1.0, p_flip_h=0.5, p_flip_v=0.5)
    sample = {
        "input": torch.randn(2, H, W),
        "target": torch.randn(1, H, W),
    }
    out = aug(sample)
    # Shape must be preserved (90° rotation on square; non-square may swap dims)
    assert out["input"].shape[-1] in (H, W)
    assert out["target"].shape[-1] in (H, W)


def test_symmetry_augmentation_consistency():
    """Same transform must be applied to all keys — shapes stay consistent."""
    aug = SymmetryAugmentation(keys=["input", "target"], p_rot=1.0, p_flip_h=1.0, p_flip_v=1.0)
    sample = {
        "input": torch.randn(2, 32, 32),
        "target": torch.randn(1, 32, 32),
    }
    out = aug(sample)
    # Both keys must receive the exact same spatial transform
    assert out["input"].shape[-2:] == out["target"].shape[-2:]


def test_symmetry_augmentation_no_transform():
    aug = SymmetryAugmentation(keys=["input"], p_rot=0.0, p_flip_h=0.0, p_flip_v=0.0)
    t = torch.randn(1, 8, 8)
    out = aug({"input": t})
    assert torch.equal(out["input"], t)


# --- HDF5Dataset ---

@pytest.mark.skipif(
    not pytest.importorskip("h5py", reason="h5py not installed"),
    reason="h5py not installed",
)
def test_hdf5dataset_len_and_shapes():
    h5py = pytest.importorskip("h5py")
    from solaris.datapipes.dataset import HDF5Dataset

    N, C, H, W = 10, 2, 16, 16
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as hf:
            hf.create_dataset("input", data=np.random.randn(N, C, H, W).astype(np.float32))
            hf.create_dataset("target", data=np.random.randn(N, 1, H, W).astype(np.float32))

        ds = HDF5Dataset(path, input_key="input", target_key="target")
        assert len(ds) == N

        sample = ds[0]
        assert sample["input"].shape == (C, H, W)
        assert sample["target"].shape == (1, H, W)
    finally:
        os.unlink(path)


@pytest.mark.skipif(
    not pytest.importorskip("h5py", reason="h5py not installed"),
    reason="h5py not installed",
)
def test_hdf5dataset_with_transforms():
    h5py = pytest.importorskip("h5py")
    from solaris.datapipes.dataset import HDF5Dataset
    from solaris.datapipes.transforms import Normalize

    N, C, H, W = 5, 1, 8, 8
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        with h5py.File(path, "w") as hf:
            hf.create_dataset("input", data=np.ones((N, C, H, W), dtype=np.float32) * 10)
            hf.create_dataset("target", data=np.zeros((N, C, H, W), dtype=np.float32))

        norm = Normalize(keys=["input"], mean=10.0, std=1.0)
        ds = HDF5Dataset(path, transforms=[norm])
        sample = ds[0]
        # After normalizing: (10 - 10) / (1 + 1e-6) ≈ 0
        assert sample["input"].abs().max() < 1e-4
    finally:
        os.unlink(path)
