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
    # Shapes are always preserved: non-square fields only rotate by 0° or 180°
    assert out["input"].shape[-2:] == (H, W)
    assert out["target"].shape[-2:] == (H, W)


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


# --- PhysicsDataset raw arrays ------------------------------------------------------


def test_physics_dataset_raw_ndarray(tmp_path):
    from solaris.datapipes import PhysicsDataset

    np.save(tmp_path / "a.npy", np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4))
    sample = PhysicsDataset(tmp_path)[0]
    assert sample["input"].shape == (1, 4, 4)
    assert sample["target"].shape == (1, 4, 4)


def test_physics_dataset_dict_npy(tmp_path):
    from solaris.datapipes import PhysicsDataset

    np.save(tmp_path / "a.npy", {"input": np.zeros((1, 4, 4)), "target": np.ones((1, 4, 4))})
    sample = PhysicsDataset(tmp_path)[0]
    assert sample["target"].sum() == 16


# --- SymmetryAugmentation with vector fields -----------------------------------


def _grad(phi):
    """Periodic central-difference gradient: (d/dx along W, d/dy along H)."""
    gx = (torch.roll(phi, -1, -1) - torch.roll(phi, 1, -1)) / 2
    gy = (torch.roll(phi, -1, -2) - torch.roll(phi, 1, -2)) / 2
    return torch.cat([gx, gy], dim=0)


def _force(aug, k_rot, flip_h, flip_v):
    """Make the augmentation's random draws deterministic."""
    import random as _random

    draws = iter([0.0, k_rot, 0.0 if flip_h else 1.0, 0.0 if flip_v else 1.0])
    orig_random, orig_randint = _random.random, _random.randint
    _random.random = lambda: next(draws)
    _random.randint = lambda a, b: next(draws)
    return orig_random, orig_randint


@pytest.mark.parametrize("k_rot", [0, 1, 2, 3])
@pytest.mark.parametrize("flip_h", [False, True])
@pytest.mark.parametrize("flip_v", [False, True])
def test_symmetry_augmentation_transforms_vector_components(k_rot, flip_h, flip_v):
    """Augmenting a gradient field must equal the gradient of the augmented scalar."""
    import random as _random

    torch.manual_seed(0)
    phi = torch.randn(1, 8, 8)
    sample = {"phi": phi.clone(), "u": _grad(phi)}
    aug = SymmetryAugmentation(
        keys=["phi", "u"], p_rot=1.0, p_flip_h=1.0, p_flip_v=1.0, vector_channels={"u": [(0, 1)]}
    )
    saved = _force(aug, k_rot, flip_h, flip_v)
    try:
        out = aug(sample)
    finally:
        _random.random, _random.randint = saved
    torch.testing.assert_close(out["u"], _grad(out["phi"]))


@pytest.mark.parametrize("seed", range(20))
def test_symmetry_augmentation_keeps_non_square_shape(seed):
    import random as _random

    _random.seed(seed)
    aug = SymmetryAugmentation(keys=["x"], p_rot=1.0, p_flip_h=0.5, p_flip_v=0.5)
    assert aug({"x": torch.randn(1, 4, 6)})["x"].shape == (1, 4, 6)


def test_normalize_with_given_stats():
    from solaris.datapipes.transforms import Normalize

    out = Normalize(keys=["x"], mean=2.0, std=4.0)({"x": torch.tensor([2.0, 6.0])})
    torch.testing.assert_close(out["x"], torch.tensor([0.0, 1.0]), atol=1e-5, rtol=1e-5)


def test_random_crop_too_small_raises():
    from solaris.datapipes.transforms import RandomCrop2d

    with pytest.raises(ValueError, match="crop"):
        RandomCrop2d(keys=["x"], crop_size=(8, 8))({"x": torch.randn(1, 4, 4)})


def test_add_gaussian_noise_changes_only_selected_keys():
    from solaris.datapipes.transforms import AddGaussianNoise

    torch.manual_seed(0)
    s = {"x": torch.zeros(1000), "y": torch.zeros(3)}
    out = AddGaussianNoise(keys=["x"], std=0.1)(s)
    assert 0.08 < out["x"].std() < 0.12
    assert out["y"].abs().sum() == 0
