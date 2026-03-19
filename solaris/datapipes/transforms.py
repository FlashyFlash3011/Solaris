# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Common data transforms for physics field data."""

import random
from typing import Dict, Optional, Tuple, Union

import torch


class Normalize:
    """Normalise selected keys to zero mean and unit variance.

    Parameters
    ----------
    keys : list[str]
        Which keys in the sample dict to normalise.
    mean : float or Tensor, optional
        Pre-computed mean. If ``None``, computed per-sample.
    std : float or Tensor, optional
        Pre-computed std. If ``None``, computed per-sample.
    eps : float
        Stability epsilon added to the denominator.
    """

    def __init__(
        self,
        keys: list,
        mean: Optional[Union[float, torch.Tensor]] = None,
        std: Optional[Union[float, torch.Tensor]] = None,
        eps: float = 1e-6,
    ) -> None:
        self.keys = keys
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k in self.keys:
            if k not in sample:
                continue
            x = sample[k]
            mean = self.mean if self.mean is not None else x.mean()
            std = self.std if self.std is not None else x.std()
            sample[k] = (x - mean) / (std + self.eps)
        return sample


class RandomCrop2d:
    """Random spatial crop for 2-D field data (..., H, W)."""

    def __init__(self, keys: list, crop_size: Tuple[int, int]) -> None:
        self.keys = keys
        self.crop_h, self.crop_w = crop_size

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Determine crop coordinates from first matching key
        h = w = None
        for k in self.keys:
            if k in sample:
                t = sample[k]
                h, w = t.shape[-2], t.shape[-1]
                break
        if h is None:
            return sample
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        for k in self.keys:
            if k in sample:
                sample[k] = sample[k][..., top : top + self.crop_h, left : left + self.crop_w]
        return sample


class ToDevice:
    """Move all tensors in a sample dict to the specified device."""

    def __init__(self, device: Union[str, torch.device]) -> None:
        self.device = torch.device(device)

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}


class SymmetryAugmentation:
    """Apply consistent random rotations and flips to all keys in a sample.

    The same transform is applied to every key so that input/target pairs
    remain aligned.  Supports 90° rotations and horizontal/vertical flips.

    Parameters
    ----------
    keys : list[str]
        Keys in the sample dict to transform.
    p_rot : float
        Probability of applying a random 90° rotation (0, 90, 180, or 270°).
    p_flip_h : float
        Probability of a horizontal flip (left-right).
    p_flip_v : float
        Probability of a vertical flip (up-down).
    """

    def __init__(self, keys: list, p_rot: float = 0.5, p_flip_h: float = 0.5, p_flip_v: float = 0.5) -> None:
        self.keys = keys
        self.p_rot = p_rot
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        k_rot = random.randint(0, 3) if random.random() < self.p_rot else 0
        do_flip_h = random.random() < self.p_flip_h
        do_flip_v = random.random() < self.p_flip_v

        for k in self.keys:
            if k not in sample:
                continue
            x = sample[k]
            if k_rot:
                x = torch.rot90(x, k_rot, dims=(-2, -1))
            if do_flip_h:
                x = torch.flip(x, dims=(-1,))
            if do_flip_v:
                x = torch.flip(x, dims=(-2,))
            sample[k] = x
        return sample


class AddGaussianNoise:
    """Add Gaussian noise to selected tensor keys (data augmentation)."""

    def __init__(self, keys: list, std: float = 0.01) -> None:
        self.keys = keys
        self.std = std

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k in self.keys:
            if k in sample:
                sample[k] = sample[k] + torch.randn_like(sample[k]) * self.std
        return sample
