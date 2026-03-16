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
