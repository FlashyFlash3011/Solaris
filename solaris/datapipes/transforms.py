# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Common data transforms for physics field data."""

import random

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
        mean: float | torch.Tensor | None = None,
        std: float | torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        self.keys = keys
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

    def __init__(self, keys: list, crop_size: tuple[int, int]) -> None:
        self.keys = keys
        self.crop_h, self.crop_w = crop_size

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Determine crop coordinates from first matching key
        h: int | None = None
        w: int | None = None
        for k in self.keys:
            if k in sample:
                t = sample[k]
                h, w = t.shape[-2], t.shape[-1]
                break
        if h is None or w is None:
            return sample
        if h < self.crop_h or w < self.crop_w:
            raise ValueError(f"cannot crop {(self.crop_h, self.crop_w)} from a {(h, w)} field")
        top = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        for k in self.keys:
            if k in sample:
                sample[k] = sample[k][..., top : top + self.crop_h, left : left + self.crop_w]
        return sample


class ToDevice:
    """Move all tensors in a sample dict to the specified device."""

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in sample.items()
        }


class SymmetryAugmentation:
    """Apply the same random rotation / flips to every key of a sample.

    Rotations are by multiples of 90° (only 0° or 180° for non-square fields,
    so shapes never change and samples still batch together); flips are
    left-right and up-down.

    **Vector fields.** Moving the grid is enough for scalar fields
    (temperature, pressure), but a vector field such as velocity must also
    have its *components* rotated/reflected, or the augmented sample breaks
    the physics (a divergence-free flow stops being divergence-free).  List
    the component channels in ``vector_channels``: for each key, pairs
    ``(x_channel, y_channel)`` where *x* points along the last axis (W) and
    *y* along the second-to-last axis (H).

    Parameters
    ----------
    keys : list[str]
        Keys in the sample dict to transform.
    p_rot : float
        Probability of a random rotation.
    p_flip_h : float
        Probability of a horizontal flip (left-right).
    p_flip_v : float
        Probability of a vertical flip (up-down).
    vector_channels : dict[str, list[tuple[int, int]]], optional
        Vector component channels per key, e.g. ``{"target": [(0, 1)]}``.
    """

    def __init__(
        self,
        keys: list,
        p_rot: float = 0.5,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        vector_channels: dict[str, list[tuple[int, int]]] | None = None,
    ) -> None:
        self.keys = keys
        self.p_rot = p_rot
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.vector_channels = vector_channels or {}

    @staticmethod
    def _rot90_once(x: torch.Tensor, pairs) -> torch.Tensor:
        # torch.rot90(·, 1) == flip_rows(transpose): vectors swap components,
        # then the row flip negates the new y component.
        x = torch.rot90(x, 1, dims=(-2, -1))
        if pairs:
            x = x.clone()
            for cx, cy in pairs:
                ux, uy = x[..., cx, :, :].clone(), x[..., cy, :, :].clone()
                x[..., cx, :, :], x[..., cy, :, :] = uy, -ux
        return x

    @staticmethod
    def _negate(x: torch.Tensor, channels) -> torch.Tensor:
        if not channels:
            return x
        x = x.clone()
        for c in channels:
            x[..., c, :, :] = -x[..., c, :, :]
        return x

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        present = [k for k in self.keys if k in sample]
        square = all(sample[k].shape[-2] == sample[k].shape[-1] for k in present)
        if random.random() < self.p_rot:
            k_rot = random.randint(0, 3) if square else 2 * random.randint(0, 1)
        else:
            k_rot = 0
        do_flip_h = random.random() < self.p_flip_h
        do_flip_v = random.random() < self.p_flip_v

        for k in present:
            x = sample[k]
            pairs = self.vector_channels.get(k, [])
            for _ in range(k_rot):
                x = self._rot90_once(x, pairs)
            if do_flip_h:  # x -> -x
                x = self._negate(torch.flip(x, dims=(-1,)), [cx for cx, _ in pairs])
            if do_flip_v:  # y -> -y
                x = self._negate(torch.flip(x, dims=(-2,)), [cy for _, cy in pairs])
            sample[k] = x
        return sample


class AddGaussianNoise:
    """Add Gaussian noise to selected tensor keys (data augmentation)."""

    def __init__(self, keys: list, std: float = 0.01) -> None:
        self.keys = keys
        self.std = std

    def __call__(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for k in self.keys:
            if k in sample:
                sample[k] = sample[k] + torch.randn_like(sample[k]) * self.std
        return sample
