# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Normalisation choice for neural-operator blocks.

The reference FNO (Li et al., 2021) uses *no* normalisation inside its
Fourier layers.  Solaris historically applied ``InstanceNorm`` in every block,
which removes each sample's per-channel mean and scale at every layer.  The
network then cannot see a constant offset or the overall amplitude of its
input, so maps as simple as ``y = 2x`` or ``y = x + c`` become unlearnable.

Models therefore take ``norm=None`` (recommended) or ``norm="instance"``
(legacy).  Not passing ``norm`` keeps the legacy behaviour for now — so
existing checkpoints load unchanged — and emits a
:class:`~solaris.exceptions.FutureFeatureWarning`; the default will become
``None`` in a future release.
"""

import warnings

import torch.nn as nn

from solaris.exceptions import FutureFeatureWarning

NORM_CHOICES = (None, "instance")


class _Unset:
    def __repr__(self) -> str:  # shows up in signatures / help()
        return "<unset: 'instance' with a FutureFeatureWarning>"


UNSET = _Unset()


def resolve_norm(norm, model_name: str):
    """Validate ``norm``; map the unset default to ``"instance"`` with a warning."""
    if norm is UNSET:
        warnings.warn(
            f"{model_name}: norm was not specified. The legacy default 'instance' is used, "
            "which hides each sample's mean and scale from the network. Pass norm=None "
            "(recommended for new models) or norm='instance' (to match older checkpoints) "
            "explicitly; the default will change to None in a future release.",
            FutureFeatureWarning,
            stacklevel=4,
        )
        return "instance"
    if norm not in NORM_CHOICES:
        raise ValueError(f"norm must be one of {NORM_CHOICES}, got {norm!r}")
    return norm


def make_norm(norm: str | None, channels: int, dim: int) -> nn.Module:
    """Build the normalisation layer for a ``dim``-dimensional block."""
    if norm is None:
        return nn.Identity()
    return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}[dim](channels)
