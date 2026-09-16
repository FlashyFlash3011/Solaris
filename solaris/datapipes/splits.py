# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Leakage-free train/validation splitting.

Simulation datasets usually contain many samples per simulation (several
time steps, lead times or snapshots of the same run).  Splitting samples at
random puts near-identical samples from one run on both sides of the split,
so validation scores look better than real generalisation.
:func:`split_by_group` splits whole groups (e.g. simulations) instead.
"""

import numpy as np


def split_by_group(
    groups: np.ndarray,
    val_fraction: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices so that no group appears on both sides.

    Parameters
    ----------
    groups :
        Group id of each sample, shape ``(N,)`` (e.g. the simulation index).
    val_fraction :
        Fraction of *groups* assigned to validation.  At least one group goes
        to each side.
    seed :
        Seed for the group permutation.

    Returns
    -------
    (train_idx, val_idx) : tuple of sorted index arrays
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("split_by_group needs at least two distinct groups")
    n_val = int(round(len(unique) * val_fraction))
    n_val = min(max(n_val, 1), len(unique) - 1)
    val_groups = np.random.default_rng(seed).permutation(unique)[:n_val]
    is_val = np.isin(groups, val_groups)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)
