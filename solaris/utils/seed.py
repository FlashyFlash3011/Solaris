# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Seed management for reproducible training."""

import random

import numpy as np
import torch


def set_seed(seed: int, rank: int = 0) -> None:
    """Set all global random seeds for reproducible training.

    Sets seeds for Python's ``random``, NumPy, and PyTorch (CPU and CUDA).
    The effective seed is ``seed + rank`` so that each distributed worker
    uses a unique but deterministic seed.

    Parameters
    ----------
    seed : int
        Base random seed.
    rank : int
        Distributed rank offset (default 0 for single-process training).
    """
    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective)


def set_deterministic(enabled: bool = True) -> None:
    """Enable or disable PyTorch deterministic algorithms.

    When enabled, operations that do not have a deterministic implementation
    will raise a ``RuntimeError`` rather than silently producing
    non-reproducible results.

    Parameters
    ----------
    enabled : bool
        Pass ``False`` to restore the default (non-deterministic) behaviour.
    """
    torch.use_deterministic_algorithms(enabled)
    if enabled and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
