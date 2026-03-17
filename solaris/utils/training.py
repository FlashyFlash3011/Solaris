# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Training utility classes: early stopping and gradient clipping."""

import torch
import torch.nn as nn


class EarlyStopping:
    """Stop training when a monitored metric has stopped improving.

    Parameters
    ----------
    patience : int
        Number of calls with no improvement before stopping.
    min_delta : float
        Minimum change to count as improvement.
    mode : str
        ``"min"`` (lower is better, e.g. loss) or ``"max"`` (higher is better,
        e.g. accuracy).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min") -> None:
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0

    def step(self, value: float) -> bool:
        """Update with new metric value.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if self.mode == "min":
            improved = value < self._best - self.min_delta
        else:
            improved = value > self._best + self.min_delta

        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience

    def reset(self) -> None:
        """Reset state (e.g. at the start of a new training run)."""
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._counter = 0


class GradientClipper:
    """Clip gradient norms of a model's parameters.

    A thin wrapper around :func:`torch.nn.utils.clip_grad_norm_` that can be
    called as a callable after ``loss.backward()``.

    Parameters
    ----------
    max_norm : float
        Maximum allowed gradient norm.
    norm_type : float
        Type of the norm (default 2.0).
    """

    def __init__(self, max_norm: float, norm_type: float = 2.0) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model: nn.Module) -> float:
        """Clip gradients and return the pre-clip total norm.

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        float
            Total gradient norm before clipping.
        """
        return float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.max_norm, norm_type=self.norm_type
            )
        )
