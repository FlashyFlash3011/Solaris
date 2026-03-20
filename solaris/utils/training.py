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


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup followed by cosine decay.

    Wraps :class:`torch.optim.lr_scheduler.CosineAnnealingLR` with an
    initial linear warmup phase.  Call :meth:`step` once per epoch.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of epochs to linearly ramp the LR from 0 to ``base_lr``.
    total_epochs : int
        Total training epochs (including warmup).
    base_lr : float
        Peak learning rate (after warmup).
    min_lr : float
        Minimum LR at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._epoch = 0
        self._cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=min_lr,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = 0.0

    def step(self) -> float:
        """Advance one epoch and return the new learning rate."""
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            lr = self.base_lr * self._epoch / max(1, self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self._cosine.step()
            lr = self.optimizer.param_groups[0]["lr"]
        return lr

    @property
    def last_lr(self) -> float:
        """Current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class AutoCheckpoint:
    """Save the best model checkpoint automatically during training.

    Monitors a scalar metric and saves the model whenever an improvement
    is recorded.  Uses :meth:`solaris.core.Module.save` when available,
    falling back to ``torch.save`` on the raw state dict.

    Parameters
    ----------
    path : str or Path
        File path for the checkpoint.
    mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    min_delta : float
        Minimum change to count as improvement.
    """

    def __init__(
        self,
        path,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        import pathlib
        assert mode in ("min", "max"), "mode must be 'min' or 'max'"
        self.path = pathlib.Path(path)
        self.mode = mode
        self.min_delta = min_delta
        self._best: float = float("inf") if mode == "min" else float("-inf")

    def update(self, model: nn.Module, metric: float) -> bool:
        """Save model if metric improved.

        Parameters
        ----------
        model : nn.Module
        metric : float

        Returns
        -------
        bool
            ``True`` if the checkpoint was updated (metric improved).
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best = metric
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save"):
                model.save(self.path)
            else:
                torch.save(model.state_dict(), self.path)
        return improved

    @property
    def best(self) -> float:
        """Best metric value seen so far."""
        return self._best
