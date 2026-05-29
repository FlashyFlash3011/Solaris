# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Training utility classes: early stopping, gradient clipping, graph capture, combined optimiser."""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


class CombinedOptimizer(torch.optim.Optimizer):
    """Compose multiple optimisers into a single object.

    Useful for :class:`~solaris.models.CoupledOperator` where different
    sub-operators benefit from independent learning rates, while the outer
    training loop (schedulers, gradient clippers) sees a single optimiser.

    Subclasses :class:`torch.optim.Optimizer` so that PyTorch LR schedulers
    (e.g. :class:`torch.optim.lr_scheduler.CosineAnnealingLR`) accept it
    without modification.

    Parameters
    ----------
    optimizers : list[torch.optim.Optimizer]
        Optimisers to combine.  Each manages its own parameter groups and
        internal state independently.

    Example
    -------
    ::

        opt = CombinedOptimizer([
            torch.optim.Adam(model.operators["thermal"].parameters(), lr=1e-3),
            torch.optim.Adam(model.operators["fluid"].parameters(), lr=5e-4),
        ])
        scheduler = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=100, base_lr=1e-3)
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("CombinedOptimizer requires at least one optimizer")
        self.optimizers = optimizers
        # Inherit param_groups from sub-optimisers so schedulers see them all.
        # torch.optim.Optimizer stores param_groups as a list of dicts —
        # we share the actual dicts (same objects) so LR updates propagate.
        all_param_groups = [pg for opt in optimizers for pg in opt.param_groups]
        # Bypass torch.optim.Optimizer.__init__ (which expects raw params).
        # We set the required attributes manually.
        self.param_groups = all_param_groups
        self.state: dict = {}
        self.defaults: dict = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Callable | None = None) -> None:  # type: ignore[override]
        for opt in self.optimizers:
            opt.step(closure)

    def state_dict(self) -> list[dict]:  # type: ignore[override]
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, states: list[dict]) -> None:  # type: ignore[override]
        if len(states) != len(self.optimizers):
            raise ValueError(f"Expected {len(self.optimizers)} state dicts, got {len(states)}")
        for opt, state in zip(self.optimizers, states):
            opt.load_state_dict(state)


class StaticCaptureTraining:
    """Accelerate a training step via CUDA/HIP graph capture.

    After *warmup_steps* warm-up iterations, the training step is captured
    into a CUDA graph and replayed on subsequent calls, eliminating CPU-side
    kernel launch overhead.  This can give 2–5× throughput improvements for
    models with many small kernels (e.g. FNO on large grids).

    Falls back gracefully to a normal forward/backward/step when:

    * No CUDA/ROCm GPU is available.
    * ``enabled=False``.
    * Graph capture fails (e.g. non-static control flow or ops that prevent
      capture — a warning is printed and training continues normally).

    **CUDA graph requirements** (applies equally to ROCm HIP graphs):

    * Input shapes must be *static* — the same ``(B, C, H, W)`` every call.
    * No Python-side branching inside the captured step that varies per call.
    * The model must be in ``train()`` mode throughout (mode switch breaks capture).

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer or CombinedOptimizer
    loss_fn : callable
        ``loss_fn(pred, target) -> Tensor``.  Must be deterministic.
    warmup_steps : int
        Number of normal steps before graph capture (default 20).
    enabled : bool
        Set ``False`` to disable graph capture entirely (useful for debugging).

    Example
    -------
    ::

        capturer = StaticCaptureTraining(model, optimizer, loss_fn=F.mse_loss)

        for x, y in dataloader:
            loss = capturer(x, y)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        warmup_steps: int = 20,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.warmup_steps = warmup_steps
        self.enabled = enabled and torch.cuda.is_available()
        self._graph: Any = None
        self._step_count = 0
        self._static_x: torch.Tensor | None = None
        self._static_y: torch.Tensor | None = None
        self._static_loss: torch.Tensor | None = None

    def _run_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Execute one normal (non-captured) training step."""
        self.optimizer.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Run a training step, using graph replay after warm-up.

        Parameters
        ----------
        x : torch.Tensor
            Model input (shape must be static after warm-up).
        y : torch.Tensor
            Target (shape must be static after warm-up).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if not self.enabled:
            return self._run_step(x, y)

        self._step_count += 1

        if self._step_count <= self.warmup_steps:
            return self._run_step(x, y)

        if self._graph is None:
            # Allocate static tensors and capture
            self._static_x = x.clone()
            self._static_y = y.clone()
            self._graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(self._graph):
                    self._static_loss = self._run_step(self._static_x, self._static_y)
            except RuntimeError as exc:
                import warnings

                warnings.warn(
                    f"CUDA graph capture failed ({exc}); falling back to eager mode.",
                    stacklevel=2,
                )
                self._graph = None
                self.enabled = False
                return self._run_step(x, y)

        # Replay: copy new data into static buffers, replay graph
        self._static_x.copy_(x)
        self._static_y.copy_(y)
        self._graph.replay()
        return self._static_loss

    def reset(self) -> None:
        """Discard the captured graph so it will be re-captured on the next call."""
        self._graph = None
        self._step_count = 0
