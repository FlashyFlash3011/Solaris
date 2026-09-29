# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Training utility classes: early stopping, gradient clipping, graph capture, combined optimiser."""  # noqa: E501

from __future__ import annotations

import math
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

    Call :meth:`step` once at the *end* of each epoch.  The learning rate is a
    closed-form function of the number of completed epochs ``e``:

    * warmup (``e < warmup_epochs``): ``base_lr * (e + 1) / warmup_epochs``, so
      the very first epoch already trains at ``base_lr / warmup_epochs``;
    * then cosine decay from ``base_lr`` to ``min_lr``, reaching ``min_lr`` on
      the last epoch and staying there afterwards.

    ``warmup_epochs=0`` starts directly at ``base_lr``.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of epochs of linear warmup.
    total_epochs : int
        Total training epochs (including warmup).
    base_lr : float
        Peak learning rate (after warmup).
    min_lr : float
        Learning rate at the end of the cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 0.0,
    ) -> None:
        if warmup_epochs < 0 or total_epochs < 1:
            raise ValueError("warmup_epochs must be >= 0 and total_epochs >= 1")
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self._epoch = 0
        self._apply(self.lr_at(0))

    def lr_at(self, epoch: int) -> float:
        """Learning rate used during epoch index ``epoch`` (0-based)."""
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        decay_epochs = self.total_epochs - self.warmup_epochs
        if decay_epochs <= 1:
            return self.base_lr if epoch < self.total_epochs else self.min_lr
        progress = min(1.0, (epoch - self.warmup_epochs) / (decay_epochs - 1))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def _apply(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            if torch.is_tensor(pg["lr"]):
                pg["lr"].fill_(lr)  # tensor LRs (e.g. under graph capture) update in place
            else:
                pg["lr"] = lr

    def step(self) -> float:
        """Advance one epoch and return the learning rate for the next epoch."""
        self._epoch += 1
        lr = self.lr_at(self._epoch)
        self._apply(lr)
        return lr

    @property
    def last_lr(self) -> float:
        """Current learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._epoch = int(state["epoch"])
        self._apply(self.lr_at(self._epoch))


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
    """Speed up a training step with CUDA/HIP graph capture.

    The first ``warmup_steps`` calls run normally (on a side stream, as graph
    capture requires).  The next call records forward + backward + optimiser
    step into a graph; every later call copies the new batch into static
    input buffers and replays the graph, which removes the CPU cost of
    launching each kernel.  The gain is largest for small models / batches
    whose step time is dominated by kernel-launch overhead; measure it for
    your workload (``benchmarks/static_capture.py``).

    Works with the standard optimisers: Adam/AdamW (and other optimisers with
    a ``capturable`` option) are switched to their graph-safe implementation,
    which performs the same arithmetic with the step counter kept on the GPU.

    Falls back to ordinary eager steps when:

    * the batch is on the CPU, or ``enabled=False``;
    * capture fails (e.g. data-dependent control flow) — a warning is issued.

    **Graph requirements** (CUDA and ROCm/HIP alike):

    * Every batch must have the same shape, dtype and device.
    * No Python-side branching that changes between steps.
    * Keep the model in ``train()`` mode; call :meth:`reset` after changing
      the learning rate schedule logic, model mode, or batch shape.

    Learning-rate schedulers keep working after capture for optimisers with a
    ``capturable`` option (Adam, AdamW, ...): their LR is moved into a device
    tensor that the graph reads on every replay, and PyTorch schedulers and
    :class:`WarmupCosineScheduler` update it in place.  Optimisers without
    that option (e.g. SGD) have their LR fixed at capture time; call
    :meth:`reset` after changing it.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer or CombinedOptimizer
    loss_fn : callable
        ``loss_fn(pred, target) -> Tensor``.
    warmup_steps : int
        Number of eager steps before capture (default 20, minimum 1).
    enabled : bool
        Set ``False`` to disable graph capture entirely (useful for debugging).

    Example
    -------
    ::

        capturer = StaticCaptureTraining(model, optimizer, loss_fn=F.mse_loss)

        for x, y in dataloader:
            loss = capturer(x.cuda(), y.cuda())
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
        self.warmup_steps = max(1, warmup_steps)
        self.enabled = enabled and torch.cuda.is_available()
        self._graph: Any = None
        self._step_count = 0
        self._side_stream: Any = None
        self._static_x: torch.Tensor | None = None
        self._static_y: torch.Tensor | None = None
        self._static_loss: torch.Tensor | None = None

    @property
    def captured(self) -> bool:
        """True once a graph has been recorded and is being replayed."""
        return self._graph is not None

    def _run_step(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Execute one normal (non-captured) training step."""
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        return loss

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        return list(getattr(self.optimizer, "optimizers", [self.optimizer]))

    def _make_capturable(self) -> None:
        """Switch optimisers with a ``capturable`` option to graph-safe mode."""
        for opt in self._optimizers():
            for group in opt.param_groups:
                if "capturable" not in group or group["capturable"]:
                    continue
                group["capturable"] = True
                # Keep the LR in a device tensor so a graph replay reads the
                # current value (a Python float would be frozen at capture).
                # Schedulers update tensor LRs in place.
                if not torch.is_tensor(group["lr"]):
                    device = group["params"][0].device
                    group["lr"] = torch.tensor(float(group["lr"]), device=device)
                for p in group["params"]:
                    st = opt.state.get(p, {})
                    if torch.is_tensor(st.get("step")):
                        st["step"] = st["step"].to(device=p.device, dtype=torch.float32)

    def _capture(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self._make_capturable()
        self._static_x = x.clone()
        self._static_y = y.clone()
        # Gradients must be None so the captured backward allocates them inside
        # the graph's memory pool and each replay overwrites (not accumulates).
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.current_stream().wait_stream(self._side_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            loss = self.loss_fn(self.model(self._static_x), self._static_y)
            loss.backward()
            self.optimizer.step()
        self._static_loss = loss
        self._graph = graph

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Run one training step; returns the (scalar) loss tensor.

        After capture the returned tensor is a static buffer that the next
        call overwrites — call ``.item()`` or ``.clone()`` to keep it.
        """
        if not (self.enabled and x.is_cuda):
            return self._run_step(x, y)

        if self._graph is None:
            self._step_count += 1
            if self._side_stream is None:
                self._side_stream = torch.cuda.Stream()
            if self._step_count <= self.warmup_steps:
                # Warm-up on a side stream, as required before graph capture.
                self._side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._side_stream):
                    loss = self._run_step(x, y)
                torch.cuda.current_stream().wait_stream(self._side_stream)
                return loss
            try:
                self._capture(x, y)
            except RuntimeError as exc:
                import warnings

                warnings.warn(
                    f"CUDA graph capture failed ({exc}); falling back to eager mode.",
                    stacklevel=2,
                )
                self._graph = None
                self.enabled = False
                self.optimizer.zero_grad(set_to_none=True)
                return self._run_step(x, y)

        self._static_x.copy_(x)
        self._static_y.copy_(y)
        self._graph.replay()
        return self._static_loss

    def reset(self) -> None:
        """Discard the captured graph so it will be re-captured on the next call."""
        self._graph = None
        self._step_count = 0
