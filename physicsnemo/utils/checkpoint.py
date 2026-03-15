# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load utilities with optional remote (fsspec) support."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: int = 0,
    loss: float = float("inf"),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint.

    Parameters
    ----------
    path : str or Path
        Destination file path (local or ``fsspec``-compatible URI).
    model : nn.Module
        Model to save.
    optimizer : Optimizer, optional
    scheduler : LR scheduler, optional
    epoch : int
    loss : float
    extra : dict, optional
        Any additional data to include in the checkpoint.
    """
    ckpt = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        ckpt.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (+ optionally optimizer/scheduler) state.

    Returns the full checkpoint dict so callers can read ``epoch``, ``loss``, etc.

    Parameters
    ----------
    path : str or Path
        Checkpoint file path.
    model : nn.Module
        Model to load weights into.
    optimizer : Optimizer, optional
    scheduler : optional
    map_location :
        Passed to ``torch.load``. Defaults to ``"cuda"`` if available else ``"cpu"``.
    """
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt
