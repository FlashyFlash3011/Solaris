# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load utilities.

Provides two checkpoint strategies:

``save_checkpoint`` / ``load_checkpoint``
    Standard single-process or DDP training.  Wraps ``torch.save`` /
    ``torch.load``.

``save_checkpoint_distributed`` / ``load_checkpoint_distributed``
    FSDP / DTensor sharded training.  Uses
    ``torch.distributed.checkpoint`` (DCP) which writes one shard file per
    rank and reconstructs on load regardless of the process count used to save.
    Falls back automatically to the standard strategy when the model is not
    wrapped in FSDP or DTensor.
"""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    epoch: int = 0,
    loss: float = float("inf"),
    extra: dict[str, Any] | None = None,
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
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
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


def _is_fsdp(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except ImportError:
        return False


def save_checkpoint_distributed(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    epoch: int = 0,
    loss: float = float("inf"),
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a sharded checkpoint using ``torch.distributed.checkpoint`` (DCP).

    When *model* is wrapped in FSDP or uses DTensor, DCP writes one shard file
    per rank under *path* (treated as a directory).  The checkpoint can be
    loaded with any number of ranks via :func:`load_checkpoint_distributed`.

    Falls back to :func:`save_checkpoint` (standard ``torch.save``) when the
    model is not FSDP-wrapped or when DCP is unavailable.

    Parameters
    ----------
    path : str or Path
        Destination directory (DCP) or file path (fallback).
    model, optimizer, scheduler, epoch, loss, extra :
        Same as :func:`save_checkpoint`.
    """
    if not _is_fsdp(model):
        save_checkpoint(path, model, optimizer, scheduler, epoch, loss, extra)
        return

    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
    except ImportError:
        save_checkpoint(path, model, optimizer, scheduler, epoch, loss, extra)
        return

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_state, optim_state = get_state_dict(model, optimizer)
    state: dict[str, Any] = {
        "model": model_state,
        "epoch": epoch,
        "loss": loss,
    }
    if optim_state:
        state["optimizer"] = optim_state
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)

    dcp.save(state, checkpoint_id=str(path))


def load_checkpoint_distributed(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a DCP sharded checkpoint into an FSDP / DTensor model.

    Falls back to :func:`load_checkpoint` for non-FSDP models or when DCP is
    unavailable.

    Parameters
    ----------
    path : str or Path
        DCP checkpoint directory (or standard ``.pt`` file for fallback).
    model, optimizer, scheduler, map_location :
        Same as :func:`load_checkpoint`.
    """
    if not _is_fsdp(model):
        return load_checkpoint(path, model, optimizer, scheduler, map_location)

    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict
    except ImportError:
        return load_checkpoint(path, model, optimizer, scheduler, map_location)

    model_state, optim_state = {}, {}
    state: dict[str, Any] = {"model": model_state}
    if optimizer is not None:
        state["optimizer"] = optim_state

    dcp.load(state, checkpoint_id=str(path))

    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optim_state if optimizer is not None else {},
    )

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return state
