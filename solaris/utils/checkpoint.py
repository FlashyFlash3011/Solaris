# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load utilities.

Provides two checkpoint strategies:

``save_checkpoint`` / ``load_checkpoint``
    Standard single-process or DDP training.  Wraps ``torch.save`` /
    ``torch.load``.

``save_checkpoint_distributed`` / ``load_checkpoint_distributed``
    FSDP sharded training.  Uses ``torch.distributed.checkpoint`` (DCP),
    which writes each rank's shards and reloads them regardless of the number
    of ranks used to save.  Falls back automatically to the standard strategy
    when the model is not wrapped in FSDP.
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

    When *model* is wrapped in FSDP, every rank writes its own shards under the
    directory *path* (call this on **all** ranks).  Model and optimizer state
    go through DCP, so the checkpoint can be loaded with a different number of
    ranks; epoch, loss, scheduler state and ``extra`` are written by rank 0 to
    ``path/meta.pt``.

    Falls back to :func:`save_checkpoint` (standard ``torch.save`` file) when
    the model is not FSDP-wrapped.

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

    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    model_state, optim_state = get_state_dict(model, optimizer if optimizer is not None else [])
    state: dict[str, Any] = {"model": model_state}
    if optimizer is not None:
        state["optimizer"] = optim_state
    dcp.save(state, checkpoint_id=str(path))

    if not dist.is_initialized() or dist.get_rank() == 0:
        meta: dict[str, Any] = {"epoch": epoch, "loss": loss}
        if scheduler is not None:
            meta["scheduler_state_dict"] = scheduler.state_dict()
        if extra:
            meta.update(extra)
        torch.save(meta, path / "meta.pt")
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint_distributed(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint_distributed`.

    Call on **all** ranks with the FSDP-wrapped *model* (and optimizer).  DCP
    loads in place into tensors shaped like the current model's shards, so the
    checkpoint may come from a run with a different number of ranks.

    Falls back to :func:`load_checkpoint` for non-FSDP models.

    Returns
    -------
    dict
        The metadata (``epoch``, ``loss``, ``scheduler_state_dict`` if saved,
        and any ``extra`` entries).
    """
    if not _is_fsdp(model):
        return load_checkpoint(path, model, optimizer, scheduler, map_location)

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    path = Path(path)
    # Templates with the right shapes/sharding; DCP fills them in place.
    model_state, optim_state = get_state_dict(model, optimizer if optimizer is not None else [])
    state: dict[str, Any] = {"model": model_state}
    if optimizer is not None:
        state["optimizer"] = optim_state
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(
        model,
        optimizer if optimizer is not None else [],
        model_state_dict=state["model"],
        optim_state_dict=state.get("optimizer", {}),
    )

    meta_path = path / "meta.pt"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    if scheduler is not None and "scheduler_state_dict" in meta:
        scheduler.load_state_dict(meta["scheduler_state_dict"])
    return meta
