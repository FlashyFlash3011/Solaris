# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Common metrics for physics-informed and operator-learning models."""

import torch


def relative_l2_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Relative L2 error ``‖û − u‖₂ / (‖u‖₂ + ε)``, computed per sample.

    The first dimension is the batch.  Each sample's error is relative to its
    own norm, the convention of the neural-operator literature (e.g. the FNO
    paper's L2 loss), so a large sample does not dominate small ones.

    Parameters
    ----------
    reduction :
        ``"mean"`` (default) — average of the per-sample errors;
        ``"none"`` — one value per sample, shape ``(B,)``;
        ``"pooled"`` — one ratio over the whole batch, ``‖û − u‖ / ‖u‖``
        across all samples (the behaviour of Solaris < 0.2).
    """
    if reduction == "pooled":
        return torch.linalg.norm(pred - target) / (torch.linalg.norm(target) + eps)
    if reduction not in ("mean", "none"):
        raise ValueError(f"reduction must be 'mean', 'none' or 'pooled', got {reduction!r}")
    diff = (pred - target).reshape(pred.shape[0], -1)
    per_sample = torch.linalg.norm(diff, dim=1) / (
        torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1) + eps
    )
    return per_sample.mean() if reduction == "mean" else per_sample


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((pred - target) ** 2))


def nrmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalised RMSE (divided by target range)."""
    target_range = target.max() - target.min()
    return rmse(pred, target) / (target_range + eps)


def r2_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Coefficient of determination R²."""
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + eps)
