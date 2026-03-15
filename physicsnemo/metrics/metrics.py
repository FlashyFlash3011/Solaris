# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Common metrics for physics-informed and operator-learning models."""

import torch


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Relative L2 error (normalised by target norm).

    .. math:: \\|\\hat{u} - u\\|_2 / (\\|u\\|_2 + \\epsilon)
    """
    return torch.linalg.norm(pred - target) / (torch.linalg.norm(target) + eps)


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
