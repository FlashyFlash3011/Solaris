# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Conformal Neural Operator — statistically rigorous uncertainty quantification.

Wraps *any* PhysicsNeMo model with split conformal prediction to produce
calibrated uncertainty intervals with a user-specified coverage guarantee:

    P(y ∈ [ŷ_low, ŷ_high]) ≥ 1 − α

This guarantee holds for **any model** and **any data distribution** with no
distributional assumptions.  The only requirement is that the calibration and
test data are exchangeable (drawn from the same distribution).

Compared to Bayesian neural networks or Monte-Carlo dropout, conformal
prediction requires no changes to model training — just a held-out calibration
set (typically 10–20 % of total data).

How it works
------------
1. Compute *nonconformity scores* on the calibration set:
       score_i = max pixel-wise |ŷ_i − y_i|
2. Compute the finite-sample-corrected (1-α) quantile q̂ of those scores.
3. At test time, output intervals: [ŷ − q̂,  ŷ + q̂]

Step 2 uses the standard Venn–Abers finite-sample correction so that the
coverage guarantee holds even for small calibration sets:
       level = (1 − α)(1 + 1/n_cal)  [clamped to 1]

References
----------
Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2021.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ConformalNeuralOperator(nn.Module):
    """Conformal prediction wrapper for any neural operator.

    Usage
    -----
    ::

        model = FNO(...)        # train normally
        predictor = ConformalNeuralOperator(model)
        predictor.calibrate(cal_inputs, cal_targets, alpha=0.1)
        lo, hi, pt = predictor.predict(test_input)
        # Guaranteed: true solution ∈ [lo, hi] for ≥ 90 % of test samples

    Parameters
    ----------
    model : nn.Module
        Any trained PhysicsNeMo model.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # q_hat is the calibrated quantile threshold; inf until calibrated
        self.register_buffer("_q_hat", torch.tensor(float("inf")))
        self.register_buffer("_calibrated_buf", torch.zeros(1, dtype=torch.bool))

    @property
    def is_calibrated(self) -> bool:
        return bool(self._calibrated_buf.item())

    @torch.no_grad()
    def calibrate(
        self,
        cal_inputs: torch.Tensor,
        cal_targets: torch.Tensor,
        alpha: float = 0.1,
        batch_size: int = 32,
    ) -> float:
        """Compute and store the conformal quantile from a calibration set.

        Parameters
        ----------
        cal_inputs : torch.Tensor  shape (N, ...)
        cal_targets : torch.Tensor  shape (N, ...)
        alpha : float
            Desired miscoverage rate.  alpha=0.1 → 90 % coverage guarantee.
        batch_size : int
            Mini-batch size for inference (avoids OOM on large cal sets).

        Returns
        -------
        float
            The calibrated threshold q̂ (stored internally for ``predict``).
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        scores = []
        for i in range(0, len(cal_inputs), batch_size):
            x_b = cal_inputs[i : i + batch_size].to(device)
            y_b = cal_targets[i : i + batch_size].to(device)
            pred = self.model(x_b)
            # Nonconformity score: worst-case absolute error per sample
            score = (pred - y_b).abs().flatten(1).max(dim=1).values
            scores.append(score.cpu())

        scores = torch.cat(scores)  # (N,)
        n_cal = len(scores)

        # Finite-sample corrected quantile (Venn-Abers)
        level = torch.tensor((1.0 - alpha) * (1.0 + 1.0 / n_cal)).clamp(max=1.0)
        q_hat = torch.quantile(scores, level)

        self._q_hat = q_hat.to(device)
        self._calibrated_buf.fill_(True)
        return q_hat.item()

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with calibrated uncertainty intervals.

        Parameters
        ----------
        x : torch.Tensor  shape (B, ...)

        Returns
        -------
        lower : torch.Tensor   point_pred − q̂
        upper : torch.Tensor   point_pred + q̂
        point_pred : torch.Tensor   raw model output
        """
        if not self.is_calibrated:
            raise RuntimeError(
                "Model not calibrated. Call calibrate(cal_inputs, cal_targets) first."
            )
        self.model.eval()
        point_pred = self.model(x)
        return point_pred - self._q_hat, point_pred + self._q_hat, point_pred

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass — point prediction only."""
        return self.model(x)

    @torch.no_grad()
    def coverage_report(
        self,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        batch_size: int = 32,
    ) -> dict:
        """Measure empirical coverage on a test set.

        Returns
        -------
        dict with keys:
            ``coverage``            — fraction of samples where all pixels are covered
            ``mean_interval_width`` — average (hi − lo) value per pixel
            ``q_hat``               — the calibrated threshold
        """
        device = next(self.model.parameters()).device
        covered, widths = [], []

        for i in range(0, len(test_inputs), batch_size):
            x_b = test_inputs[i : i + batch_size].to(device)
            y_b = test_targets[i : i + batch_size].to(device)
            lo, hi, _ = self.predict(x_b)
            covered.append(
                ((y_b >= lo) & (y_b <= hi)).flatten(1).all(dim=1).float().cpu()
            )
            widths.append((hi - lo).flatten(1).mean(dim=1).cpu())

        return {
            "coverage": torch.cat(covered).mean().item(),
            "mean_interval_width": torch.cat(widths).mean().item(),
            "q_hat": self._q_hat.item(),
        }
