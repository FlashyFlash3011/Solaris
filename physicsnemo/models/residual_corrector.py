# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Neural Residual Corrector — hybrid numerical + neural solver.

Instead of replacing a numerical solver with a neural black box, this module
*corrects* the residual error of a cheap coarse solver:

    final_output = coarse_solver(input) + neural_correction(input, coarse_output)

The neural network only needs to learn what the coarse solver got *wrong* —
a small, smooth correction field that is far easier to learn than the full
physical mapping.  This typically reduces data requirements by 10–100× and
dramatically improves generalisation to out-of-distribution boundary conditions
and geometries.

Key idea
--------
The coarse solver (e.g. 20 Jacobi iterations instead of 10 000) already
captures the dominant physics.  The residual δ = truth − coarse is compact,
lower-amplitude, and has a simpler spectral structure — exactly the kind of
signal a Fourier-based network learns efficiently.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.fno import FNOBlock2d


class NeuralResidualCorrector(Module):
    """Hybrid solver: cheap coarse solution + learned residual correction.

    Architecture
    ------------
    Given problem input x:

    1. ``x_coarse = solver(x)``          — cheap/few-iteration numerical solution
    2. ``z = cat([x, x_coarse], dim=1)`` — concatenate along channel dimension
    3. ``δ = FNO_backbone(z)``           — network predicts the error
    4. ``ŷ = x_coarse + δ``             — corrected output

    Parameters
    ----------
    solver : callable
        Coarse solver.  Must accept ``(x: Tensor) → Tensor`` and return a
        tensor of shape ``(B, solver_out_channels, H, W)``.
    in_channels : int
        Number of input channels (not including solver output).
    out_channels : int
        Number of output channels.
    solver_out_channels : int
        Number of channels returned by the coarse solver.
    hidden_channels : int
    n_layers : int
    modes : int
    solver_detach : bool
        If ``True`` (default), stop gradients through the solver output.
        Set ``False`` only for differentiable solvers.
    """

    _meta = ModelMetaData(
        name="NeuralResidualCorrector",
        nvp_tags=["pde", "hybrid-solver", "residual-learning", "operator-learning"],
        amp=True,
    )

    def __init__(
        self,
        solver: Callable,
        in_channels: int,
        out_channels: int,
        solver_out_channels: int = 1,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int = 12,
        solver_detach: bool = True,
    ) -> None:
        super().__init__(meta=self._meta)
        self.solver = solver
        self.solver_detach = solver_detach
        self.out_channels = out_channels

        # Backbone sees [input | coarse_output] concatenated
        backbone_in = in_channels + solver_out_channels
        self.lift = nn.Conv2d(backbone_in, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(hidden_channels, modes, modes) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, out_channels, 1),
        )

        # solver is not JSON-serialisable; set it manually after load()
        self._capture_init_args(
            solver=None,
            in_channels=in_channels,
            out_channels=out_channels,
            solver_out_channels=solver_out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            solver_detach=solver_detach,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, in_channels, H, W)

        Returns
        -------
        torch.Tensor  shape (B, out_channels, H, W)
            Coarse solution plus learned residual correction.
        """
        # Step 1: coarse solution
        x_coarse = self.solver(x)
        if self.solver_detach:
            x_coarse = x_coarse.detach()

        # Step 2: neural correction — sees both input and coarse output
        z = torch.cat([x, x_coarse], dim=1)
        z = self.lift(z)
        for block in self.blocks:
            z = block(z)
        delta = self.proj(z)

        # Step 3: corrected output
        return x_coarse + delta

    @torch.no_grad()
    def correction_diagnostics(self, x: torch.Tensor) -> dict:
        """Report relative magnitude of the learned correction vs coarse solution.

        Useful for monitoring training — a well-trained corrector should have a
        correction norm that decreases over epochs as the model improves the
        coarse solver's accuracy.
        """
        x_coarse = self.solver(x)
        if self.solver_detach:
            x_coarse = x_coarse.detach()

        z = torch.cat([x, x_coarse], dim=1)
        z = self.lift(z)
        for block in self.blocks:
            z = block(z)
        delta = self.proj(z)

        coarse_norm = x_coarse.norm().item()
        delta_norm = delta.norm().item()
        return {
            "coarse_norm": coarse_norm,
            "correction_norm": delta_norm,
            "relative_correction": delta_norm / (coarse_norm + 1e-8),
        }
