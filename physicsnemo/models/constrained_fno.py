# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Physics-Constrained Fourier Neural Operator.

Standard PINNs *penalise* constraint violations via loss terms — the network
can still produce outputs that break physics, just less so.  ConstrainedFNO
*enforces* constraints at the architecture level: the output is mathematically
guaranteed to satisfy the specified physical law regardless of network weights.

Supported constraints
---------------------
``"divergence_free"``
    Output velocity field (2 channels) satisfies ∇·u = 0 exactly.
    Enforced via Helmholtz decomposition in Fourier space.  Zero extra params.

``"conservative"``
    Output field has the same total spatial integral as the input, enforcing
    global conservation of mass / energy.  Zero extra params.

``"none"``
    No constraint — identical to standard FNO.
"""

from typing import List

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.models.fno import FNOBlock2d
from physicsnemo.nn.constraints import ConservationProjection, DivergenceFreeProjection2d


class ConstrainedFNO(Module):
    """Fourier Neural Operator with hard physics constraint enforcement.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    hidden_channels : int
    n_layers : int
    modes : int or list[int]
        Fourier modes per spatial dimension (2-D only).
    constraint : str
        One of ``"divergence_free"``, ``"conservative"``, or ``"none"``.
    """

    _meta = ModelMetaData(
        name="ConstrainedFNO",
        nvp_tags=["pde", "operator-learning", "fourier", "physics-constrained"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int | List[int] = 12,
        constraint: str = "divergence_free",
    ) -> None:
        super().__init__(meta=self._meta)
        if isinstance(modes, int):
            modes = [modes, modes]

        self.constraint = constraint

        # FNO backbone (2-D only)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(hidden_channels, modes[0], modes[1]) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, out_channels, 1),
        )

        # Constraint layer — zero learnable parameters for "divergence_free"
        if constraint == "divergence_free":
            assert out_channels == 2, (
                "divergence_free constraint requires out_channels=2 (u_x, u_y)"
            )
            self.constraint_layer: nn.Module = DivergenceFreeProjection2d()
        elif constraint == "conservative":
            self.constraint_layer = ConservationProjection(
                source_channel=0, output_channel=0
            )
        elif constraint == "none":
            self.constraint_layer = nn.Identity()
        else:
            raise ValueError(f"Unknown constraint: {constraint!r}")

        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            constraint=constraint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = x  # keep reference for conservative constraint

        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        out = self.proj(h)

        if self.constraint == "divergence_free":
            out = self.constraint_layer(out)
        elif self.constraint == "conservative":
            out = self.constraint_layer(source, out)
        # "none" — no-op

        return out
