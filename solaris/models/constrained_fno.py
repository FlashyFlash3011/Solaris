# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Physics-Constrained Fourier Neural Operator.

Standard PINNs *penalise* constraint violations via loss terms — the network
can still produce outputs that break physics, just less so.  ConstrainedFNO
*enforces* constraints at the architecture level: the output satisfies the
specified law exactly, for any network weights.

Supported constraints
---------------------
``"divergence_free"``
    Output velocity field (2 channels) satisfies ∇·u = 0 exactly (periodic
    domain), via Helmholtz projection in Fourier space.  Zero extra params.

``"conservative"``
    The spatial integral of output channel 0 equals that of input channel 0
    (global conservation of mass / energy / volume).  Only meaningful when
    both channels are in the same physical units.

``"dirichlet"``
    Every boundary pixel of the output equals ``boundary_value`` (e.g. a fixed
    ambient temperature on the domain edge).

``"nonnegative"``
    Output is ≥ 0 everywhere (depths, densities, concentrations), via a
    smooth softplus so gradients never vanish.

``"none"``
    No constraint — identical to a standard 2-D FNO.

Use :meth:`ConstrainedFNO.apply_constraint` to re-impose the constraint on an
output that was modified afterwards; :class:`~solaris.models.CoupledOperator`
does this automatically after coupling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module
from solaris.models.fno import FNOBlock2d
from solaris.nn.constraints import (
    BoundaryValueProjection,
    ConservationProjection,
    DivergenceFreeProjection2d,
)
from solaris.nn.norm import UNSET, resolve_norm

CONSTRAINTS = ("divergence_free", "conservative", "dirichlet", "nonnegative", "none")


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
        One of ``"divergence_free"``, ``"conservative"``, ``"dirichlet"``,
        ``"nonnegative"`` or ``"none"``.
    boundary_value : float, optional
        Required for (and only accepted with) ``constraint="dirichlet"``.
    conservation_mode : str, optional
        ``"additive"`` (default) or ``"multiplicative"``; only accepted with
        ``constraint="conservative"``.  See
        :class:`~solaris.nn.ConservationProjection`.
    norm : None or "instance"
        Normalisation inside the Fourier blocks; see :class:`~solaris.models.FNO`.
    """

    _meta = ModelMetaData(
        name="ConstrainedFNO",
        tags=["pde", "operator-learning", "fourier", "physics-constrained"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_layers: int = 4,
        modes: int | list[int] = 12,
        constraint: str = "divergence_free",
        boundary_value: float | None = None,
        conservation_mode: str | None = None,
        norm: str | None = UNSET,
    ) -> None:
        super().__init__(meta=self._meta)
        norm = resolve_norm(norm, "ConstrainedFNO")
        if constraint not in CONSTRAINTS:
            raise ValueError(f"Unknown constraint: {constraint!r}; expected one of {CONSTRAINTS}")
        if boundary_value is not None and constraint != "dirichlet":
            raise ValueError("boundary_value is only used with constraint='dirichlet'")
        if conservation_mode is not None and constraint != "conservative":
            raise ValueError("conservation_mode is only used with constraint='conservative'")
        if isinstance(modes, int):
            modes = [modes, modes]

        self.constraint = constraint

        # FNO backbone (2-D only)
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            [FNOBlock2d(hidden_channels, modes[0], modes[1], norm) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels * 4, out_channels, 1),
        )

        # Constraint layer — none of them has learnable parameters
        if constraint == "divergence_free":
            if out_channels != 2:
                raise ValueError("divergence_free constraint requires out_channels=2 (u_x, u_y)")
            self.constraint_layer: nn.Module = DivergenceFreeProjection2d()
        elif constraint == "conservative":
            self.constraint_layer = ConservationProjection(
                source_channel=0, output_channel=0, mode=conservation_mode or "additive"
            )
        elif constraint == "dirichlet":
            if boundary_value is None:
                raise ValueError("constraint='dirichlet' requires boundary_value")
            self.constraint_layer = BoundaryValueProjection(boundary_value)
        elif constraint == "nonnegative":
            self.constraint_layer = nn.Softplus()
        else:
            self.constraint_layer = nn.Identity()

        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            constraint=constraint,
            boundary_value=boundary_value,
            conservation_mode=conservation_mode,
            norm=norm,
        )

    def apply_constraint(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Project ``out`` back onto this model's constraint (``x`` = model input).

        Used after ``out`` has been modified downstream (e.g. by coupling).
        Every projection is idempotent: applying it to an output that already
        satisfies the constraint leaves it unchanged.  For ``"nonnegative"``
        the projection is ``relu`` (the forward pass uses a smooth softplus).
        """
        if self.constraint == "conservative":
            return self.constraint_layer(x, out)
        if self.constraint == "nonnegative":
            return F.relu(out)
        return self.constraint_layer(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        out = self.proj(h)
        if self.constraint == "conservative":
            return self.constraint_layer(x, out)
        return self.constraint_layer(out)
