# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-physics coupling framework.

Composes multiple neural operators into a coupled system where different
physical fields interact.  In real engineering problems, physics domains do
not live in isolation — chip temperature drives thermal expansion, which
deforms the geometry, which changes the fluid flow, which changes the cooling.
This module provides a systematic API for that kind of coupling.

Coupling modes
--------------
``"sequential"``
    Each operator's output is concatenated with the next operator's input.
    Simple, no extra parameters, captures one-directional influence.

``"learned"``
    A ``LearnedCouplingLayer`` discovers which fields influence which others
    via a soft gating matrix W[i,j] ∈ (0,1): 'how much does field j affect
    field i?'  Initialised near zero so training starts from the uncoupled
    (independent operator) baseline and gradually learns inter-field coupling.

``"direct"``
    Operators run independently; outputs are returned without modification.
    Useful as an ablation baseline.

Example
-------
::

    coupled = CoupledOperator(
        operators={
            "thermal": FNO(in_channels=1, out_channels=1, ...),
            "fluid":   ConstrainedFNO(in_channels=2, out_channels=2, ...),
        },
        coupling_channels={"thermal": 1, "fluid": 2},
        coupling_mode="learned",
        n_coupling_steps=2,
    )
    outputs = coupled({"thermal": power_map, "fluid": velocity})
    T_pred  = outputs["thermal"]
    u_pred  = outputs["fluid"]
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


class LearnedCouplingLayer(nn.Module):
    """Learned coupling between N physical fields via a gated interaction matrix.

    The coupling matrix W[i,j] (shape n_fields × n_fields, values in (0,1))
    represents how strongly field j influences field i.  Because it is
    initialised near zero, the uncoupled solution is the training baseline;
    non-zero coupling only emerges if it reduces the loss.

    Parameters
    ----------
    n_fields : int
    spatial_channels : int
        Spatial feature channels at the coupling interface.
    """

    def __init__(self, n_fields: int, spatial_channels: int) -> None:
        super().__init__()
        self.n_fields = n_fields

        # Soft coupling matrix — initialised near zero (uncoupled baseline)
        self.coupling_weights = nn.Parameter(0.01 * torch.randn(n_fields, n_fields))

        # Lightweight per-field projections at the coupling interface
        self.proj_in = nn.ModuleList(
            [nn.Conv2d(spatial_channels, spatial_channels, 1) for _ in range(n_fields)]
        )
        self.proj_out = nn.ModuleList(
            [nn.Conv2d(spatial_channels, spatial_channels, 1) for _ in range(n_fields)]
        )
        self.act = nn.GELU()

    def forward(self, field_outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        projected = [self.act(self.proj_in[i](f)) for i, f in enumerate(field_outputs)]

        W = torch.sigmoid(self.coupling_weights)  # (n_fields, n_fields)

        coupled = []
        for i in range(self.n_fields):
            out_i = projected[i]
            for j in range(self.n_fields):
                if i != j:
                    # Resize j to match i's spatial size if needed
                    p_j = projected[j]
                    if p_j.shape[-2:] != out_i.shape[-2:]:
                        p_j = F.interpolate(
                            p_j, size=out_i.shape[-2:], mode="bilinear", align_corners=False
                        )
                    out_i = out_i + W[i, j] * p_j
            coupled.append(self.act(self.proj_out[i](out_i)))
        return coupled

    def coupling_matrix(self) -> torch.Tensor:
        """Return the learned coupling strengths (n_fields × n_fields)."""
        return torch.sigmoid(self.coupling_weights).detach()


class CoupledOperator(Module):
    """Multi-physics operator composition with optional learned coupling.

    Parameters
    ----------
    operators : dict[str, nn.Module]
        Mapping from physics domain name to its neural operator.
    coupling_channels : dict[str, int]
        Output channels of each operator (used as coupling interface width).
    coupling_mode : str
        ``"learned"``, ``"sequential"``, or ``"direct"``.
    n_coupling_steps : int
        Number of iterative coupling rounds (only used in ``"learned"`` mode).
    """

    _meta = ModelMetaData(
        name="CoupledOperator",
        nvp_tags=["multi-physics", "coupling", "operator-learning", "pde"],
        amp=False,
    )

    def __init__(
        self,
        operators: Dict[str, nn.Module],
        coupling_channels: Dict[str, int],
        coupling_mode: str = "learned",
        n_coupling_steps: int = 2,
    ) -> None:
        super().__init__(meta=self._meta)
        assert coupling_mode in ("sequential", "direct", "learned"), (
            f"coupling_mode must be one of 'sequential', 'direct', 'learned'; "
            f"got {coupling_mode!r}"
        )

        self.coupling_mode = coupling_mode
        self.n_coupling_steps = n_coupling_steps
        self.field_names = list(operators.keys())
        self.n_fields = len(self.field_names)
        self.operators = nn.ModuleDict(operators)

        if coupling_mode == "learned":
            min_ch = min(coupling_channels.values())
            self.coupling_layers = nn.ModuleList(
                [LearnedCouplingLayer(self.n_fields, min_ch) for _ in range(n_coupling_steps)]
            )
            # Adapters bring all fields to a common channel width for coupling
            self.adapters = nn.ModuleDict(
                {name: nn.Conv2d(ch, min_ch, 1) for name, ch in coupling_channels.items()}
            )
            self.unadapters = nn.ModuleDict(
                {name: nn.Conv2d(min_ch, ch, 1) for name, ch in coupling_channels.items()}
            )

        # operators are not JSON-serialisable
        self._capture_init_args(
            operators={},
            coupling_channels=coupling_channels,
            coupling_mode=coupling_mode,
            n_coupling_steps=n_coupling_steps,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : dict[str, Tensor]
            One input tensor per physics domain.

        Returns
        -------
        dict[str, Tensor]
            Predicted output for each domain, with inter-physics coupling applied.
        """
        # Step 1: independent predictions from each operator
        outputs = {name: self.operators[name](inputs[name]) for name in self.field_names}

        if self.coupling_mode == "sequential":
            # Each operator's output nudges the next operator's prediction
            field_list = self.field_names
            for i in range(1, len(field_list)):
                prev_out = outputs[field_list[i - 1]]
                curr_out = outputs[field_list[i]]
                if prev_out.shape[-2:] != curr_out.shape[-2:]:
                    prev_out = F.interpolate(
                        prev_out, size=curr_out.shape[-2:], mode="bilinear", align_corners=False
                    )
                # Add coupling residual (capped channel overlap)
                n_ch = min(prev_out.shape[1], curr_out.shape[1])
                outputs[field_list[i]] = curr_out + 0.1 * F.pad(
                    prev_out[:, :n_ch], [0, 0, 0, 0, 0, curr_out.shape[1] - n_ch]
                )

        elif self.coupling_mode == "learned":
            # Bring all field outputs to common channel width
            field_feats = [
                self.adapters[name](outputs[name]) for name in self.field_names
            ]
            # Iterative coupling rounds
            for layer in self.coupling_layers:
                field_feats = layer(field_feats)
            # Project back and add as a residual correction
            for i, name in enumerate(self.field_names):
                delta = self.unadapters[name](field_feats[i])
                outputs[name] = outputs[name] + delta

        # "direct" mode: return independent outputs unchanged
        return outputs

    def coupling_strengths(self) -> Optional[torch.Tensor]:
        """Return learned coupling matrix for the final coupling layer.

        Returns ``None`` if not in ``"learned"`` mode.  Entry [i, j] is the
        strength with which field j influences field i (values in (0, 1)).
        """
        if self.coupling_mode != "learned":
            return None
        return self.coupling_layers[-1].coupling_matrix()
