# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Deep Operator Network (DeepONet) — Lu et al., 2021."""

from typing import List

import torch
import torch.nn as nn

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module


_ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def _build_mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, act: str) -> nn.Sequential:
    act_cls = _ACT_MAP[act]
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden), act_cls()]
    for _ in range(n_layers - 2):
        layers += [nn.Linear(hidden, hidden), act_cls()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepONet(Module):
    """Deep Operator Network.

    Consists of a *branch net* that encodes function evaluations at fixed
    sensor points, and a *trunk net* that encodes query coordinates.  The
    operator output is the dot product of branch and trunk outputs plus a
    scalar bias.

    Parameters
    ----------
    n_sensors : int
        Number of sensor points where the input function is evaluated.
    n_query_dim : int
        Dimensionality of query coordinates (e.g. 1 for time, 2 for (x,y)).
    hidden_features : int
        Width of each hidden layer in both nets.
    n_layers : int
        Total number of linear layers in each net (including input/output).
    p : int
        Output width of both nets (dot-product dimension).
    activation : str
        Activation function: one of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"silu"``.
    bias : bool
        Whether to add a learnable scalar bias term.
    """

    _meta = ModelMetaData(
        name="DeepONet",
        nvp_tags=["operator-learning", "deeponet", "pde"],
        amp=True,
    )

    def __init__(
        self,
        n_sensors: int,
        n_query_dim: int,
        hidden_features: int = 128,
        n_layers: int = 4,
        p: int = 128,
        activation: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__(meta=self._meta)
        assert activation in _ACT_MAP, f"activation must be one of {list(_ACT_MAP)}"
        self.branch = _build_mlp(n_sensors, p, hidden_features, n_layers, activation)
        self.trunk = _build_mlp(n_query_dim, p, hidden_features, n_layers, activation)
        self.bias = nn.Parameter(torch.zeros(1)) if bias else None
        self._capture_init_args(
            n_sensors=n_sensors,
            n_query_dim=n_query_dim,
            hidden_features=hidden_features,
            n_layers=n_layers,
            p=p,
            activation=activation,
            bias=bias,
        )

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Evaluate the operator.

        Parameters
        ----------
        u : torch.Tensor  shape (B, n_sensors)
            Input function values at sensor locations.
        y : torch.Tensor  shape (Q, n_query_dim)
            Query coordinates.

        Returns
        -------
        torch.Tensor  shape (B, Q)
        """
        b = self.branch(u)   # (B, p)
        t = self.trunk(y)    # (Q, p)
        out = b @ t.T        # (B, Q)
        if self.bias is not None:
            out = out + self.bias
        return out
