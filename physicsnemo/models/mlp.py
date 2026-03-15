# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fully-connected (MLP) baseline model."""

from typing import List, Optional, Type

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn.activations import Stan


class FullyConnected(Module):
    """Multi-layer perceptron for physics-informed applications.

    Supports standard activations plus the learnable ``Stan`` activation.

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    hidden_features : int
        Width of each hidden layer.
    n_layers : int
        Number of hidden layers.
    activation : str
        One of ``"relu"``, ``"gelu"``, ``"tanh"``, ``"silu"``, ``"stan"``.
    """

    _meta = ModelMetaData(name="FullyConnected", nvp_tags=["pinn", "mlp"])

    _ACT_MAP = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 512,
        n_layers: int = 6,
        activation: str = "silu",
    ) -> None:
        super().__init__(meta=self._meta)
        layers: List[nn.Module] = []
        in_dim = in_features
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_features))
            if activation == "stan":
                layers.append(Stan(hidden_features))
            else:
                layers.append(self._ACT_MAP[activation]())
            in_dim = hidden_features
        layers.append(nn.Linear(in_dim, out_features))
        self.net = nn.Sequential(*layers)
        self._capture_init_args(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            n_layers=n_layers,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
