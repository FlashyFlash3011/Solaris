# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Custom activation functions for physics-informed networks."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CappedGELU(nn.Module):
    """GELU activation capped at a maximum value to improve training stability."""

    def __init__(self, cap_value: float = 10.0) -> None:
        super().__init__()
        self.cap = cap_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.gelu(x), max=self.cap)


class CappedLeakyReLU(nn.Module):
    """LeakyReLU capped at *cap_value* to prevent activation explosion."""

    def __init__(self, cap_value: float = 10.0, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.cap = cap_value
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.leaky_relu(x, self.negative_slope), max=self.cap)


class Stan(nn.Module):
    """Self-scalable Tanh (Stan) activation for PINNs.

    Reference: Gnanasambandam et al., 2022.
    Stan(x) = x * tanh(beta * x),  beta is a learnable parameter per neuron.
    """

    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(self.beta * x)


class Siren(nn.Module):
    """Sine activation used in SIREN networks (Sitzmann et al., 2020).

    Parameters
    ----------
    omega_0 : float
        Frequency multiplier. Typically 30.0 for the first layer, 1.0 for hidden.
    """

    def __init__(self, omega_0: float = 30.0) -> None:
        super().__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)

    @staticmethod
    def init_weights(linear: nn.Linear, omega_0: float, is_first: bool) -> None:
        """In-place weight initialisation following the SIREN paper."""
        with torch.no_grad():
            if is_first:
                linear.weight.uniform_(-1 / linear.in_features, 1 / linear.in_features)
            else:
                bound = math.sqrt(6 / linear.in_features) / omega_0
                linear.weight.uniform_(-bound, bound)
