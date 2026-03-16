# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Embedding layers for physics-informed and generative models."""

import math
import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """Random Fourier features for coordinate-based networks.

    Maps input coordinates to a higher-dimensional space using random
    Fourier features, enabling networks to learn high-frequency functions.

    Parameters
    ----------
    in_features : int
        Dimensionality of input coordinates.
    embed_dim : int
        Output embedding dimension (must be even).
    scale : float
        Standard deviation of the Gaussian used to sample frequencies.
    """

    def __init__(self, in_features: int, embed_dim: int, scale: float = 1.0) -> None:
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        B = torch.randn(in_features, embed_dim // 2) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (..., in_features)
        Returns
        -------
        Tensor, shape (..., embed_dim)
        """
        proj = x @ self.B  # (..., embed_dim // 2)
        return torch.cat([torch.cos(2 * math.pi * proj), torch.sin(2 * math.pi * proj)], dim=-1)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (Vaswani et al., 2017).

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must be even).
    max_len : int
        Maximum sequence length.
    """

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embedding to x. x: (batch, seq_len, embed_dim)."""
        return x + self.pe[:, : x.size(1)]


class SinusoidalTimestepEmbedding(nn.Module):
    """Timestep embedding used in diffusion models.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension.
    max_period : int
        Controls the minimum frequency of the embeddings.
    """

    def __init__(self, embed_dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        timesteps : Tensor, shape (batch,) — continuous or integer timesteps.
        Returns
        -------
        Tensor, shape (batch, embed_dim)
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float, device=timesteps.device) / half
        )
        args = timesteps[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
