# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""MeshGraphNet — message-passing GNN for mesh-based physics simulations.

Reference: Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks", ICLR 2021.

Note: edge_index is expected as a (2, E) LongTensor.
      node_features: (N, node_feat_dim)
      edge_features: (E, edge_feat_dim)
"""

from typing import Optional

import torch
import torch.nn as nn

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class EdgeBlock(nn.Module):
    """Edge update: aggregate sender/receiver node features + current edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = _mlp(2 * node_dim + edge_dim, hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        inp = torch.cat([node_feat[src], node_feat[dst], edge_feat], dim=-1)
        return self.norm(self.mlp(inp))


class NodeBlock(nn.Module):
    """Node update: aggregate incoming edge features + current node features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = _mlp(node_dim + edge_dim, hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        _, dst = edge_index
        agg = torch.zeros(n_nodes, edge_feat.size(-1), device=node_feat.device, dtype=node_feat.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_feat), edge_feat)
        inp = torch.cat([node_feat, agg], dim=-1)
        return self.norm(self.mlp(inp))


class MeshGraphNet(Module):
    """Graph Network-based Simulator for mesh-based physics.

    Parameters
    ----------
    node_feat_dim : int
        Dimensionality of input node features.
    edge_feat_dim : int
        Dimensionality of input edge features.
    out_dim : int
        Dimensionality of per-node output (e.g., velocity, pressure).
    hidden_dim : int
        Width of internal MLP layers.
    n_layers : int
        Number of message-passing iterations.
    """

    _meta = ModelMetaData(
        name="MeshGraphNet",
        nvp_tags=["cfd", "gnn", "mesh"],
    )

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 6,
    ) -> None:
        super().__init__(meta=self._meta)
        # Encoders
        self.node_enc = _mlp(node_feat_dim, hidden_dim, hidden_dim)
        self.edge_enc = _mlp(edge_feat_dim, hidden_dim, hidden_dim)
        # Processor (alternating edge/node blocks with residual connections)
        self.edge_blocks = nn.ModuleList([EdgeBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.node_blocks = nn.ModuleList([NodeBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim) for _ in range(n_layers)])
        # Decoder
        self.decoder = _mlp(hidden_dim, hidden_dim, out_dim)
        self._capture_init_args(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim, out_dim=out_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
        )

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        n_nodes = node_feat.size(0)
        v = self.node_enc(node_feat)
        e = self.edge_enc(edge_feat)
        for edge_blk, node_blk in zip(self.edge_blocks, self.node_blocks):
            e = e + edge_blk(v, e, edge_index)
            v = v + node_blk(v, e, edge_index, n_nodes)
        return self.decoder(v)
