# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from physicsnemo.models import FNO, AFNO, FullyConnected, MeshGraphNet


@pytest.mark.parametrize("dim,shape", [
    (1, (2, 3, 64)),
    (2, (2, 3, 32, 32)),
    (3, (2, 3, 16, 16, 16)),
])
def test_fno_forward(dim, shape):
    model = FNO(in_channels=3, out_channels=1, hidden_channels=16, n_layers=2, modes=4, dim=dim)
    x = torch.randn(*shape)
    out = model(x)
    expected = list(shape)
    expected[1] = 1
    assert list(out.shape) == expected


def test_afno_forward():
    model = AFNO(
        in_channels=2, out_channels=1,
        img_size=(32, 32), patch_size=4,
        hidden_size=64, n_layers=2, num_blocks=4,
    )
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_meshgraphnet_forward():
    n_nodes, n_edges = 20, 50
    model = MeshGraphNet(node_feat_dim=4, edge_feat_dim=3, out_dim=2, hidden_dim=16, n_layers=2)
    node_feat = torch.randn(n_nodes, 4)
    edge_feat = torch.randn(n_edges, 3)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    out = model(node_feat, edge_feat, edge_index)
    assert out.shape == (n_nodes, 2)


def test_fno_parameter_count():
    model = FNO(in_channels=2, out_channels=1, hidden_channels=16, n_layers=2, modes=4, dim=2)
    assert model.num_parameters() > 0
