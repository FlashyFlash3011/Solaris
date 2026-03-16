# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from solaris.core import Module, ModelMetaData, ModelRegistry
from solaris.models.mlp import FullyConnected


class DummyModel(Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__(meta=ModelMetaData(name="Dummy"))
        self.linear = torch.nn.Linear(size, size)
        self._capture_init_args(size=size)

    def forward(self, x):
        return self.linear(x)


def test_module_device():
    model = DummyModel()
    assert model.device == torch.device("cpu")


def test_module_num_parameters():
    model = DummyModel(size=8)
    # 8*8 weights + 8 bias = 72
    assert model.num_parameters() == 72


def test_module_save_load(tmp_path):
    model = DummyModel(size=8)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = DummyModel.load(path, map_location="cpu")
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_fully_connected_forward():
    model = FullyConnected(in_features=4, out_features=2, hidden_features=16, n_layers=2)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 2)


def test_model_metadata():
    meta = ModelMetaData(name="Test", nvp_tags=["pde"])
    assert meta.name == "Test"
    assert "pde" in meta.nvp_tags
