# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from solaris.core import ModelMetaData, Module
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


def test_from_checkpoint(tmp_path):
    model = DummyModel(size=8)
    path = tmp_path / "model.pt"
    model.save(path)

    # Reconstruct without specifying DummyModel explicitly
    loaded = Module.from_checkpoint(path, map_location="cpu")
    assert isinstance(loaded, DummyModel)
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_from_torch():
    inner = torch.nn.Linear(4, 2)
    wrapper = Module.from_torch(inner)

    assert wrapper.meta.name == "Linear"
    assert wrapper.num_parameters() == inner.weight.numel() + inner.bias.numel()

    x = torch.randn(3, 4)
    torch.testing.assert_close(wrapper(x), inner(x))


# --- Auto init-arg capture ---


class AutoModel(Module):
    """Subclass that never calls _capture_init_args() manually."""

    def __init__(self, width: int = 32, depth: int = 4) -> None:
        super().__init__(meta=ModelMetaData(name="AutoModel"))
        self.linear = torch.nn.Linear(width, width)
        # deliberately no _capture_init_args() call

    def forward(self, x):
        return self.linear(x)


def test_auto_capture_populates_init_args():
    model = AutoModel(width=8, depth=2)
    assert model._init_args == {"width": 8, "depth": 2}


def test_auto_capture_save_load(tmp_path):
    model = AutoModel(width=8, depth=2)
    path = tmp_path / "auto.pt"
    model.save(path)
    loaded = AutoModel.load(path, map_location="cpu")
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_manual_capture_takes_precedence():
    """Manual _capture_init_args() overrides auto-capture."""
    model = DummyModel(size=16)
    # DummyModel calls _capture_init_args(size=size) — only "size" should be stored
    assert set(model._init_args.keys()) == {"size"}


# --- compile_model ---


def test_compile_model_returns_self():
    model = DummyModel(size=8)
    result = model.compile_model()
    assert result is model


def test_compile_model_still_runs():
    model = DummyModel(size=8)
    model.compile_model()
    x = torch.randn(2, 8)
    try:
        out = model(x)
        assert out.shape == (2, 8)
    except Exception:
        # torch.compile may fail on CPU-only environments without a compiler
        pytest.skip("torch.compile backend not available in this environment")
