# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from solaris.core import ModelRegistry


def test_registry_is_singleton():
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2


def test_registry_contains_known_models():
    registry = ModelRegistry()
    # These are registered in pyproject.toml entry points
    known = [
        "fno", "afno", "mlp", "wno", "uno", "deeponet",
        "constrained_fno", "multiscale_fno", "coupled_operator",
        "conformal", "residual_corrector", "meshgraphnet",
    ]
    registered = registry.list_models()
    for name in known:
        assert name in registered, f"'{name}' not found in registry"


def test_registry_lookup_returns_class():
    registry = ModelRegistry()
    FNO = registry["fno"]
    assert callable(FNO)
    # Should be instantiable
    import torch
    model = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    out = model(torch.randn(1, 1, 16, 16))
    assert out.shape == (1, 1, 16, 16)


def test_registry_missing_key_raises():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        _ = registry["does_not_exist_xyz"]


def test_registry_list_models():
    registry = ModelRegistry()
    names = registry.list_models()
    assert len(names) >= 10  # at least 10 registered models


def test_registry_manual_register():
    """Manually registered classes are accessible by key."""
    from solaris.core import Module

    class DummyModel(Module):
        def forward(self, x):
            return x

    registry = ModelRegistry()
    registry.register("dummy_test_model", DummyModel)
    assert registry["dummy_test_model"] is DummyModel
