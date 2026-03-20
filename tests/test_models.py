# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
# Tests for FNO, AFNO, ConstrainedFNO, WNO, UNO, DeepONet, and other models

import pytest
import torch

from solaris.models import AFNO, FNO, MeshGraphNet
from solaris.models.conformal import ConformalNeuralOperator
from solaris.models.constrained_fno import ConstrainedFNO
from solaris.models.coupled import CoupledOperator
from solaris.models.deeponet import DeepONet
from solaris.models.multiscale_fno import MultiScaleFNO
from solaris.models.residual_corrector import NeuralResidualCorrector
from solaris.models.uno import UNO
from solaris.models.wno import WNO


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


# --- ConstrainedFNO ---

def test_constrained_fno_divergence_free():
    model = ConstrainedFNO(in_channels=2, out_channels=2, hidden_channels=16, n_layers=2, modes=4, constraint="divergence_free")
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 2, 32, 32)


def test_constrained_fno_conservative():
    model = ConstrainedFNO(in_channels=1, out_channels=1, hidden_channels=16, n_layers=2, modes=4, constraint="conservative")
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_constrained_fno_none():
    model = ConstrainedFNO(in_channels=2, out_channels=3, hidden_channels=16, n_layers=2, modes=4, constraint="none")
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 3, 32, 32)


def test_constrained_fno_invalid_constraint():
    with pytest.raises(ValueError):
        ConstrainedFNO(in_channels=1, out_channels=1, constraint="bad")


# --- WNO ---

def test_wno_forward():
    model = WNO(in_channels=2, out_channels=1, hidden_channels=16, n_layers=2, levels=2)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


def test_wno_non_power_of_two():
    # 30 + padding must be divisible by 2^levels (4). 30+2=32. ✓
    model = WNO(in_channels=1, out_channels=1, hidden_channels=16, n_layers=2, levels=2, padding=2)
    x = torch.randn(2, 1, 30, 30)
    out = model(x)
    assert out.shape == (2, 1, 30, 30)


# --- UNO ---

def test_uno_forward():
    model = UNO(in_channels=2, out_channels=1, hidden_channels=16, n_levels=2, modes=4)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


# --- DeepONet ---

def test_deeponet_forward():
    model = DeepONet(n_sensors=32, n_query_dim=2, hidden_features=32, n_layers=3, p=32)
    u = torch.randn(4, 32)   # (B, n_sensors)
    y = torch.randn(16, 2)   # (Q, n_query_dim)
    out = model(u, y)
    assert out.shape == (4, 16)


def test_deeponet_no_bias():
    model = DeepONet(n_sensors=16, n_query_dim=1, hidden_features=16, n_layers=2, p=16, bias=False)
    u = torch.randn(2, 16)
    y = torch.randn(8, 1)
    out = model(u, y)
    assert out.shape == (2, 8)


# --- MultiScaleFNO ---

def test_multiscale_fno_forward():
    model = MultiScaleFNO(in_channels=2, out_channels=1, hidden_channels=16, n_layers=2, n_scales=3, max_modes=8)
    x = torch.randn(2, 2, 32, 32)
    out = model(x)
    assert out.shape == (2, 1, 32, 32)


# --- CoupledOperator ---

def test_coupled_operator_direct():
    ops = {
        "a": FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2),
        "b": FNO(in_channels=2, out_channels=2, hidden_channels=8, n_layers=1, modes=4, dim=2),
    }
    model = CoupledOperator(operators=ops, coupling_channels={"a": 1, "b": 2}, coupling_mode="direct")
    inputs = {"a": torch.randn(2, 1, 16, 16), "b": torch.randn(2, 2, 16, 16)}
    out = model(inputs)
    assert out["a"].shape == (2, 1, 16, 16)
    assert out["b"].shape == (2, 2, 16, 16)


def test_coupled_operator_sequential():
    ops = {
        "a": FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2),
        "b": FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2),
    }
    model = CoupledOperator(operators=ops, coupling_channels={"a": 1, "b": 1}, coupling_mode="sequential")
    inputs = {"a": torch.randn(2, 1, 16, 16), "b": torch.randn(2, 1, 16, 16)}
    out = model(inputs)
    assert out["a"].shape == (2, 1, 16, 16)
    assert out["b"].shape == (2, 1, 16, 16)


def test_coupled_operator_learned():
    ops = {
        "a": FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2),
        "b": FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2),
    }
    model = CoupledOperator(operators=ops, coupling_channels={"a": 1, "b": 1}, coupling_mode="learned", n_coupling_steps=2)
    inputs = {"a": torch.randn(2, 1, 16, 16), "b": torch.randn(2, 1, 16, 16)}
    out = model(inputs)
    assert out["a"].shape == (2, 1, 16, 16)
    assert out["b"].shape == (2, 1, 16, 16)
    strengths = model.coupling_strengths()
    assert strengths.shape == (2, 2)


# --- ConformalNeuralOperator ---

def test_conformal_predict_uncalibrated_raises():
    base = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    model = ConformalNeuralOperator(base)
    assert not model.is_calibrated
    with pytest.raises(RuntimeError):
        model.predict(torch.randn(1, 1, 16, 16))


def test_conformal_calibrate_and_predict():
    base = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    model = ConformalNeuralOperator(base)
    cal_x = torch.randn(20, 1, 16, 16)
    cal_y = torch.randn(20, 1, 16, 16)
    q = model.calibrate(cal_x, cal_y, alpha=0.1)
    assert q > 0
    assert model.is_calibrated
    lo, hi, pt = model.predict(torch.randn(2, 1, 16, 16))
    assert lo.shape == pt.shape == hi.shape
    assert (hi - lo).min().item() > 0


# --- NeuralResidualCorrector ---

def test_residual_corrector_forward():
    solver = lambda x: x[:, :1]  # trivial coarse solver: return first channel
    model = NeuralResidualCorrector(
        solver=solver, in_channels=2, out_channels=1,
        solver_out_channels=1, hidden_channels=8, n_layers=1, modes=4,
    )
    x = torch.randn(2, 2, 16, 16)
    out = model(x)
    assert out.shape == (2, 1, 16, 16)
