# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests: train → save → load → eval pipeline."""

import torch
import torch.nn as nn

from solaris.models.fno import FNO
from solaris.utils.checkpoint import load_checkpoint, save_checkpoint
from solaris.utils.training import EarlyStopping, GradientClipper


def _make_data(n=32, in_ch=1, out_ch=1, size=16):
    x = torch.randn(n, in_ch, size, size)
    y = torch.randn(n, out_ch, size, size)
    return x, y


def test_train_save_load_eval(tmp_path):
    """Full pipeline: train for a few steps, save, load, assert outputs match."""
    model = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=2, modes=4, dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x, y = _make_data()

    # Mini training loop
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()

    # Save using Module.save (self-describing checkpoint)
    ckpt_path = tmp_path / "model.pt"
    model.save(ckpt_path)

    # Load and compare
    loaded = FNO.load(ckpt_path, map_location="cpu")
    model.eval()
    loaded.eval()
    test_x = torch.randn(2, 1, 16, 16)
    with torch.no_grad():
        out_orig = model(test_x)
        out_loaded = loaded(test_x)
    torch.testing.assert_close(out_orig, out_loaded)


def test_checkpoint_utils(tmp_path):
    """save_checkpoint / load_checkpoint round-trip."""
    model = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt.pt"

    save_checkpoint(ckpt_path, model, optimizer, scheduler=None, epoch=5, loss=0.42)

    # load_checkpoint requires the model to restore weights into
    model2 = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    ckpt = load_checkpoint(ckpt_path, model2, map_location="cpu")

    assert ckpt["epoch"] == 5
    assert abs(ckpt["loss"] - 0.42) < 1e-6
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt


def test_early_stopping_integration():
    """EarlyStopping halts a simulated training loop."""
    es = EarlyStopping(patience=3, min_delta=1e-4, mode="min")
    losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
    stopped_at = None
    for epoch, loss in enumerate(losses):
        if es.step(loss):
            stopped_at = epoch
            break
    assert stopped_at is not None, "Training should have stopped early"


def test_gradient_clipper_integration():
    """GradientClipper works inside a real training step."""
    model = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    clipper = GradientClipper(max_norm=1.0)

    x, y = _make_data(n=4)
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    pre_norm = clipper(model)
    optimizer.step()

    assert isinstance(pre_norm, float)


def test_conformal_save_load_preserves_calibration(tmp_path):
    """ConformalNeuralOperator q_hat survives a save/load round-trip."""
    from solaris.models.conformal import ConformalNeuralOperator
    from solaris.models.fno import FNO

    base = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    predictor = ConformalNeuralOperator(base)

    cal_x = torch.randn(20, 1, 16, 16)
    cal_y = torch.randn(20, 1, 16, 16)
    q_hat_before = predictor.calibrate(cal_x, cal_y, alpha=0.1)
    assert predictor.is_calibrated

    # Save and reload using torch.save on state_dict
    ckpt_path = tmp_path / "conformal.pt"
    torch.save(predictor.state_dict(), ckpt_path)

    base2 = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=1, modes=4, dim=2)
    predictor2 = ConformalNeuralOperator(base2)
    predictor2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))

    assert predictor2.is_calibrated
    assert abs(predictor2._q_hat.item() - q_hat_before) < 1e-6


def test_backward_pass_fno():
    """FNO gradients flow correctly through the full computation graph."""
    model = FNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=2, modes=4, dim=2)
    x = torch.randn(2, 1, 16, 16, requires_grad=False)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # At least one parameter should have a non-None gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_backward_pass_uno():
    from solaris.models.uno import UNO
    model = UNO(in_channels=1, out_channels=1, hidden_channels=8, n_levels=2, modes=4)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_backward_pass_wno():
    from solaris.models.wno import WNO
    model = WNO(in_channels=1, out_channels=1, hidden_channels=8, n_layers=2, levels=2)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
