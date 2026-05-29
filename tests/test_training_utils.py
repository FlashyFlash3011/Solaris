# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn

from solaris.utils.training import (
    CombinedOptimizer,
    EarlyStopping,
    GradientClipper,
    StaticCaptureTraining,
    WarmupCosineScheduler,
)
from solaris.utils.wandb_logger import WandbLogger

# --- EarlyStopping ---


def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, min_delta=0.0, mode="min")
    # Improving values — should not stop
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.8)
    # No improvement for patience=3 steps
    assert not es.step(0.8)  # counter=1
    assert not es.step(0.8)  # counter=2
    assert es.step(0.8)  # counter=3 → stop


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)
    # No improvement
    assert not es.step(0.6)  # counter=1
    assert es.step(0.6)  # counter=2 → stop


def test_early_stopping_reset():
    es = EarlyStopping(patience=2, mode="min")
    es.step(1.0)
    es.step(1.0)  # counter=1
    es.reset()
    assert es._counter == 0
    assert not es.step(1.0)  # fresh start — counter=1


def test_early_stopping_min_delta():
    es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
    es.step(1.0)
    # Decrease < min_delta is not considered improvement
    assert not es.step(0.95)  # counter=1
    assert es.step(0.95)  # counter=2 → stop


# --- GradientClipper ---


def test_gradient_clipper_clips_norm():
    model = nn.Linear(10, 10)
    # Set large gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 1000.0
    clipper = GradientClipper(max_norm=1.0)
    pre_norm = clipper(model)
    assert isinstance(pre_norm, float)
    total_norm = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5
    assert total_norm <= 1.0 + 1e-6


def test_gradient_clipper_returns_float():
    model = nn.Linear(4, 4)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    clipper = GradientClipper(max_norm=10.0)
    result = clipper(model)
    assert isinstance(result, float)


# --- WandbLogger (no-op when wandb not installed) ---


def test_wandb_logger_noop_when_disabled():
    logger = WandbLogger(project="test", enabled=False)
    assert logger._run is None
    logger.log_metrics({"loss": 0.1}, step=1)
    logger.finish()


def test_wandb_logger_noop_without_wandb(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    logger = WandbLogger(project="test", enabled=True)
    assert logger._run is None
    logger.log_metrics({"val_loss": 0.5})
    logger.finish()


# --- CombinedOptimizer ---


def test_combined_optimizer_zero_grad_and_step():
    m1 = nn.Linear(4, 4)
    m2 = nn.Linear(4, 2)
    opt = CombinedOptimizer(
        [
            torch.optim.Adam(m1.parameters(), lr=1e-3),
            torch.optim.Adam(m2.parameters(), lr=5e-4),
        ]
    )
    loss = (m2(m1(torch.randn(2, 4)))).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    for p in list(m1.parameters()) + list(m2.parameters()):
        assert p.grad is None or (p.grad == 0).all()


def test_combined_optimizer_param_groups_flat():
    m1, m2 = nn.Linear(4, 4), nn.Linear(4, 2)
    opt1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    opt2 = torch.optim.Adam(m2.parameters(), lr=5e-4)
    combined = CombinedOptimizer([opt1, opt2])
    assert len(combined.param_groups) == len(opt1.param_groups) + len(opt2.param_groups)


def test_combined_optimizer_state_dict_roundtrip():
    m = nn.Linear(4, 4)
    opt = CombinedOptimizer([torch.optim.Adam(m.parameters(), lr=1e-3)])
    m(torch.randn(2, 4)).sum().backward()
    opt.step()
    opt.load_state_dict(opt.state_dict())  # should not raise


def test_combined_optimizer_with_warmup_scheduler():
    m = nn.Linear(4, 4)
    combined = CombinedOptimizer([torch.optim.Adam(m.parameters(), lr=1e-3)])
    sched = WarmupCosineScheduler(combined, warmup_epochs=2, total_epochs=10, base_lr=1e-3)
    lr = sched.step()
    assert 0 < lr <= 1e-3


def test_combined_optimizer_requires_nonempty():
    with pytest.raises(ValueError):
        CombinedOptimizer([])


# --- StaticCaptureTraining (CPU fallback — no graph capture) ---


def test_static_capture_disabled_runs_normally():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    capturer = StaticCaptureTraining(model, opt, loss_fn=nn.functional.mse_loss, enabled=False)
    for _ in range(5):
        loss = capturer(torch.randn(2, 4), torch.randn(2, 2))
        assert isinstance(loss.item(), float)


def test_static_capture_cpu_fallback():
    """On CPU all steps run as normal eager steps (CUDA not available)."""
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    capturer = StaticCaptureTraining(model, opt, loss_fn=nn.functional.mse_loss, warmup_steps=3)
    for _ in range(6):
        loss = capturer(torch.randn(2, 4), torch.randn(2, 2))
        assert isinstance(loss.item(), float)


def test_static_capture_reset():
    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    capturer = StaticCaptureTraining(model, opt, loss_fn=nn.functional.mse_loss, warmup_steps=2)
    capturer._step_count = 10
    capturer.reset()
    assert capturer._step_count == 0
    assert capturer._graph is None
