# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
from solaris.utils.training import EarlyStopping, GradientClipper
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
    assert es.step(0.8)      # counter=3 → stop


def test_early_stopping_max_mode():
    es = EarlyStopping(patience=2, mode="max")
    assert not es.step(0.5)
    assert not es.step(0.6)
    # No improvement
    assert not es.step(0.6)  # counter=1
    assert es.step(0.6)      # counter=2 → stop


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
    assert es.step(0.95)      # counter=2 → stop


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
