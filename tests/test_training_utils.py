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


def test_static_capture_cpu_tensors_never_attempt_capture():
    """CPU data must run eagerly without trying (and warning about) graph capture."""
    import warnings

    model = nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    capturer = StaticCaptureTraining(model, opt, loss_fn=nn.functional.mse_loss, warmup_steps=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(4):
            capturer(torch.randn(2, 4), torch.randn(2, 2))
    assert capturer._graph is None


def _capture_vs_eager(optimizer_cls):
    torch.manual_seed(0)
    data = [
        (torch.randn(8, 16, device="cuda"), torch.randn(8, 4, device="cuda")) for _ in range(30)
    ]

    def make():
        torch.manual_seed(1)
        m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).cuda()
        return m, optimizer_cls(m.parameters(), lr=1e-2)

    m_eager, o_eager = make()
    for x, y in data:
        o_eager.zero_grad()
        nn.functional.mse_loss(m_eager(x), y).backward()
        o_eager.step()

    m_graph, o_graph = make()
    capturer = StaticCaptureTraining(m_graph, o_graph, nn.functional.mse_loss, warmup_steps=3)
    for x, y in data:
        capturer(x, y)
    return capturer, m_eager, m_graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD])
def test_static_capture_really_captures_and_matches_eager(optimizer_cls):
    capturer, m_eager, m_graph = _capture_vs_eager(optimizer_cls)
    assert capturer.captured, "graph capture fell back to eager mode"
    for pe, pg in zip(m_eager.parameters(), m_graph.parameters()):
        torch.testing.assert_close(pg, pe, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("scheduler", ["torch_cosine", "warmup_cosine"])
@pytest.mark.parametrize("optimizer_cls", [torch.optim.Adam, torch.optim.AdamW])
def test_static_capture_follows_lr_scheduler(optimizer_cls, scheduler):
    """A captured step must use the *current* LR, not the one at capture time."""
    from solaris.utils import WarmupCosineScheduler

    torch.manual_seed(0)
    data = [
        (torch.randn(8, 16, device="cuda"), torch.randn(8, 4, device="cuda")) for _ in range(20)
    ]

    def make():
        torch.manual_seed(1)
        m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4)).cuda()
        o = optimizer_cls(m.parameters(), lr=1e-2)
        if scheduler == "torch_cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=20)
        else:
            sch = WarmupCosineScheduler(o, warmup_epochs=3, total_epochs=20, base_lr=1e-2)
        return m, o, sch

    m_eager, o_eager, s_eager = make()
    for x, y in data:
        o_eager.zero_grad()
        nn.functional.mse_loss(m_eager(x), y).backward()
        o_eager.step()
        s_eager.step()

    m_graph, o_graph, s_graph = make()
    capturer = StaticCaptureTraining(m_graph, o_graph, nn.functional.mse_loss, warmup_steps=3)
    for x, y in data:
        capturer(x, y)
        s_graph.step()

    assert capturer.captured
    for pe, pg in zip(m_eager.parameters(), m_graph.parameters()):
        torch.testing.assert_close(pg, pe, atol=1e-4, rtol=1e-4)


# --- WarmupCosineScheduler ---------------------------------------------------------


def _sched(warmup, total, base=1e-3, min_lr=0.0):
    from solaris.utils import WarmupCosineScheduler

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base)
    return opt, WarmupCosineScheduler(opt, warmup, total, base_lr=base, min_lr=min_lr)


def test_warmup_cosine_without_warmup_starts_at_base_lr():
    opt, s = _sched(0, 10)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    lrs = [s.step() for _ in range(10)]
    assert lrs[0] > 0.9e-3
    assert lrs == sorted(lrs, reverse=True)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)


def test_warmup_cosine_first_epoch_lr_is_nonzero():
    opt, s = _sched(4, 20)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 / 4)
    lrs = [s.step() for _ in range(3)]
    assert lrs == pytest.approx([2e-3 / 4, 3e-3 / 4, 1e-3])


def test_warmup_cosine_ends_at_min_lr():
    _, s = _sched(2, 12, min_lr=1e-5)
    lrs = [s.step() for _ in range(11)]
    assert lrs[-1] == pytest.approx(1e-5)
    assert max(lrs) == pytest.approx(1e-3)
