import torch
from solaris.metrics import relative_l2_error, rmse, nrmse, r2_score


def test_perfect_prediction():
    x = torch.randn(100)
    assert relative_l2_error(x, x).item() < 1e-6
    assert rmse(x, x).item() < 1e-6
    assert r2_score(x, x).item() > 0.999


def test_metrics_shapes():
    pred = torch.randn(4, 3, 16, 16)
    target = torch.randn(4, 3, 16, 16)
    # Should return scalars
    assert relative_l2_error(pred, target).ndim == 0
    assert rmse(pred, target).ndim == 0
    assert r2_score(pred, target).ndim == 0
