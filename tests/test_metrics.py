import pytest
import torch

from solaris.metrics import r2_score, relative_l2_error, rmse


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


def _two_samples():
    target = torch.ones(2, 1, 4, 4)
    pred = target.clone()
    pred[0] *= 2.0  # 100% error
    pred[1] *= 1.01  # 1% error
    return pred, target


def test_relative_l2_default_is_mean_of_per_sample_errors():
    pred, target = _two_samples()
    assert relative_l2_error(pred, target).item() == pytest.approx(0.505, abs=1e-6)


def test_relative_l2_reduction_none_returns_each_sample():
    pred, target = _two_samples()
    torch.testing.assert_close(
        relative_l2_error(pred, target, reduction="none"), torch.tensor([1.0, 0.01])
    )


def test_relative_l2_pooled_matches_legacy_behaviour():
    pred, target = _two_samples()
    expected = torch.linalg.norm(pred - target) / torch.linalg.norm(target)
    torch.testing.assert_close(relative_l2_error(pred, target, reduction="pooled"), expected)


def test_relative_l2_rejects_unknown_reduction():
    pred, target = _two_samples()
    with pytest.raises(ValueError):
        relative_l2_error(pred, target, reduction="sum")
