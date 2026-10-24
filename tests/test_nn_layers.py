"""Behaviour tests for activations and embeddings."""

import math

import pytest
import torch

from solaris.nn import (
    CappedGELU,
    CappedLeakyReLU,
    FourierEmbedding,
    PositionalEmbedding,
    SinusoidalTimestepEmbedding,
    Siren,
    Stan,
)

# --- activations ---------------------------------------------------------------


def test_capped_gelu_matches_gelu_below_cap_and_caps_above():
    act = CappedGELU(cap_value=2.0)
    x = torch.tensor([-1.0, 0.5, 1.5, 10.0])
    out = act(x)
    torch.testing.assert_close(out[:3], torch.nn.functional.gelu(x[:3]))
    assert out[3] == 2.0


def test_capped_leaky_relu():
    act = CappedLeakyReLU(cap_value=3.0, negative_slope=0.1)
    torch.testing.assert_close(act(torch.tensor([-2.0, 1.0, 5.0])), torch.tensor([-0.2, 1.0, 3.0]))


def test_stan_is_x_tanh_beta_x_and_beta_is_learnable():
    act = Stan(out_features=3)
    x = torch.randn(4, 3)
    torch.testing.assert_close(act(x), x * torch.tanh(x))  # beta initialised to 1
    act(x).sum().backward()
    assert act.beta.grad is not None and act.beta.grad.shape == (3,)


def test_siren_activation_and_init_bounds():
    act = Siren(omega_0=30.0)
    x = torch.randn(5)
    torch.testing.assert_close(act(x), torch.sin(30.0 * x))
    first, hidden = torch.nn.Linear(4, 16), torch.nn.Linear(16, 16)
    Siren.init_weights(first, omega_0=30.0, is_first=True)
    Siren.init_weights(hidden, omega_0=30.0, is_first=False)
    assert first.weight.abs().max() <= 1 / 4
    assert hidden.weight.abs().max() <= math.sqrt(6 / 16) / 30.0


# --- embeddings ------------------------------------------------------------------


def test_fourier_embedding_shape_range_and_periodicity():
    emb = FourierEmbedding(in_features=2, embed_dim=8, scale=1.0)
    x = torch.randn(5, 2)
    out = emb(x)
    assert out.shape == (5, 8)
    assert out.abs().max() <= 1.0 + 1e-6
    # cos²+sin² = 1 for each frequency
    torch.testing.assert_close(out[:, :4] ** 2 + out[:, 4:] ** 2, torch.ones(5, 4))


def test_fourier_embedding_rejects_odd_dim():
    with pytest.raises(ValueError):
        FourierEmbedding(2, 7)


@pytest.mark.parametrize("dim", [8, 7])
def test_positional_embedding_any_dim(dim):
    emb = PositionalEmbedding(embed_dim=dim, max_len=50)
    x = torch.zeros(2, 10, dim)
    out = emb(x)
    assert out.shape == (2, 10, dim)
    torch.testing.assert_close(out[0, 0, 0::2], torch.zeros(len(range(0, dim, 2))))  # sin(0)


def test_positional_embedding_too_long_sequence_raises():
    emb = PositionalEmbedding(embed_dim=4, max_len=5)
    with pytest.raises(ValueError):
        emb(torch.zeros(1, 6, 4))


@pytest.mark.parametrize("dim", [16, 15])
def test_timestep_embedding_returns_requested_dim(dim):
    emb = SinusoidalTimestepEmbedding(embed_dim=dim)
    out = emb(torch.tensor([0.0, 1.0, 50.0]))
    assert out.shape == (3, dim)
    assert torch.isfinite(out).all()


def test_timestep_embedding_distinguishes_timesteps():
    emb = SinusoidalTimestepEmbedding(embed_dim=32)
    out = emb(torch.arange(100.0))
    dists = torch.cdist(out, out)
    assert dists[~torch.eye(100, dtype=torch.bool)].min() > 1e-3
