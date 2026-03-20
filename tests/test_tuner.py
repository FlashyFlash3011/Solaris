# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from solaris.utils.tuner import HyperparameterTuner


def test_tuner_raises_without_optuna(monkeypatch):
    """HyperparameterTuner raises ImportError when optuna is not installed."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "optuna":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    tuner = HyperparameterTuner(n_trials=1)
    with pytest.raises(ImportError):
        tuner.optimize(lambda trial: 0.0)


def test_tuner_suggest_fno_params():
    """suggest_fno_params returns expected keys for a mock trial."""

    class MockTrial:
        def suggest_categorical(self, name, choices):
            return choices[0]

        def suggest_int(self, name, low, high):
            return low

        def suggest_float(self, name, low, high, **kwargs):
            return low

    tuner = HyperparameterTuner(n_trials=1)
    params = tuner.suggest_fno_params(MockTrial())
    assert set(params.keys()) == {"hidden_channels", "n_layers", "modes", "lr", "batch_size"}
    assert isinstance(params["hidden_channels"], int)
    assert isinstance(params["lr"], float)


def test_tuner_defaults():
    tuner = HyperparameterTuner()
    assert tuner.n_trials == 50
    assert tuner.direction == "minimize"
    assert tuner.study_name is None


def test_tuner_custom_params():
    """Constructor arguments are stored correctly."""
    tuner = HyperparameterTuner(n_trials=10, direction="maximize", study_name="my_study")
    assert tuner.n_trials == 10
    assert tuner.direction == "maximize"
    assert tuner.study_name == "my_study"


def test_tuner_suggest_fno_params_value_ranges():
    """Suggested FNO params satisfy expected value constraints."""

    class MaxTrial:
        def suggest_categorical(self, name, choices):
            return choices[-1]

        def suggest_int(self, name, low, high):
            return high

        def suggest_float(self, name, low, high, **kwargs):
            return high

    tuner = HyperparameterTuner()
    params = tuner.suggest_fno_params(MaxTrial())
    assert params["hidden_channels"] in [32, 64, 128]
    assert 2 <= params["n_layers"] <= 6
    assert 4 <= params["modes"] <= 24
    assert 1e-4 <= params["lr"] <= 1e-2
    assert params["batch_size"] in [8, 16, 32]
