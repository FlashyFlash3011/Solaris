# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hyperparameter tuning wrapper around Optuna."""

from typing import Any, Callable, Dict, Optional


class HyperparameterTuner:
    """Wraps Optuna for hyperparameter optimisation of Solaris models.

    Degrades gracefully if ``optuna`` is not installed — ``optimize`` will
    raise ``ImportError`` with a helpful message.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials.
    direction : str
        ``"minimize"`` or ``"maximize"``.
    study_name : str, optional
        Name for the Optuna study.
    """

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "minimize",
        study_name: Optional[str] = None,
    ) -> None:
        self.n_trials = n_trials
        self.direction = direction
        self.study_name = study_name

    def suggest_fno_params(self, trial: Any) -> Dict[str, Any]:
        """Suggest FNO hyperparameters for an Optuna trial.

        Parameters
        ----------
        trial : optuna.trial.Trial

        Returns
        -------
        dict with keys: hidden_channels, n_layers, modes, lr, batch_size
        """
        return {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
            "n_layers": trial.suggest_int("n_layers", 2, 6),
            "modes": trial.suggest_int("modes", 4, 24),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        }

    def optimize(self, objective_fn: Callable[[Any], float]) -> Dict[str, Any]:
        """Run the Optuna optimisation loop.

        Parameters
        ----------
        objective_fn : callable
            Function that accepts an Optuna ``Trial`` and returns a scalar
            metric to optimise.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        try:
            import optuna  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "optuna is required for HyperparameterTuner. "
                "Install it with: pip install 'solaris-rocm[tuning]'"
            ) from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
        )
        study.optimize(objective_fn, n_trials=self.n_trials)
        return study.best_params
