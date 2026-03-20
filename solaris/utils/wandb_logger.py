# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Weights & Biases logger with graceful no-op fallback."""

from typing import Any

import torch.nn as nn


class WandbLogger:
    """Thin wrapper around ``wandb`` that degrades gracefully when not installed.

    If ``wandb`` is not installed or ``enabled=False``, every method is a
    no-op so training code never needs to branch on availability.

    Parameters
    ----------
    project : str
        W&B project name.
    name : str, optional
        Run name.
    config : dict, optional
        Hyperparameter config to log.
    enabled : bool
        Set to ``False`` to unconditionally disable logging.
    """

    def __init__(
        self,
        project: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        self._run = None
        if not enabled:
            return
        try:
            import wandb  # noqa: PLC0415

            self._run = wandb.init(project=project, name=name, config=config or {})
        except ImportError:
            pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics : dict[str, float]
        step : int, optional
        """
        if self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_model(self, model: nn.Module, name: str = "model") -> None:
        """Save model weights as a W&B artifact.

        Parameters
        ----------
        model : nn.Module
        name : str
        """
        if self._run is None:
            return
        import os
        import tempfile  # noqa: PLC0415, E401

        import torch
        import wandb

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            torch.save(model.state_dict(), path)
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(path)
            self._run.log_artifact(artifact)
        finally:
            os.unlink(path)

    def finish(self) -> None:
        """Mark the run as finished."""
        if self._run is None:
            return
        self._run.finish()
