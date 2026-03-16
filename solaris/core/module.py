# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Base Module class for all PhysicsNeMo models."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from solaris.core.meta import ModelMetaData


class Module(nn.Module):
    """Base class for all PhysicsNeMo models.

    Extends ``torch.nn.Module`` with:
    - JSON-serialisable ``__init__`` argument capture for checkpoint reconstruction.
    - A ``device`` property backed by a persistent buffer (no manual tracking).
    - Convenience ``save`` / ``load`` class methods.

    Parameters
    ----------
    meta : ModelMetaData, optional
        Model metadata. Defaults to a blank ``ModelMetaData`` instance.
    """

    _solaris_version: str = "0.1.0"

    def __init__(self, meta: Optional[ModelMetaData] = None) -> None:
        super().__init__()
        self.meta = meta or ModelMetaData()
        # Zero-element buffer used purely for device tracking — never moved manually.
        self.register_buffer("_device_buf", torch.empty(0), persistent=False)
        self._init_args: Dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        """Return the device this model lives on."""
        return self._device_buf.device

    def _capture_init_args(self, **kwargs: Any) -> None:
        """Store constructor arguments for checkpoint serialisation.

        Call this at the *end* of ``__init__`` in every subclass::

            self._capture_init_args(in_channels=in_channels, ...)
        """
        self._init_args = kwargs

    def save(self, path: Union[str, Path]) -> None:
        """Save weights + init-args to *path* (a ``.pt`` file)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "init_args": self._init_args,
                "class": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "solaris_version": self._solaris_version,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "Module":
        """Reconstruct model from a checkpoint saved with :meth:`save`.

        Parameters
        ----------
        path :
            Path to the ``.pt`` checkpoint file.
        map_location :
            Passed directly to ``torch.load`` for device remapping. If ``None``
            and a ROCm/CUDA GPU is available the checkpoint is loaded on GPU;
            otherwise CPU.
        """
        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["init_args"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def num_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover
        base = super().__repr__()
        return f"{base}\n[PhysicsNeMo | {self.meta.name} | params={self.num_parameters():,}]"
