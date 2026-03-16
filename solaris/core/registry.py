# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Model registry with entry-point discovery."""

from importlib.metadata import entry_points
from typing import Dict, Optional, Type

from solaris.core.module import Module


class ModelRegistry:
    """Singleton registry that discovers PhysicsNeMo models via entry points.

    Entry points should be declared in ``pyproject.toml`` under the group
    ``solaris.models``::

        [project.entry-points."solaris.models"]
        fno = "solaris.models.fno:FNO"
    """

    _instance: Optional["ModelRegistry"] = None
    _registry: Dict[str, Type[Module]] = {}

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._discover()
        return cls._instance

    def _discover(self) -> None:
        eps = entry_points(group="solaris.models")
        for ep in eps:
            try:
                self._registry[ep.name] = ep.load()
            except Exception as exc:  # noqa: BLE001
                import warnings
                warnings.warn(f"Failed to load model entry point '{ep.name}': {exc}", stacklevel=2)

    def register(self, name: str, cls: Type[Module]) -> None:
        """Manually register a model class."""
        self._registry[name] = cls

    def __getitem__(self, name: str) -> Type[Module]:
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._registry)}")
        return self._registry[name]

    def list_models(self) -> list:
        return list(self._registry.keys())

    def __repr__(self) -> str:  # pragma: no cover
        return f"ModelRegistry({self.list_models()})"
