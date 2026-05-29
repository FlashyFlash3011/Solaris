# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""ROCm-aware distributed training manager.

On AMD GPUs PyTorch uses the HIP backend which maps ``torch.cuda.*`` to HIP
calls transparently.  Distributed collectives go through RCCL (AMD's NCCL
equivalent), but PyTorch exposes it via the same ``"nccl"`` backend name.
"""

import os
from typing import Optional

import torch
import torch.distributed as dist


class DistributedManager:
    """Singleton for distributed training setup.

    Supports single process-group (data-parallel) training out of the box and
    optionally a multi-dimensional ``DeviceMesh`` for combined tensor + data
    parallelism (requires PyTorch >= 2.4).

    Usage — data parallel
    ---------------------
    >>> manager = DistributedManager()
    >>> manager.initialize()

    Usage — tensor + data parallel (2-D mesh)
    ------------------------------------------
    >>> manager = DistributedManager()
    >>> manager.initialize()
    >>> manager.initialize_mesh((2, 4), ("tensor_parallel", "data_parallel"))
    >>> tp_group = manager.get_group("tensor_parallel")
    """

    _instance: Optional["DistributedManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "DistributedManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._named_groups: dict[str, dist.ProcessGroup] = {}
            cls._instance._mesh = None
        return cls._instance

    def initialize(self) -> None:
        """Initialize the global process group.

        Reads standard PyTorch environment variables:
        ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``.

        On ROCm, ``"nccl"`` transparently uses RCCL.
        Falls back to ``"gloo"`` when no GPU is available (CPU-only testing).
        """
        if self._initialized:
            return
        self._rank = int(os.environ.get("RANK", 0))
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device(f"cuda:{self._local_rank}")
            backend = "nccl"  # Maps to RCCL on ROCm, NCCL on CUDA
        else:
            self._device = torch.device("cpu")
            backend = "gloo"

        if self._world_size > 1:
            dist.init_process_group(
                backend=backend,
                rank=self._rank,
                world_size=self._world_size,
            )
        self._initialized = True

    def initialize_mesh(
        self,
        mesh_shape: tuple[int, ...],
        mesh_dim_names: tuple[str, ...],
    ) -> None:
        """Set up a multi-dimensional ``DeviceMesh`` for tensor + data parallelism.

        Requires PyTorch >= 2.4 and an already-initialized process group
        (call :meth:`initialize` first).

        Parameters
        ----------
        mesh_shape :
            Shape of the device mesh.  E.g. ``(2, 4)`` creates a 2 × 4 grid
            of 8 GPUs: 2 per model replica, 4 replicas.
        mesh_dim_names :
            Name for each mesh dimension.  E.g.
            ``("tensor_parallel", "data_parallel")``.
            Names are used to retrieve sub-groups via :meth:`get_group`.

        Example
        -------
        >>> manager.initialize_mesh((2, 4), ("tensor_parallel", "data_parallel"))
        >>> tp_group = manager.get_group("tensor_parallel")
        """
        try:
            from torch.distributed.device_mesh import init_device_mesh
        except ImportError as exc:
            raise RuntimeError(
                f"initialize_mesh() requires PyTorch >= 2.4. Current version: {torch.__version__}"
            ) from exc

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._mesh = init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)

        for name in mesh_dim_names:
            self._named_groups[name] = self._mesh[name].get_group()

    def get_group(self, name: str) -> dist.ProcessGroup:
        """Return a named sub-group created by :meth:`initialize_mesh`.

        Parameters
        ----------
        name :
            Mesh dimension name, e.g. ``"tensor_parallel"``.

        Raises
        ------
        KeyError
            If *name* was not registered via :meth:`initialize_mesh`.
        """
        if name not in self._named_groups:
            available = list(self._named_groups)
            raise KeyError(
                f"Process group '{name}' not found. "
                f"Available: {available}. Call initialize_mesh() first."
            )
        return self._named_groups[name]

    def create_group(
        self,
        name: str,
        ranks: list,
        backend: str | None = None,
    ) -> dist.ProcessGroup:
        """Manually create and register a named process group.

        Use this when you need a custom sub-group outside of a DeviceMesh.

        Parameters
        ----------
        name :
            Identifier for later retrieval via :meth:`get_group`.
        ranks :
            Global ranks that form this group.
        backend :
            Collective backend (``"nccl"`` / ``"gloo"``).  Defaults to the
            same backend as the global group.
        """
        group = dist.new_group(ranks=ranks, backend=backend)
        self._named_groups[name] = group
        return group

    @property
    def mesh(self):
        """The active ``DeviceMesh``, or ``None`` if not initialised."""
        return self._mesh

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_main(self) -> bool:
        """True only on rank 0."""
        return self._rank == 0

    @property
    def distributed(self) -> bool:
        return self._world_size > 1

    @staticmethod
    def cleanup() -> None:
        """Tear down the process group."""
        if dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        """Synchronise all ranks."""
        if self.distributed and dist.is_initialized():
            dist.barrier()

    def __repr__(self) -> str:
        groups = list(self._named_groups)
        return (
            f"DistributedManager(rank={self._rank}, world_size={self._world_size}, "
            f"local_rank={self._local_rank}, device={self._device}, "
            f"named_groups={groups})"
        )
