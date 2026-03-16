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
    """Lightweight singleton for distributed training setup.

    Usage
    -----
    >>> manager = DistributedManager()
    >>> manager.initialize()
    >>> # ... training loop ...
    >>> DistributedManager.cleanup()
    """

    _instance: Optional["DistributedManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "DistributedManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self) -> None:
        """Initialize the process group.

        Reads standard PyTorch environment variables:
        ``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``, ``MASTER_ADDR``, ``MASTER_PORT``.

        On ROCm, the ``"nccl"`` backend transparently uses RCCL.
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
        return (
            f"DistributedManager(rank={self._rank}, world_size={self._world_size}, "
            f"local_rank={self._local_rank}, device={self._device})"
        )
