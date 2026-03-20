# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-process tests for DistributedManager.

These tests run without spawning multiple processes; they validate the
manager's behaviour in the common single-GPU / single-CPU case.
"""

import torch

from solaris.distributed import DistributedManager


def test_distributed_manager_is_singleton():
    m1 = DistributedManager()
    m2 = DistributedManager()
    assert m1 is m2


def test_distributed_manager_initialize_single():
    manager = DistributedManager()
    manager.initialize()
    assert manager.world_size >= 1
    assert manager.rank == 0  # default env var
    assert manager.local_rank == 0


def test_distributed_manager_is_main():
    manager = DistributedManager()
    manager.initialize()
    assert manager.is_main  # rank 0 is always main


def test_distributed_manager_not_distributed_single():
    manager = DistributedManager()
    manager.initialize()
    # world_size=1 means not distributed
    assert not manager.distributed


def test_distributed_manager_device():
    manager = DistributedManager()
    manager.initialize()
    assert isinstance(manager.device, torch.device)


def test_distributed_manager_barrier_single():
    """Barrier should be a no-op for world_size=1."""
    manager = DistributedManager()
    manager.initialize()
    # Should not raise
    manager.barrier()


def test_distributed_manager_repr():
    manager = DistributedManager()
    manager.initialize()
    r = repr(manager)
    assert "DistributedManager" in r
    assert "rank=0" in r


def test_distributed_manager_initialize_idempotent():
    """Calling initialize() multiple times is safe (early-return guard)."""
    manager = DistributedManager()
    manager.initialize()
    manager.initialize()  # should not raise
    assert manager.rank == 0
