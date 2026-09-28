#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""Example: train a 2-D FNO, on one device or data-parallel across several.

The data is a quick synthetic stand-in for Darcy flow (a log-normal
"permeability" field mapped to a locally averaged "pressure" field) — it is
meant to exercise the training stack, not to be physically exact.

Run on CPU:
    python examples/train_fno_darcy.py

Run on an AMD (ROCm) or NVIDIA GPU:
    python examples/train_fno_darcy.py --device cuda

Data-parallel on 4 GPUs (one process per GPU; collectives use RCCL on ROCm):
    torchrun --nproc_per_node=4 examples/train_fno_darcy.py --device cuda

Each process trains on its own shard of every epoch (DistributedSampler), and
DistributedDataParallel averages gradients so all replicas stay identical.
Rank 0 alone logs and writes checkpoints.
"""

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from solaris.distributed import DistributedManager
from solaris.metrics import relative_l2_error
from solaris.models.fno import FNO
from solaris.utils import get_logger, save_checkpoint


def generate_darcy_like_data(n_samples: int = 512, resolution: int = 64, seed: int = 42):
    """Synthetic Darcy-like pairs: log-normal field → 5×5 local average."""
    g = torch.Generator().manual_seed(seed)
    k = torch.exp(torch.randn(n_samples, 1, resolution, resolution, generator=g))
    p = torch.nn.functional.avg_pool2d(k, 5, stride=1, padding=2)
    return k, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16, help="per process")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    manager = DistributedManager()
    manager.initialize(device_type=args.device)
    device = manager.device
    log = get_logger(f"train_fno_darcy[rank {manager.rank}]")
    if manager.is_main:
        log.info(f"device={device}  world_size={manager.world_size}  backend={manager.backend}")

    # Every process builds the same dataset (same seed); the sampler shards it.
    k, p = generate_darcy_like_data(args.n_samples, args.resolution)
    n_train = int(0.8 * args.n_samples)
    train_ds = TensorDataset(k[:n_train], p[:n_train])
    val_ds = TensorDataset(k[n_train:], p[n_train:])
    sampler = DistributedSampler(train_ds, shuffle=True) if manager.distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = FNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        modes=args.modes,
        dim=2,
        norm=None,
    ).to(device)
    if manager.distributed:
        ddp = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    else:
        ddp = model
    if manager.is_main:
        log.info(f"Model: {model.meta.name} | Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.Adam(ddp.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)  # a different shuffle every epoch

        ddp.train()
        # [sum of losses, number of samples] on this rank
        totals = torch.zeros(2, device=device)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp(x), y)
            loss.backward()  # DDP all-reduces (averages) gradients here
            optimizer.step()
            totals += torch.stack([loss.detach() * len(x), torch.tensor(len(x), device=device)])
        if manager.distributed:
            dist.all_reduce(totals)  # combine every rank's shard
        train_loss = (totals[0] / totals[1]).item()
        scheduler.step()

        # Validation is cheap here, so every rank evaluates the full val set
        # with identical weights; only rank 0 reports.
        model.eval()
        val_loss, val_rel_l2 = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * len(x)
                val_rel_l2 += relative_l2_error(pred, y).item() * len(x)
        val_loss /= len(val_ds)
        val_rel_l2 /= len(val_ds)

        if manager.is_main:
            log.info(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4e} | val_loss={val_loss:.4e}"
                f" | rel_L2={val_rel_l2:.4f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    f"{args.checkpoint_dir}/best_fno.pt",
                    model,  # the unwrapped module, so the checkpoint loads without DDP
                    optimizer,
                    scheduler,
                    epoch,
                    val_loss,
                )

    # Sanity line: identical on every rank when DDP kept the replicas in sync.
    checksum = sum(
        (torch.view_as_real(p) if p.is_complex() else p).detach().double().sum().item()
        for p in model.parameters()
    )
    log.info(f"param_checksum={checksum:.10e}")
    if manager.is_main:
        log.info(f"Training complete. Best val loss: {best_val_loss:.4e}")
    manager.barrier()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
