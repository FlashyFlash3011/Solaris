#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""Example: Train a 2-D FNO on a synthetic Darcy flow dataset.

Run on CPU:
    python examples/train_fno_darcy.py

Run on AMD GPU (ROCm):
    python examples/train_fno_darcy.py --device cuda

Distributed (e.g. 2 GPUs):
    torchrun --nproc_per_node=2 examples/train_fno_darcy.py --device cuda
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.models.fno import FNO
from physicsnemo.metrics import relative_l2_error, rmse
from physicsnemo.utils import get_logger, save_checkpoint


def generate_darcy_data(n_samples: int = 512, resolution: int = 64, seed: int = 42):
    """Generate synthetic Darcy-flow-like data for demonstration.
    Input: random permeability field. Output: smoothed pressure field.
    """
    torch.manual_seed(seed)
    # Random permeability (log-normal)
    k = torch.exp(torch.randn(n_samples, 1, resolution, resolution))
    # Simple pressure approximation via blurring (not real Darcy, but illustrative)
    p = torch.nn.functional.avg_pool2d(k, 5, stride=1, padding=2)
    return k, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    args = parser.parse_args()

    log = get_logger("train_fno_darcy")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Using device: {device}")

    # Data
    log.info("Generating synthetic Darcy data...")
    k, p = generate_darcy_data(n_samples=512, resolution=64)
    n_train = 400
    train_ds = TensorDataset(k[:n_train], p[:n_train])
    val_ds = TensorDataset(k[n_train:], p[n_train:])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = FNO(in_channels=1, out_channels=1, hidden_channels=args.hidden, n_layers=args.n_layers, modes=args.modes, dim=2)
    model = model.to(device)
    log.info(f"Model: {model.meta.name} | Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
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

        log.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4e} | val_loss={val_loss:.4e} | rel_L2={val_rel_l2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(f"{args.checkpoint_dir}/best_fno.pt", model, optimizer, scheduler, epoch, val_loss)

    log.info(f"Training complete. Best val loss: {best_val_loss:.4e}")


if __name__ == "__main__":
    main()
