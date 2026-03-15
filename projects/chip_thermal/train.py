# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train an FNO surrogate for chip thermal prediction.

Usage
-----
# CPU
python train.py

# AMD GPU (ROCm)
python train.py --device cuda

# Tune data size / epochs
python train.py --device cuda --n_train 2000 --epochs 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.models.fno import FNO
from physicsnemo.metrics import relative_l2_error
from physicsnemo.utils import get_logger, save_checkpoint
from solver import random_power_map, solve_heat


# ─── Data generation ─────────────────────────────────────────────────────────

def generate_dataset(n: int, resolution: int, seed: int = 0) -> tuple:
    """Generate (power_map, temperature) pairs using the FD solver.

    Returns arrays of shape (n, 1, H, W).
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    Q_all, T_all = [], []

    log.info(f"Generating {n} samples at {resolution}×{resolution} ...")
    t0 = time.perf_counter()
    for i in range(n):
        Q = random_power_map(resolution, resolution, rng=rng)
        T, _, _ = solve_heat(Q, max_iter=10_000, tol=1e-4)
        Q_all.append(Q)
        T_all.append(T)
        if (i + 1) % max(1, n // 10) == 0:
            pct = 100 * (i + 1) / n
            log.info(f"  {i+1}/{n} ({pct:.0f}%) — {time.perf_counter()-t0:.1f}s elapsed")

    elapsed = time.perf_counter() - t0
    log.info(f"Dataset done in {elapsed:.1f}s  ({elapsed/n*1000:.0f} ms/sample)")

    Q_arr = np.stack(Q_all)[:, None]  # (N, 1, H, W)
    T_arr = np.stack(T_all)[:, None]
    return Q_arr, T_arr


def normalize(arr: np.ndarray):
    mean, std = arr.mean(), arr.std() + 1e-8
    return (arr - mean) / std, mean, std


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    log = get_logger("train")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    cache = Path(args.cache)
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        Q_arr, T_arr = d["Q"], d["T"]
        if len(Q_arr) < args.n_train + args.n_val:
            log.info("Cache too small — regenerating")
            Q_arr, T_arr = None, None
    else:
        Q_arr, T_arr = None, None

    if Q_arr is None:
        Q_arr, T_arr = generate_dataset(args.n_train + args.n_val, args.resolution, seed=42)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Q=Q_arr, T=T_arr)
        log.info(f"Saved dataset → {cache}")

    # Normalise
    Q_norm, Q_mean, Q_std = normalize(Q_arr)
    T_norm, T_mean, T_std = normalize(T_arr)

    # Save stats for inference
    stats_path = Path(args.checkpoint_dir) / "norm_stats.npz"
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    np.savez(stats_path, Q_mean=Q_mean, Q_std=Q_std, T_mean=T_mean, T_std=T_std)

    Q_t = torch.as_tensor(Q_norm, dtype=torch.float32)
    T_t = torch.as_tensor(T_norm, dtype=torch.float32)

    train_ds = TensorDataset(Q_t[: args.n_train], T_t[: args.n_train])
    val_ds   = TensorDataset(Q_t[args.n_train :], T_t[args.n_train :])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              pin_memory=(device.type == "cuda"))

    # ── Model ──
    model = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        modes=args.modes,
        dim=2,
    ).to(device)
    log.info(f"FNO parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    log.info("Starting training …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for Q_b, T_b in train_loader:
            Q_b, T_b = Q_b.to(device), T_b.to(device)
            optimizer.zero_grad()
            pred = model(Q_b)
            loss = loss_fn(pred, T_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(Q_b)
        tr_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, val_l2 = 0.0, 0.0
        with torch.no_grad():
            for Q_b, T_b in val_loader:
                Q_b, T_b = Q_b.to(device), T_b.to(device)
                pred = model(Q_b)
                val_loss += loss_fn(pred, T_b).item() * len(Q_b)
                val_l2   += relative_l2_error(pred, T_b).item() * len(Q_b)
        val_loss /= len(val_ds)
        val_l2   /= len(val_ds)

        log.info(f"Epoch {epoch:3d}/{args.epochs} | train={tr_loss:.3e} | val={val_loss:.3e} | rel-L2={val_l2:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                Path(args.checkpoint_dir) / "best_fno.pt",
                model, optimizer, scheduler, epoch, val_loss,
                extra={"resolution": args.resolution},
            )

    log.info(f"Training complete. Best val loss: {best_val:.4e}")
    log.info(f"Checkpoint → {args.checkpoint_dir}/best_fno.pt")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",         default="cpu")
    p.add_argument("--resolution",     type=int,   default=64)
    p.add_argument("--n_train",        type=int,   default=800)
    p.add_argument("--n_val",          type=int,   default=200)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden",         type=int,   default=64)
    p.add_argument("--modes",          type=int,   default=16)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--cache",          default="data/thermal_dataset.npz")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    train(p.parse_args())
