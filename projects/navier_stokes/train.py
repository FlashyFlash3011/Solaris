# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train an FNO surrogate to predict future vorticity in 2-D Navier-Stokes.

Usage
-----
# CPU baseline
python train.py --epochs 5 --n_sims 20

# AMD GPU
python train.py --device cuda --epochs 50 --n_sims 500

# With divergence-free constraint
python train.py --constraint divergence_free
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing solver from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from solver import random_vorticity_ic, solve_ns  # noqa: E402

from solaris.models.fno import FNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger, save_checkpoint


def generate_dataset(
    n_sims: int,
    H: int,
    W: int,
    nu: float,
    dt: float,
    n_steps: int,
    forecast_steps: int,
    seed: int = 0,
) -> tuple:
    """Generate (ω_t, ω_{t+k}) pairs from multiple NS simulations.

    Returns arrays of shape (N, 1, H, W).
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    inputs, targets = [], []

    log.info(f"Generating {n_sims} NS simulations at {H}×{W} …")
    t0 = time.perf_counter()
    for i in range(n_sims):
        omega0 = random_vorticity_ic(H, W, n_modes=6, rng=rng)
        total_steps = n_steps + forecast_steps
        snaps = solve_ns(omega0, nu=nu, dt=dt, n_steps=total_steps, n_snapshots=total_steps + 1)
        # Use every step as a training pair
        for t_idx in range(n_steps):
            inputs.append(snaps[t_idx])
            targets.append(snaps[t_idx + forecast_steps])
        if (i + 1) % max(1, n_sims // 5) == 0:
            log.info(f"  {i+1}/{n_sims} — {time.perf_counter()-t0:.1f}s")

    inputs_arr = torch.stack(inputs).unsqueeze(1).numpy()   # (N, 1, H, W)
    targets_arr = torch.stack(targets).unsqueeze(1).numpy()
    log.info(f"Dataset: {len(inputs)} pairs, {time.perf_counter()-t0:.1f}s total")
    return inputs_arr.astype(np.float32), targets_arr.astype(np.float32)


def train(args):
    log = get_logger("ns_train")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")

    # ── Data ──
    cache = Path(args.cache)
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        inputs_arr, targets_arr = d["inputs"], d["targets"]
    else:
        inputs_arr, targets_arr = generate_dataset(
            args.n_sims, args.H, args.W, args.nu, args.dt,
            args.n_steps, args.forecast_steps, seed=42,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, inputs=inputs_arr, targets=targets_arr)
        log.info(f"Saved dataset → {cache}")

    # Normalise
    in_mean, in_std = inputs_arr.mean(), inputs_arr.std() + 1e-8
    tgt_mean, tgt_std = targets_arr.mean(), targets_arr.std() + 1e-8
    inputs_norm = (inputs_arr - in_mean) / in_std
    targets_norm = (targets_arr - tgt_mean) / tgt_std

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ckpt_dir / "norm_stats.npz",
             in_mean=in_mean, in_std=in_std, tgt_mean=tgt_mean, tgt_std=tgt_std)

    n_total = len(inputs_norm)
    n_train = int(n_total * 0.8)
    inp_t = torch.as_tensor(inputs_norm, dtype=torch.float32)
    tgt_t = torch.as_tensor(targets_norm, dtype=torch.float32)

    train_ds = TensorDataset(inp_t[:n_train], tgt_t[:n_train])
    val_ds   = TensorDataset(inp_t[n_train:], tgt_t[n_train:])
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

    # Optional divergence-free constraint (vorticity is scalar; use as a
    # spectral band filter demonstration)
    constraint = None
    if args.constraint == "divergence_free":
        from solaris.nn import DivergenceFreeProjection2d
        log.info("Note: divergence-free constraint not applicable to scalar vorticity — skipping.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    log.info("Starting training …")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, val_l2 = 0.0, 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * len(xb)
                val_l2   += relative_l2_error(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        val_l2   /= len(val_ds)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={tr_loss:.3e} | val={val_loss:.3e} | "
            f"rel-L2={val_l2:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best_fno.pt",
                model, optimizer, scheduler, epoch, val_loss,
                extra={"H": args.H, "W": args.W, "nu": args.nu},
            )

    log.info(f"Done. Best val loss: {best_val:.4e} → {ckpt_dir}/best_fno.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train FNO on 2-D Navier-Stokes vorticity")
    p.add_argument("--device",          default="cpu")
    p.add_argument("--H",               type=int,   default=64)
    p.add_argument("--W",               type=int,   default=64)
    p.add_argument("--nu",              type=float, default=1e-3)
    p.add_argument("--dt",              type=float, default=0.01)
    p.add_argument("--n_steps",         type=int,   default=20)
    p.add_argument("--forecast_steps",  type=int,   default=4)
    p.add_argument("--n_sims",          type=int,   default=100)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--hidden",          type=int,   default=64)
    p.add_argument("--modes",           type=int,   default=16)
    p.add_argument("--n_layers",        type=int,   default=4)
    p.add_argument("--constraint",      default="none", choices=["none", "divergence_free"])
    p.add_argument("--cache",           default="data/ns_dataset.npz")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    train(p.parse_args())
