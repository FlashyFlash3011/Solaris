# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train an FNO surrogate for the 2-D wave equation.

Input:  [u(t), v(t), t_embed]  — 3 channels
Output: u(t+Δt)               — 1 channel

Uses SinusoidalTimestepEmbedding from solaris.nn.embeddings to encode the
simulation time as an additional spatial channel.

Usage
-----
python train.py --epochs 5 --n_sims 20
python train.py --device cuda --epochs 50 --n_sims 500
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from solver import random_gaussian_ic, solve_wave_snapshots  # noqa: E402

from solaris.models.fno import FNO
from solaris.metrics import relative_l2_error
from solaris.nn.embeddings import SinusoidalTimestepEmbedding
from solaris.utils import get_logger, save_checkpoint


def build_timestep_channel(t: float, H: int, W: int, embed_dim: int = 32) -> torch.Tensor:
    """Create a (1, H, W) channel filled with the sinusoidal timestep embedding.

    The embedding is reduced to a scalar via sum and broadcast spatially.
    """
    embedder = SinusoidalTimestepEmbedding(dim=embed_dim)
    t_tensor = torch.tensor([t], dtype=torch.float32)
    emb = embedder(t_tensor).sum()  # scalar
    return emb.expand(1, H, W)


def generate_dataset(
    n_sims: int,
    H: int,
    W: int,
    c: float,
    dt_solver: float,
    n_steps: int,
    n_snapshots: int,
    seed: int = 0,
) -> tuple:
    """Generate (input_3ch, u_next) pairs from multiple wave simulations.

    Returns
    -------
    inputs : ndarray  shape (N, 3, H, W)   [u, v, t_embed]
    targets : ndarray  shape (N, 1, H, W)  [u(t+Δt)]
    """
    log = get_logger("wave_generate")
    rng = np.random.default_rng(seed)
    inputs_list, targets_list = [], []

    dt_snap = (n_steps * dt_solver) / (n_snapshots - 1)

    log.info(f"Generating {n_sims} wave simulations at {H}×{W} …")
    t0 = time.perf_counter()
    for i in range(n_sims):
        u0, v0 = random_gaussian_ic(H, W, rng=rng)
        u_snaps, v_snaps = solve_wave_snapshots(
            u0, v0, c=c, dt=dt_solver, n_steps=n_steps, n_snapshots=n_snapshots
        )
        for t_idx in range(n_snapshots - 1):
            t_phys = t_idx * dt_snap
            t_ch = build_timestep_channel(t_phys, H, W).numpy()
            inp = np.concatenate([
                u_snaps[t_idx:t_idx+1],
                v_snaps[t_idx:t_idx+1],
                t_ch,
            ], axis=0)  # (3, H, W)
            tgt = u_snaps[t_idx + 1:t_idx + 2]  # (1, H, W)
            inputs_list.append(inp)
            targets_list.append(tgt)

        if (i + 1) % max(1, n_sims // 5) == 0:
            log.info(f"  {i+1}/{n_sims} — {time.perf_counter()-t0:.1f}s")

    inputs_arr = np.stack(inputs_list).astype(np.float32)
    targets_arr = np.stack(targets_list).astype(np.float32)
    log.info(f"Dataset: {len(inputs_arr)} pairs, {time.perf_counter()-t0:.1f}s")
    return inputs_arr, targets_arr


def train(args):
    log = get_logger("wave_train")
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
            args.n_sims, args.H, args.W, args.c,
            args.dt, args.n_steps, args.n_snapshots, seed=42,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, inputs=inputs_arr, targets=targets_arr)
        log.info(f"Saved dataset → {cache}")

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

    # ── Model: 3 input channels (u, v, t_embed), 1 output (u_next) ──
    model = FNO(
        in_channels=3, out_channels=1,
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
                ckpt_dir / "best_wave_fno.pt",
                model, optimizer, scheduler, epoch, val_loss,
                extra={"H": args.H, "W": args.W, "c": args.c},
            )

    log.info(f"Done. Best val loss: {best_val:.4e} → {ckpt_dir}/best_wave_fno.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train FNO on 2-D wave equation")
    p.add_argument("--device",          default="cpu")
    p.add_argument("--H",               type=int,   default=64)
    p.add_argument("--W",               type=int,   default=64)
    p.add_argument("--c",               type=float, default=1.0,  help="Wave speed")
    p.add_argument("--dt",              type=float, default=5e-4, help="Solver time step")
    p.add_argument("--n_steps",         type=int,   default=400,  help="Solver steps per sim")
    p.add_argument("--n_snapshots",     type=int,   default=10,   help="Snapshots per sim")
    p.add_argument("--n_sims",          type=int,   default=100)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--hidden",          type=int,   default=64)
    p.add_argument("--modes",           type=int,   default=16)
    p.add_argument("--n_layers",        type=int,   default=4)
    p.add_argument("--cache",           default="data/wave_dataset.npz")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    train(p.parse_args())
