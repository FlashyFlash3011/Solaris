# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train a time-aware FNO surrogate for 2-D heat diffusion in water.

The model takes:
    input  : T(t=0)  — initial temperature field  (1, H, W)
    + time : t       — how far ahead to predict    scalar → broadcast to (1, H, W)
    output : T(t)    — temperature field at time t (1, H, W)

The time channel is concatenated to the spatial field, so the FNO
receives a 2-channel input: [T_0, t_broadcast].

Usage
-----
python train.py --device cuda --epochs 80
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaris.models.fno import FNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger, save_checkpoint
from solver import make_initial_field, solve_diffusion, ALPHA_WATER


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(n: int, resolution: int, t_end: float, n_times: int, seed: int = 0):
    """Generate (T0, t, T_t) triples.

    Returns
    -------
    T0_arr   : (N*n_times, 1, H, W)  initial field (repeated for each time)
    t_arr    : (N*n_times, 1, H, W)  time channel (broadcast scalar)
    Tt_arr   : (N*n_times, 1, H, W)  target field at time t
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    times = np.linspace(t_end / n_times, t_end, n_times)  # skip t=0

    T0_list, t_list, Tt_list = [], [], []
    log.info(f"Generating {n} simulations × {n_times} snapshots at {resolution}²…")
    t_wall = time.perf_counter()

    for i in range(n):
        T0 = make_initial_field(resolution, resolution, rng=rng)
        snaps, snap_times, _, _ = solve_diffusion(T0, t_end=t_end, n_snapshots=n_times + 1)
        # snaps[0] is t=0, snaps[1:] are the targets
        for j in range(n_times):
            T0_list.append(T0[None])                                    # (1, H, W)
            t_norm = times[j] / t_end                                   # normalise to [0,1]
            t_broadcast = np.full((1, resolution, resolution), t_norm, dtype=np.float32)
            t_list.append(t_broadcast)
            Tt_list.append(snaps[j + 1][None])

        if (i + 1) % max(1, n // 5) == 0:
            elapsed = time.perf_counter() - t_wall
            log.info(f"  {i+1}/{n} — {elapsed:.1f}s")

    log.info(f"Done in {time.perf_counter()-t_wall:.1f}s")
    return (
        np.stack(T0_list),
        np.stack(t_list),
        np.stack(Tt_list),
        times,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    log = get_logger("train")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    cache = Path(args.cache)
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        T0_arr, t_arr, Tt_arr = d["T0"], d["t"], d["Tt"]
        times = d["times"]
    else:
        T0_arr, t_arr, Tt_arr, times = generate_dataset(
            args.n_sims, args.resolution, args.t_end, args.n_times, seed=42
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, T0=T0_arr, t=t_arr, Tt=Tt_arr, times=times)
        log.info(f"Saved → {cache}")

    # Work with temperature RISE above ambient.
    # This removes the large constant offset (ambient=20°C) so the model
    # only needs to learn the evolving hot-spot shape — much easier.
    T_ambient = 20.0  # °C — matches make_initial_field default
    T0_rise = T0_arr - T_ambient          # (N, 1, H, W)  range [0, ~55°C]
    Tt_rise = Tt_arr - T_ambient          # (N, 1, H, W)  range [0, ~55°C] decaying

    # Normalise by per-sample initial peak so every sample lives in [0, 1]
    peak = T0_rise.max(axis=(1, 2, 3), keepdims=True) + 1e-8   # (N, 1, 1, 1)
    T0_n = T0_rise / peak
    Tt_n = Tt_rise / peak   # same scale — target is also in [0, 1] but decaying
    # t_arr already in [0, 1]

    # Stack [T0_norm, t_channel] → 2-channel input
    inp = np.concatenate([T0_n, t_arr], axis=1)   # (N, 2, H, W)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.checkpoint_dir) / "norm_stats.npz",
             T_ambient=T_ambient, t_end=args.t_end, times=times)

    n_total = len(inp)
    n_val   = int(n_total * 0.15)
    idx     = np.random.default_rng(0).permutation(n_total)

    def to_tensor(a): return torch.as_tensor(a, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(inp[idx[n_val:]]), to_tensor(Tt_n[idx[n_val:]]))
    val_ds   = TensorDataset(to_tensor(inp[idx[:n_val]]), to_tensor(Tt_n[idx[:n_val]]))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"), num_workers=2)
    val_loader   = DataLoader(val_ds, args.batch_size,
                              pin_memory=(device.type == "cuda"), num_workers=2)

    model = FNO(in_channels=2, out_channels=1,
                hidden_channels=args.hidden, n_layers=args.n_layers,
                modes=args.modes, dim=2).to(device)
    log.info(f"FNO params: {model.num_parameters():,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(x)
        tr /= len(train_ds)
        sched.step()

        model.eval()
        vl, vr = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                vl += loss_fn(p, y).item() * len(x)
                vr += relative_l2_error(p, y).item() * len(x)
        vl /= len(val_ds); vr /= len(val_ds)

        log.info(f"Epoch {epoch:3d}/{args.epochs} | train={tr:.3e} | val={vl:.3e} | rel-L2={vr:.4f} | lr={sched.get_last_lr()[0]:.2e}")

        if vl < best:
            best = vl
            save_checkpoint(Path(args.checkpoint_dir) / "best_fno.pt",
                            model, opt, sched, epoch, vl)

    log.info(f"Done. Best val={best:.4e}  →  {args.checkpoint_dir}/best_fno.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",         default="cuda")
    p.add_argument("--resolution",     type=int,   default=128)
    p.add_argument("--n_sims",         type=int,   default=600,  help="Number of full simulations")
    p.add_argument("--n_times",        type=int,   default=8,    help="Time snapshots per sim")
    p.add_argument("--t_end",          type=float, default=0.5,  help="Seconds of physical time")
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden",         type=int,   default=64)
    p.add_argument("--modes",          type=int,   default=16)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--cache",          default="data/water_heat_dataset.npz")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    train(p.parse_args())
