# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train a 3-D transient FNO surrogate for chip heat conduction.

The model maps a volumetric power map Q(x,y,z) and a normalised time t to
the temperature rise ΔT(x,y,z,t):

    Input  : [Q_norm (1,32,32,16),  t_broadcast (1,32,32,16)]  → (B, 2, 32, 32, 16)
    Output : ΔT_norm (1,32,32,16)                               → (B, 1, 32, 32, 16)

Normalisation (per-simulation):
    Q_norm   = Q / Q.max()              (inputs in [0, 1])
    ΔT_norm  = (T − T_ambient) / Q.max()
    t_norm   = t / t_end                (in [0, 1])

At inference:  T_pred = model_output × Q.max() + T_ambient

Usage
-----
# Quick smoke-test  (10 sims, 5 epochs)
python3.11 projects/chip_thermal/train.py --n_sim 10 --epochs 5

# Full training on GPU
python3.11 projects/chip_thermal/train.py --device cuda --n_sim 300 --epochs 80
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from repo root or from this directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaris.models.fno import FNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger, save_checkpoint
from solaris.utils.training import EarlyStopping, GradientClipper
from solver import random_power_map_3d, solve_heat_3d, T_AMBIENT_3D


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    n_sim: int,
    Nx: int = 32,
    Ny: int = 32,
    Nz: int = 16,
    t_end: float = 0.01,
    n_times: int = 8,
    seed: int = 0,
):
    """Generate (Q_norm, t_broadcast, ΔT_norm) triples for 3-D transient heat.

    For each simulation we produce ``n_times`` training samples — one per
    snapshot time — giving ``n_sim × n_times`` total samples.

    Returns
    -------
    Q_arr : ndarray (N, 1, Nx, Ny, Nz)  normalised power map (repeated per time)
    t_arr : ndarray (N, 1, Nx, Ny, Nz)  normalised time broadcast to grid
    T_arr : ndarray (N, 1, Nx, Ny, Nz)  normalised temperature rise
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    times = np.linspace(t_end / n_times, t_end, n_times)

    Q_list, t_list, T_list = [], [], []

    log.info(f"Generating {n_sim} simulations × {n_times} snapshots "
             f"on {Nx}×{Ny}×{Nz} grid …")
    t_wall = time.perf_counter()

    for i in range(n_sim):
        Q = random_power_map_3d(Nx, Ny, Nz, rng=rng)      # (Nx, Ny, Nz)
        snaps, _times, _ = solve_heat_3d(Q, t_end=t_end, n_snapshots=n_times)
        # snaps: (n_times, Nx, Ny, Nz)

        peak_Q = float(Q.max()) + 1e-8

        Q_norm = Q / peak_Q                                # [0, 1]

        for j in range(n_times):
            T_rise = snaps[j] - T_AMBIENT_3D              # °C rise
            T_norm = T_rise / peak_Q                       # same scale as Q

            t_norm = float(times[j] / t_end)
            t_bc = np.full((1, Nx, Ny, Nz), t_norm, dtype=np.float32)

            Q_list.append(Q_norm[None].astype(np.float32))   # (1,Nx,Ny,Nz)
            t_list.append(t_bc)
            T_list.append(T_norm[None].astype(np.float32))   # (1,Nx,Ny,Nz)

        if (i + 1) % max(1, n_sim // 5) == 0:
            elapsed = time.perf_counter() - t_wall
            log.info(f"  {i+1}/{n_sim} sims done — {elapsed:.1f}s elapsed")

    log.info(f"Dataset complete in {time.perf_counter()-t_wall:.1f}s")
    return np.stack(Q_list), np.stack(t_list), np.stack(T_list)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    log = get_logger("train")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ────────────────────────────────────────────────────────────────
    cache = Path(args.cache)
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        Q_arr, t_arr, T_arr = d["Q"], d["t"], d["T"]
        n_expected = args.n_sim * args.n_times
        if len(Q_arr) < n_expected:
            log.info(f"Cache has {len(Q_arr)} samples, need {n_expected} — regenerating")
            Q_arr = None
    else:
        Q_arr = None

    if Q_arr is None:
        Q_arr, t_arr, T_arr = generate_dataset(
            args.n_sim, n_times=args.n_times, t_end=args.t_end, seed=42
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Q=Q_arr, t=t_arr, T=T_arr, t_end=args.t_end)
        log.info(f"Saved dataset → {cache}")

    # ── Norm stats (T_ambient and t_end are all that's needed for inference) ─
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ckpt_dir / "norm_stats_3d.npz",
        T_ambient=T_AMBIENT_3D,
        t_end=args.t_end,
    )

    # ── Concatenate [Q, t] into 2-channel input ──────────────────────────────
    inp = np.concatenate([Q_arr, t_arr], axis=1)    # (N, 2, Nx, Ny, Nz)

    n_total = len(inp)
    n_val   = max(1, int(n_total * 0.15))
    idx     = np.random.default_rng(0).permutation(n_total)

    def to_t(a):
        return torch.as_tensor(a, dtype=torch.float32)

    train_ds = TensorDataset(to_t(inp[idx[n_val:]]), to_t(T_arr[idx[n_val:]]))
    val_ds   = TensorDataset(to_t(inp[idx[:n_val]]), to_t(T_arr[idx[:n_val]]))

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, args.batch_size,
        pin_memory=(device.type == "cuda"), num_workers=0,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = FNO(
        in_channels=2, out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        modes=args.modes,
        dim=3,
    ).to(device)
    log.info(f"FNO-3D parameters: {model.num_parameters():,}")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn  = nn.MSELoss()
    stopper  = EarlyStopping(patience=15, min_delta=1e-6, mode="min")
    clipper  = GradientClipper(max_norm=1.0)

    best = float("inf")
    log.info("Starting training …")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            clipper(model)
            opt.step()
            tr += loss.item() * len(x)
        tr /= len(train_ds)
        sched.step()

        model.eval()
        vl = vr = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                vl += loss_fn(p, y).item() * len(x)
                vr += relative_l2_error(p, y).item() * len(x)
        vl /= len(val_ds)
        vr /= len(val_ds)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={tr:.3e} | val={vl:.3e} | rel-L2={vr:.4f} | "
            f"lr={sched.get_last_lr()[0]:.2e}"
        )

        if vl < best:
            best = vl
            save_checkpoint(
                ckpt_dir / "best_fno_3d.pt",
                model, opt, sched, epoch, vl,
                extra={
                    "hidden_channels": args.hidden,
                    "n_layers": args.n_layers,
                    "modes": args.modes,
                },
            )

        if stopper.step(vl):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Done.  Best val={best:.4e} → {ckpt_dir}/best_fno_3d.pt")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train 3-D transient chip-thermal FNO")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--n_sim",          type=int,   default=300,  help="Simulations to generate")
    p.add_argument("--n_times",        type=int,   default=8,    help="Snapshots per simulation")
    p.add_argument("--t_end",          type=float, default=0.01, help="Physical end time [s] (10 ms)")
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden",         type=int,   default=32)
    p.add_argument("--modes",          type=int,   default=8)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--cache",          default="data/thermal_3d_dataset.npz")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    train(p.parse_args())
