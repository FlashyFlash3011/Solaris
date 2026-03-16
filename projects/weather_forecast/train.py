# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train an AFNO surrogate for 5-day weather forecasting.

Input  : (z500_day0, t850_day0)          — 2 channels, shape (2, 64, 128)
Output : (z500_dayN, t850_dayN) for N in [1,2,3,4,5]

We train a single model that predicts any lead time by concatenating
a normalised lead-time channel (like the heat diffusion project).
Total input channels = 3: z500, t850, lead_time_broadcast.

Usage
-----
python train.py --device cuda --n_sims 800 --epochs 100
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaris.models.afno import AFNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger, save_checkpoint
from data_gen import make_initial_state, simulate


# ── Dataset ───────────────────────────────────────────────────────────────────

def generate_dataset(n_sims: int, n_days: float, lead_days: list,
                     nlat: int, nlon: int, seed: int = 0):
    log = get_logger("generate")
    rng = np.random.default_rng(seed)

    X_list, lt_list, Y_list = [], [], []
    t0_wall = time.perf_counter()
    log.info(f"Generating {n_sims} simulations × {len(lead_days)} lead times …")

    for i in range(n_sims):
        z0, t0 = make_initial_state(nlat, nlon, rng=rng)
        z_snaps, t_snaps, snap_days, _, _ = simulate(
            z0, t0, n_days=n_days, n_snapshots=len(lead_days) + 1
        )
        # snaps[0] = day 0 (initial), snaps[1:] = targets
        for j, day in enumerate(lead_days):
            X_list.append(np.stack([z0, t0]))           # (2, H, W)
            lt_broadcast = np.full((1, nlat, nlon), day / n_days, dtype=np.float32)
            lt_list.append(lt_broadcast)                 # (1, H, W)
            Y_list.append(np.stack([z_snaps[j+1], t_snaps[j+1]]))  # (2, H, W)

        if (i + 1) % max(1, n_sims // 5) == 0:
            log.info(f"  {i+1}/{n_sims} — {time.perf_counter()-t0_wall:.1f}s")

    log.info(f"Done in {time.perf_counter()-t0_wall:.1f}s")
    return np.stack(X_list), np.stack(lt_list), np.stack(Y_list)


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_stats(X: np.ndarray, Y: np.ndarray):
    """Per-channel mean and std across all samples and spatial dims."""
    # X/Y shape: (N, 2, H, W)
    stats = {}
    for ch, name in enumerate(["z500", "t850"]):
        vals = np.concatenate([X[:, ch].ravel(), Y[:, ch].ravel()])
        stats[f"{name}_mean"] = float(vals.mean())
        stats[f"{name}_std"]  = float(vals.std()) + 1e-8
    return stats


def normalise(arr: np.ndarray, stats: dict, keys: list) -> np.ndarray:
    out = arr.copy()
    for ch, key in enumerate(keys):
        out[:, ch] = (arr[:, ch] - stats[f"{key}_mean"]) / stats[f"{key}_std"]
    return out


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    log = get_logger("train")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    lead_days = list(range(1, args.n_days + 1))   # [1, 2, 3, 4, 5]
    nlat, nlon = args.nlat, args.nlon

    cache = Path(args.cache)
    if cache.exists():
        log.info(f"Loading cache from {cache}")
        d = np.load(cache)
        X, LT, Y = d["X"], d["LT"], d["Y"]
    else:
        X, LT, Y = generate_dataset(args.n_sims, args.n_days, lead_days,
                                     nlat, nlon, seed=42)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, LT=LT, Y=Y)
        log.info(f"Saved → {cache}")

    # Normalise
    stats = compute_stats(X, Y)
    X_n = normalise(X, stats, ["z500", "t850"])
    Y_n = normalise(Y, stats, ["z500", "t850"])
    # Concatenate lead-time channel: (N, 3, H, W)
    inp = np.concatenate([X_n, LT], axis=1).astype(np.float32)
    tgt = Y_n.astype(np.float32)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.checkpoint_dir) / "norm_stats.npz",
             n_days=args.n_days, nlat=nlat, nlon=nlon, **stats)

    rng_idx = np.random.default_rng(0).permutation(len(inp))
    n_val = int(len(inp) * 0.15)
    tr_idx, va_idx = rng_idx[n_val:], rng_idx[:n_val]

    def ds(idx): return TensorDataset(
        torch.as_tensor(inp[idx]), torch.as_tensor(tgt[idx]))
    train_loader = DataLoader(ds(tr_idx), args.batch_size, shuffle=True,
                              pin_memory=(device.type == "cuda"), num_workers=2)
    val_loader   = DataLoader(ds(va_idx), args.batch_size,
                              pin_memory=(device.type == "cuda"), num_workers=2)

    # AFNO: patch size 4 on a 64×128 grid → 16×32 = 512 tokens
    model = AFNO(
        in_channels=3, out_channels=2,
        img_size=(nlat, nlon),
        patch_size=args.patch_size,
        hidden_size=args.hidden,
        n_layers=args.n_layers,
        num_blocks=8,
    ).to(device)
    log.info(f"AFNO params: {model.num_parameters():,}")

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
        tr /= len(tr_idx)
        sched.step()

        model.eval()
        vl = vr = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                vl += loss_fn(p, y).item() * len(x)
                vr += relative_l2_error(p, y).item() * len(x)
        vl /= len(va_idx); vr /= len(va_idx)

        log.info(f"Epoch {epoch:3d}/{args.epochs} | train={tr:.3e} | val={vl:.3e} | rel-L2={vr:.4f} | lr={sched.get_last_lr()[0]:.2e}")

        if vl < best:
            best = vl
            save_checkpoint(Path(args.checkpoint_dir) / "best_afno.pt",
                            model, opt, sched, epoch, vl)

    log.info(f"Done. Best val={best:.4e}  →  {args.checkpoint_dir}/best_afno.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",         default="cuda")
    p.add_argument("--nlat",           type=int,   default=64)
    p.add_argument("--nlon",           type=int,   default=128)
    p.add_argument("--n_sims",         type=int,   default=800)
    p.add_argument("--n_days",         type=int,   default=5)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=5e-4)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--n_layers",       type=int,   default=6)
    p.add_argument("--patch_size",     type=int,   default=4)
    p.add_argument("--cache",          default="data/weather_dataset.npz")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    train(p.parse_args())
