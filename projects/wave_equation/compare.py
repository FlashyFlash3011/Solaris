# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Compare FD wave solver vs trained FNO surrogate.

Panels: wave speed preservation (peak position vs time) and amplitude decay.

Usage
-----
python compare.py --checkpoint_dir checkpoints --n_test 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from solver import random_gaussian_ic, solve_wave, solve_wave_snapshots  # noqa: E402
from train import build_timestep_channel  # noqa: E402

from solaris.models.fno import FNO
from solaris.utils import get_logger


def compare(args):
    log = get_logger("wave_compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / "best_wave_fno.pt"
    stats_path = ckpt_dir / "norm_stats.npz"

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return

    stats = np.load(stats_path)
    in_mean, in_std = float(stats["in_mean"]), float(stats["in_std"])
    tgt_mean, tgt_std = float(stats["tgt_mean"]), float(stats["tgt_std"])

    ckpt = torch.load(ckpt_path, map_location=device)
    model = FNO(in_channels=3, out_channels=1, hidden_channels=64, n_layers=4, modes=16, dim=2)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    log.info(f"Loaded FNO from {ckpt_path}")

    H, W = 64, 64
    rng = np.random.default_rng(1234)
    solver_times, fno_times, l2_errors = [], [], []

    for i in range(args.n_test):
        u0, v0 = random_gaussian_ic(H, W, rng=rng)

        t0 = time.perf_counter()
        u_gt, _ = solve_wave(u0, v0, c=1.0, dt=5e-4, n_steps=400)
        solver_times.append(time.perf_counter() - t0)

        # FNO: step autoregressively from t=0
        t_ch = build_timestep_channel(0.0, H, W).numpy()
        inp = np.concatenate([u0[None], v0[None], t_ch], axis=0)[None]  # (1,3,H,W)
        inp_norm = (inp - in_mean) / in_std

        t0 = time.perf_counter()
        with torch.no_grad():
            pred_norm = model(torch.as_tensor(inp_norm, dtype=torch.float32).to(device))
        fno_times.append(time.perf_counter() - t0)

        pred = pred_norm.squeeze().cpu().numpy() * tgt_std + tgt_mean
        l2 = np.linalg.norm(pred - u_gt) / (np.linalg.norm(u_gt) + 1e-8)
        l2_errors.append(l2)

    solver_ms = np.mean(solver_times) * 1000
    fno_ms = np.mean(fno_times) * 1000
    speedup = solver_ms / (fno_ms + 1e-9)
    mean_l2 = np.mean(l2_errors)

    log.info(f"Solver:  {solver_ms:.1f} ms/sample")
    log.info(f"FNO:     {fno_ms:.2f} ms/sample  ({speedup:.1f}× speedup)")
    log.info(f"Mean rel-L2 error: {mean_l2:.4f}")

    # Visualisation — wave propagation snapshots
    try:
        import matplotlib.pyplot as plt

        rng2 = np.random.default_rng(9999)
        u0_vis, v0_vis = random_gaussian_ic(H, W, rng=rng2)
        u_snaps, _ = solve_wave_snapshots(u0_vis, v0_vis, c=1.0, dt=5e-4, n_steps=400, n_snapshots=4)

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        vmax = np.abs(u_snaps).max()
        kw = dict(cmap="seismic", vmin=-vmax, vmax=vmax, origin="lower")

        # Top row: solver snapshots
        for col, snap in enumerate(u_snaps):
            axes[0, col].imshow(snap, **kw)
            axes[0, col].set_title(f"Solver t={col}")
            axes[0, col].axis("off")

        # Bottom row: FNO one-step predictions from each snapshot
        for col in range(len(u_snaps)):
            t_ch_c = build_timestep_channel(col * 0.2, H, W).numpy()
            inp_c = np.concatenate([u_snaps[col][None],
                                    np.zeros((1, H, W)),
                                    t_ch_c], axis=0)[None]
            inp_c_norm = (inp_c - in_mean) / in_std
            with torch.no_grad():
                pr_n = model(torch.as_tensor(inp_c_norm, dtype=torch.float32).to(device))
            pr = pr_n.squeeze().cpu().numpy() * tgt_std + tgt_mean
            axes[1, col].imshow(pr, **kw)
            axes[1, col].set_title(f"FNO t={col}")
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Solver", rotation=90, labelpad=4)
        axes[1, 0].set_ylabel("FNO", rotation=90, labelpad=4)
        plt.suptitle(f"Wave Equation: FD vs FNO  (mean rel-L2={mean_l2:.3f})")
        plt.tight_layout()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=120)
        log.info(f"Saved {out}")
    except ImportError:
        log.warning("matplotlib not installed — skipping visualisation")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",          default="cpu")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--n_test",          type=int, default=20)
    p.add_argument("--output",          default="comparison.png")
    compare(p.parse_args())
