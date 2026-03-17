# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Compare spectral NS solver vs trained FNO surrogate.

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
from solver import random_vorticity_ic, solve_ns  # noqa: E402

from solaris.models.fno import FNO
from solaris.utils import get_logger


def compare(args):
    log = get_logger("ns_compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / "best_fno.pt"
    stats_path = ckpt_dir / "norm_stats.npz"

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return

    # Load normalisation stats
    stats = np.load(stats_path)
    in_mean, in_std = float(stats["in_mean"]), float(stats["in_std"])
    tgt_mean, tgt_std = float(stats["tgt_mean"]), float(stats["tgt_std"])

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device)
    model = FNO(in_channels=1, out_channels=1, hidden_channels=64, n_layers=4, modes=16, dim=2)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    log.info(f"Loaded FNO from {ckpt_path}")

    rng = np.random.default_rng(999)
    solver_times, fno_times, l2_errors = [], [], []

    for i in range(args.n_test):
        omega0 = random_vorticity_ic(64, 64, n_modes=6, rng=rng)

        # Solver reference
        t0 = time.perf_counter()
        snaps = solve_ns(omega0, nu=1e-3, dt=0.01, n_steps=4, n_snapshots=5)
        solver_times.append(time.perf_counter() - t0)
        omega_t = snaps[0].unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        omega_gt = snaps[-1]                             # (H,W) ground truth

        # FNO inference
        inp_norm = (omega_t - in_mean) / in_std
        t0 = time.perf_counter()
        with torch.no_grad():
            pred_norm = model(inp_norm.to(device)).squeeze().cpu()
        fno_times.append(time.perf_counter() - t0)

        pred = pred_norm * tgt_std + tgt_mean
        l2 = (pred - omega_gt).norm() / (omega_gt.norm() + 1e-8)
        l2_errors.append(l2.item())

    solver_ms = np.mean(solver_times) * 1000
    fno_ms = np.mean(fno_times) * 1000
    speedup = solver_ms / (fno_ms + 1e-9)
    mean_l2 = np.mean(l2_errors)

    log.info(f"Solver:    {solver_ms:.1f} ms/sample")
    log.info(f"FNO:       {fno_ms:.2f} ms/sample  ({speedup:.1f}× speedup)")
    log.info(f"Mean rel-L2 error: {mean_l2:.4f}")

    # Visualise last test case
    try:
        import matplotlib.pyplot as plt

        omega0_last = random_vorticity_ic(64, 64, n_modes=6, rng=np.random.default_rng(999 + args.n_test - 1))
        snaps = solve_ns(omega0_last, nu=1e-3, dt=0.01, n_steps=4, n_snapshots=5)
        inp = snaps[0]; gt = snaps[-1]
        inp_norm = (inp.unsqueeze(0).unsqueeze(0) - in_mean) / in_std
        with torch.no_grad():
            pr_norm = model(inp_norm.to(device)).squeeze().cpu()
        pr = pr_norm * tgt_std + tgt_mean

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        vmax = max(gt.abs().max().item(), pr.abs().max().item())
        kw = dict(cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[0].imshow(inp.numpy(), **kw); axes[0].set_title("Input ω(t)")
        im1 = axes[1].imshow(gt.numpy(), **kw); axes[1].set_title("Ground Truth ω(t+Δt)")
        im2 = axes[2].imshow(pr.numpy(), **kw); axes[2].set_title("FNO Prediction")
        plt.colorbar(im2, ax=axes[2])
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
