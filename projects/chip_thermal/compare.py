# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — head-to-head: FD solver vs trained FNO surrogate.

For N unseen chip designs this script:
  1. Runs the finite-difference solver                 (baseline, slow)
  2. Runs the FNO surrogate on GPU/CPU                 (fast)
  3. Prints a timing table and error statistics
  4. Saves a side-by-side figure (compare.png)

Usage
-----
python compare.py --checkpoint checkpoints/best_fno.pt --device cuda --n 20
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from solaris.models.fno import FNO
from solaris.utils import get_logger
from solaris.utils.checkpoint import load_checkpoint
from solver import random_power_map, solve_heat


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    resolution = ckpt.get("resolution", 64)

    # Rebuild model with same architecture
    model = FNO(in_channels=1, out_channels=1, hidden_channels=64,
                n_layers=4, modes=16, dim=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, resolution


def run_comparison(args):
    log = get_logger("compare")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load norm stats
    stats = np.load(Path(args.checkpoint).parent / "norm_stats.npz")
    Q_mean, Q_std = float(stats["Q_mean"]), float(stats["Q_std"])
    T_mean, T_std = float(stats["T_mean"]), float(stats["T_std"])

    # Load trained FNO
    model, resolution = load_model(args.checkpoint, device)
    log.info(f"Loaded FNO | resolution={resolution} | params={model.num_parameters():,}")

    rng = np.random.default_rng(args.seed)

    solver_times, fno_times, rel_l2s = [], [], []
    samples_to_plot = []

    log.info(f"\nRunning {args.n} comparisons …\n")
    log.info(f"{'#':>4}  {'Solver (s)':>12}  {'FNO (ms)':>10}  {'Speedup':>9}  {'Rel-L2':>8}")
    log.info("-" * 55)

    for i in range(args.n):
        Q = random_power_map(resolution, resolution, rng=rng)

        # ── Baseline: FD solver ──────────────────────────────────────────
        T_ref, n_iters, t_solver = solve_heat(Q, max_iter=20_000, tol=1e-5)
        solver_times.append(t_solver)

        # ── FNO surrogate ────────────────────────────────────────────────
        Q_norm = (Q - Q_mean) / Q_std
        Q_t = torch.as_tensor(Q_norm[None, None], dtype=torch.float32).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            T_pred_norm = model(Q_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fno = time.perf_counter() - t0
        fno_times.append(t_fno)

        # Denormalise
        T_pred = (T_pred_norm.cpu().numpy()[0, 0] * T_std + T_mean)

        # Error (relative L2 on interior)
        diff = T_pred[1:-1, 1:-1] - T_ref[1:-1, 1:-1]
        rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(T_ref[1:-1, 1:-1]) + 1e-8)
        rel_l2s.append(rel_l2)

        speedup = t_solver / t_fno
        log.info(f"{i+1:>4}  {t_solver:>12.3f}  {t_fno*1000:>10.2f}  {speedup:>8.0f}×  {rel_l2:>8.4f}")

        if i < args.n_plot:
            samples_to_plot.append((Q, T_ref, T_pred, rel_l2))

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info(f"  Solver   avg: {np.mean(solver_times):.3f}s  (min {np.min(solver_times):.3f}s)")
    log.info(f"  FNO      avg: {np.mean(fno_times)*1000:.2f}ms  (min {np.min(fno_times)*1000:.2f}ms)")
    log.info(f"  Speedup  avg: {np.mean(solver_times)/np.mean(fno_times):.0f}×")
    log.info(f"  Rel-L2   avg: {np.mean(rel_l2s):.4f}  (max {np.max(rel_l2s):.4f})")

    # ── Plot ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_plot = len(samples_to_plot)
        fig = plt.figure(figsize=(15, 4 * n_plot))
        gs = gridspec.GridSpec(n_plot, 4, figure=fig, hspace=0.4, wspace=0.3)

        for row, (Q, T_ref, T_pred, rl2) in enumerate(samples_to_plot):
            err = np.abs(T_pred - T_ref)
            vmin_T = min(T_ref.min(), T_pred.min())
            vmax_T = max(T_ref.max(), T_pred.max())

            ax0 = fig.add_subplot(gs[row, 0])
            im0 = ax0.imshow(Q, cmap="hot", origin="lower")
            ax0.set_title(f"Sample {row+1}: Power map")
            plt.colorbar(im0, ax=ax0, label="W/m²")

            ax1 = fig.add_subplot(gs[row, 1])
            im1 = ax1.imshow(T_ref, cmap="inferno", origin="lower", vmin=vmin_T, vmax=vmax_T)
            ax1.set_title(f"FD Solver ({solver_times[row]:.2f}s)")
            plt.colorbar(im1, ax=ax1, label="°C")

            ax2 = fig.add_subplot(gs[row, 2])
            im2 = ax2.imshow(T_pred, cmap="inferno", origin="lower", vmin=vmin_T, vmax=vmax_T)
            ax2.set_title(f"FNO Surrogate ({fno_times[row]*1000:.1f}ms)")
            plt.colorbar(im2, ax=ax2, label="°C")

            ax3 = fig.add_subplot(gs[row, 3])
            im3 = ax3.imshow(err, cmap="RdBu_r", origin="lower")
            ax3.set_title(f"|Error|  (rel-L2={rl2:.4f})")
            plt.colorbar(im3, ax=ax3, label="°C")

        avg_speedup = np.mean(solver_times) / np.mean(fno_times)
        fig.suptitle(
            f"Chip Thermal: FD Solver vs FNO Surrogate  |  "
            f"Avg speedup: {avg_speedup:.0f}×  |  Avg rel-L2: {np.mean(rel_l2s):.4f}",
            fontsize=13, fontweight="bold",
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        log.info(f"\nFigure saved → {out}")
        plt.close(fig)
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_fno.pt")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n",          type=int, default=20,  help="Number of test cases")
    p.add_argument("--n_plot",     type=int, default=3,   help="Samples to include in figure")
    p.add_argument("--seed",       type=int, default=999, help="RNG seed (different from training)")
    p.add_argument("--output",     default="results/compare.png")
    run_comparison(p.parse_args())
