# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — head-to-head: scipy FD solver vs trained FNO surrogate.

The traditional solver must assemble and solve a ~175 k × 175 k sparse linear
system at 421×421 resolution.  The FNO returns a prediction in a single
forward pass at the subsampled resolution (~70×70).

For N unseen Darcy instances this script:
  1. Runs the scipy FD solver at full 421×421 resolution     (baseline)
  2. Runs the FNO surrogate on GPU/CPU at subsampled resolution (fast)
  3. Prints a timing table and error statistics
  4. Saves a side-by-side figure (results/compare.png)

Usage
-----
python compare.py --checkpoint checkpoints/best_fno.pt --device cuda --n 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solaris.models.fno import FNO
from solaris.utils import get_logger
from solver import load_darcy_data, solve_darcy_fd


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_channels = ckpt.get("hidden_channels", 64)
    n_layers        = ckpt.get("n_layers",         4)
    modes           = ckpt.get("modes",            12)
    subsample       = ckpt.get("subsample",         1)
    resolution      = ckpt.get("resolution",       128)
    model = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        modes=modes,
        dim=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, subsample, resolution


# ─── Main ────────────────────────────────────────────────────────────────────

def run_comparison(args):
    log = get_logger("compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load norm stats ──
    stats = np.load(Path(args.checkpoint).parent / "norm_stats.npz")
    a_mean, a_std = float(stats["a_mean"]), float(stats["a_std"])
    u_mean, u_std = float(stats["u_mean"]), float(stats["u_std"])

    # ── Load trained FNO ──
    model, subsample, resolution = load_model(args.checkpoint, device)
    log.info(f"Loaded FNO  |  resolution={resolution}  |  subsample={subsample}  |  params={model.num_parameters():,}")

    # ── GPU warm-up ──
    if device.type == "cuda":
        dummy_H = (resolution + subsample - 1) // subsample
        dummy = torch.zeros(1, 1, dummy_H, dummy_H, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    # ── Load test data ──
    log.info(f"Loading test instances from Darcy {resolution}×{resolution} dataset …")
    _, _, a_test_full, u_test_full = load_darcy_data(
        n_train=5, n_test=args.n + 10, resolution=resolution, subsample=1
    )  # full-resolution for FD solver

    # Sub-sample in-memory for FNO input (avoids a second dataset load)
    a_test_sub = a_test_full[:, :, ::subsample, ::subsample]
    u_test_sub = u_test_full[:, :, ::subsample, ::subsample]

    solver_times, fno_times, rel_l2s = [], [], []
    samples_to_plot = []

    H_sub = a_test_sub.shape[-1]
    log.info(f"\nFD solver resolution: {resolution}×{resolution} | FNO resolution: {H_sub}×{H_sub}")
    log.info(f"Running {args.n} comparisons …\n")
    log.info(f"{'#':>4}  {'FD Solver (s)':>14}  {'FNO (ms)':>10}  {'Speedup':>9}  {'Rel-L2':>8}")
    log.info("-" * 57)

    for i in range(args.n):
        a_full = a_test_full[i, 0]   # (421, 421)
        a_sub  = a_test_sub[i, 0]    # (H_sub, H_sub)
        u_ref  = u_test_sub[i, 0]    # ground truth at sub resolution

        # ── Baseline: scipy FD solver (full 421×421) ──
        _, t_solver = solve_darcy_fd(a_full)
        solver_times.append(t_solver)

        # ── FNO surrogate ──
        a_norm = (a_sub - a_mean) / a_std
        a_t    = torch.as_tensor(a_norm[None, None], dtype=torch.float32).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            u_pred_norm = model(a_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fno = time.perf_counter() - t0
        fno_times.append(t_fno)

        u_pred = u_pred_norm.cpu().numpy()[0, 0] * u_std + u_mean

        # Error
        diff   = u_pred - u_ref
        rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(u_ref) + 1e-8)
        rel_l2s.append(rel_l2)

        speedup = t_solver / t_fno
        log.info(
            f"{i+1:>4}  {t_solver:>14.3f}  {t_fno*1000:>10.2f}"
            f"  {speedup:>8.0f}×  {rel_l2:>8.4f}"
        )

        if i < args.n_plot:
            samples_to_plot.append((a_sub, u_ref, u_pred, rel_l2))

    # ── Summary ──
    log.info("\n" + "=" * 57)
    log.info(f"  FD Solver  avg: {np.mean(solver_times):.3f}s  (min {np.min(solver_times):.3f}s)")
    log.info(f"  FNO        avg: {np.mean(fno_times)*1000:.2f}ms  (min {np.min(fno_times)*1000:.2f}ms)")
    log.info(f"  Speedup    avg: {np.mean(solver_times)/np.mean(fno_times):.0f}×")
    log.info(f"  Rel-L2     avg: {np.mean(rel_l2s):.4f}  (max {np.max(rel_l2s):.4f})")

    # ── Plot ──
    try:
        import matplotlib.pyplot as plt

        a0, u_ref0, u_pred0, rl2 = samples_to_plot[0]
        err0 = np.abs(u_pred0 - u_ref0)
        vmin_u = min(u_ref0.min(), u_pred0.min())
        vmax_u = max(u_ref0.max(), u_pred0.max())

        avg_solver_s  = np.mean(solver_times)
        avg_fno_ms    = np.mean(fno_times) * 1000
        avg_speedup   = avg_solver_s / (avg_fno_ms / 1000)
        avg_rel_l2    = np.mean(rel_l2s)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.subplots_adjust(wspace=0.35)

        panels = [
            (axes[0], a0,     "hot",     None,   None,   "Conductivity  a(x)",                         ""),
            (axes[1], u_ref0, "viridis", vmin_u, vmax_u, f"FEM ground truth  (FD ≈ {avg_solver_s:.2f}s)", ""),
            (axes[2], u_pred0,"viridis", vmin_u, vmax_u, f"FNO prediction  ({avg_fno_ms:.1f} ms)",       ""),
            (axes[3], err0,   "RdBu_r",  None,   None,   f"|Error|  rel-L2 = {rl2:.4f}",                ""),
        ]
        for ax, data, cmap, vmin, vmax, title, label in panels:
            kw = dict(cmap=cmap, origin="lower", interpolation="bilinear")
            if vmin is not None:
                kw["vmin"], kw["vmax"] = vmin, vmax
            im = ax.imshow(data, **kw)
            ax.set_title(title, fontsize=9, pad=6)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Darcy Flow  ·  FD Solver ({resolution}×{resolution}) vs FNO Surrogate  ·  "
            f"{avg_speedup:.0f}× speedup  ·  avg rel-L2 = {avg_rel_l2:.4f}",
            fontsize=11, fontweight="bold", y=1.02,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        log.info(f"\nFigure saved → {out}")
        plt.close(fig)
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_fno.pt")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n",          type=int, default=20, help="Number of test cases")
    p.add_argument("--n_plot",     type=int, default=1,  help="Samples to include in figure")
    p.add_argument("--output",     default="results/compare.png")
    run_comparison(p.parse_args())
