# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — 3-D transient: FD solver vs trained FNO surrogate.

For N unseen chip designs this script:
  1. Runs the 3-D finite-difference solver      (ground truth, slow)
  2. Runs the 3-D FNO surrogate on GPU/CPU      (fast)
  3. Prints a timing + error table
  4. Saves a slice-panel figure:
       rows  = XY mid-plane | XZ mid-plane | YZ mid-plane
       cols  = Q map | Solver T | FNO T | |Error|
     (one figure page per sample, up to --n_plot samples)

Usage
-----
python3.11 projects/chip_thermal/compare.py --device cuda --n 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from solaris.models.fno import FNO
from solaris.utils import get_logger
from solver import random_power_map_3d, solve_heat_3d, T_AMBIENT_3D


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    """Load the trained FNO-3D and its architecture hyper-parameters."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = ckpt.get("hidden_channels", 32)
    n_layers = ckpt.get("n_layers", 4)
    modes = ckpt.get("modes", 8)
    model = FNO(
        in_channels=2, out_channels=1,
        hidden_channels=hidden, n_layers=n_layers,
        modes=modes, dim=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def run_comparison(args):
    log = get_logger("compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load norm stats ──────────────────────────────────────────────────────
    stats_path = Path(args.checkpoint).parent / "norm_stats_3d.npz"
    stats = np.load(stats_path)
    T_ambient      = float(stats["T_ambient"])
    t_end          = float(stats["t_end"])
    T_scale_global = float(stats.get("T_scale_global", np.array(1.0)))
    n_times   = args.n_times

    # ── Load model ───────────────────────────────────────────────────────────
    model, _ = load_model(args.checkpoint, device)
    log.info(f"Loaded FNO-3D | params={model.num_parameters():,}")

    rng = np.random.default_rng(args.seed)
    Nx, Ny, Nz = 32, 32, 16
    times = np.linspace(t_end / n_times, t_end, n_times)   # snapshot times

    solver_times, fno_times, rel_l2s = [], [], []
    samples_to_plot = []

    log.info(f"\nRunning {args.n} comparisons …\n")
    log.info(f"{'#':>4}  {'Solver (s)':>12}  {'FNO (ms)':>10}  {'Speedup':>9}  {'Rel-L2':>8}")
    log.info("-" * 57)

    for i in range(args.n):
        Q = random_power_map_3d(Nx, Ny, Nz, rng=rng)
        peak_Q = float(Q.max()) + 1e-8

        # ── Baseline: 3-D FD solver ──────────────────────────────────────────
        t0 = time.perf_counter()
        snaps_ref, _, _ = solve_heat_3d(Q, t_end=t_end, n_snapshots=n_times)
        t_solver = time.perf_counter() - t0
        solver_times.append(t_solver)

        # ── FNO surrogate (all time steps in one batch) ───────────────────────
        Q_norm = Q / peak_Q
        Q_t = torch.as_tensor(Q_norm[None], dtype=torch.float32)   # (1,Nx,Ny,Nz)

        t_norms = times / t_end                                    # (n_times,)
        # Build batched input: (n_times, 2, Nx, Ny, Nz)
        t_channels = torch.tensor(
            t_norms[:, None, None, None, None] * np.ones((n_times, 1, Nx, Ny, Nz)),
            dtype=torch.float32,
        )
        Q_rep = Q_t.unsqueeze(0).expand(n_times, -1, -1, -1, -1)  # (n_times,1,Nx,Ny,Nz)
        inp = torch.cat([Q_rep, t_channels], dim=1).to(device)     # (n_times,2,Nx,Ny,Nz)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(inp)   # (n_times, 1, Nx, Ny, Nz)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fno = time.perf_counter() - t0
        fno_times.append(t_fno)

        # Denormalise: T = output × T_scale_global + T_ambient
        snaps_fno = out.cpu().numpy()[:, 0] * T_scale_global + T_ambient  # (n_times,Nx,Ny,Nz)

        # ── Error (mean over time steps, relative L2 in interior) ────────────
        diff = snaps_fno[:, 1:-1, 1:-1, 1:-1] - snaps_ref[:, 1:-1, 1:-1, 1:-1]
        rel_l2 = float(
            np.linalg.norm(diff) / (np.linalg.norm(snaps_ref[:, 1:-1, 1:-1, 1:-1]) + 1e-8)
        )
        rel_l2s.append(rel_l2)

        speedup = t_solver / (t_fno + 1e-12)
        log.info(
            f"{i+1:>4}  {t_solver:>12.3f}  {t_fno*1000:>10.2f}  "
            f"{speedup:>8.0f}×  {rel_l2:>8.4f}"
        )

        if i < args.n_plot:
            # Store final snapshot for plotting
            samples_to_plot.append((Q, snaps_ref[-1], snaps_fno[-1], rel_l2,
                                    t_solver, t_fno))

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 57)
    log.info(f"  Solver  avg: {np.mean(solver_times):.3f}s   "
             f"(min {np.min(solver_times):.3f}s)")
    log.info(f"  FNO     avg: {np.mean(fno_times)*1000:.2f}ms  "
             f"(min {np.min(fno_times)*1000:.2f}ms)")
    log.info(f"  Speedup avg: {np.mean(solver_times)/np.mean(fno_times):.0f}×")
    log.info(f"  Rel-L2  avg: {np.mean(rel_l2s):.4f}  "
             f"(max {np.max(rel_l2s):.4f})")

    # ── Plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        for s_idx, (Q, T_ref, T_fno, rl2, t_s, t_f) in enumerate(samples_to_plot):
            mid_x = Nx // 2
            mid_y = Ny // 2
            mid_z = Nz // 2

            # 3 planes × 4 panels
            planes = [
                ("XY  z=mid", Q[:, :, mid_z], T_ref[:, :, mid_z], T_fno[:, :, mid_z]),
                ("XZ  y=mid", Q[:, mid_y, :], T_ref[:, mid_y, :], T_fno[:, mid_y, :]),
                ("YZ  x=mid", Q[mid_x, :, :], T_ref[mid_x, :, :], T_fno[mid_x, :, :]),
            ]
            fig = plt.figure(figsize=(16, 4 * 3))
            gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)

            for row, (plane_name, Q_sl, T_ref_sl, T_fno_sl) in enumerate(planes):
                err_sl = np.abs(T_fno_sl - T_ref_sl)
                vmin_T = min(T_ref_sl.min(), T_fno_sl.min())
                vmax_T = max(T_ref_sl.max(), T_fno_sl.max())

                ax0 = fig.add_subplot(gs[row, 0])
                im0 = ax0.imshow(Q_sl.T, cmap="hot", origin="lower", aspect="auto")
                ax0.set_title(f"{plane_name} | Q [W/m³]")
                plt.colorbar(im0, ax=ax0)

                ax1 = fig.add_subplot(gs[row, 1])
                im1 = ax1.imshow(T_ref_sl.T, cmap="inferno", origin="lower",
                                 vmin=vmin_T, vmax=vmax_T, aspect="auto")
                ax1.set_title(f"Solver ({t_s:.2f}s)")
                plt.colorbar(im1, ax=ax1, label="°C")

                ax2 = fig.add_subplot(gs[row, 2])
                im2 = ax2.imshow(T_fno_sl.T, cmap="inferno", origin="lower",
                                 vmin=vmin_T, vmax=vmax_T, aspect="auto")
                ax2.set_title(f"FNO-3D ({t_f*1000:.1f}ms)")
                plt.colorbar(im2, ax=ax2, label="°C")

                ax3 = fig.add_subplot(gs[row, 3])
                im3 = ax3.imshow(err_sl.T, cmap="RdBu_r", origin="lower", aspect="auto")
                ax3.set_title(f"|Error| — rel-L2={rl2:.4f}" if row == 0 else "|Error|")
                plt.colorbar(im3, ax=ax3, label="°C")

            avg_speedup = np.mean(solver_times) / (np.mean(fno_times) + 1e-12)
            fig.suptitle(
                f"Sample {s_idx+1} — Chip Thermal 3-D  |  "
                f"Avg speedup {avg_speedup:.0f}×  |  "
                f"Avg rel-L2 {np.mean(rel_l2s):.4f}",
                fontsize=12, fontweight="bold",
            )
            suffix = f"_{s_idx+1}" if len(samples_to_plot) > 1 else ""
            out = Path(args.output).with_suffix("").as_posix() + suffix + ".png"
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=130, bbox_inches="tight")
            log.info(f"Figure saved → {out}")
            plt.close(fig)

    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="3-D chip-thermal: solver vs FNO benchmark")
    p.add_argument("--checkpoint", default="checkpoints/best_fno_3d.pt")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--n",          type=int, default=20,  help="Test cases to compare")
    p.add_argument("--n_plot",     type=int, default=3,   help="Samples to include in figure")
    p.add_argument("--n_times",    type=int, default=8,   help="Snapshot count (must match training)")
    p.add_argument("--seed",       type=int, default=999, help="RNG seed (differs from training)")
    p.add_argument("--output",     default="results/compare_3d.png")
    run_comparison(p.parse_args())
