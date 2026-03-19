# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — Batch throughput: scipy FD solver vs trained FNO surrogate.

Generates N_batch brand-new chip layouts (never seen during training), times:
  1. scipy sparse FD solver — solved sequentially on CPU
  2. FNO surrogate          — batched GPU (or CPU) inference

Prints a summary table and saves results/compare.png — a 4-panel figure
showing power map with chip architecture labels, FD temperature (°C),
FNO temperature (°C), and absolute error.

Usage
-----
python compare.py --checkpoint checkpoints/best_fno.pt --device cuda
python compare.py --checkpoint checkpoints/best_fno.pt --device cuda --n_batch 100
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
from solver import chip_floorplan_power_map, solve_heat_fd, LAYOUT_LABELS


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden_channels = ckpt.get("hidden_channels", 64)
    n_layers        = ckpt.get("n_layers",         4)
    modes           = ckpt.get("modes",            16)
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
    return model, resolution


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
    stats  = np.load(Path(args.checkpoint).parent / "norm_stats.npz")
    Q_mean = float(stats["Q_mean"]);  Q_std = float(stats["Q_std"])
    T_mean = float(stats["T_mean"]);  T_std = float(stats["T_std"])

    # ── Load trained FNO ──
    model, resolution = load_model(args.checkpoint, device)
    H = W = resolution
    log.info(f"Loaded FNO  |  resolution={resolution}  |  params={model.num_parameters():,}")

    # ── GPU warm-up ──
    if device.type == "cuda":
        dummy = torch.zeros(1, 1, H, W, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    # ── Generate N_batch new chip layouts ──
    N = args.n_batch
    log.info(f"Generating {N} new chip layouts …")
    rng = np.random.default_rng(args.seed)
    Q_all = np.stack([chip_floorplan_power_map(H, W, rng) for _ in range(N)])  # (N, H, W)

    # ── Baseline: scipy FD solver — sequential ──
    log.info(f"Running scipy FD solver on {N} layouts (sequential) …")
    t0 = time.perf_counter()
    T_fd_all = np.empty_like(Q_all)
    for i in range(N):
        T_fd_all[i], _ = solve_heat_fd(Q_all[i])
    solver_total = time.perf_counter() - t0
    solver_per_ms = solver_total / N * 1000

    # ── FNO surrogate — batched GPU inference ──
    log.info(f"Running FNO surrogate on {N} layouts (batched, batch_size={args.batch_size}) …")
    Q_norm = (Q_all - Q_mean) / Q_std                                 # (N, H, W)
    Q_t    = torch.as_tensor(Q_norm[:, None], dtype=torch.float32)    # (N, 1, H, W)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    T_fno_norm_chunks = []
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            chunk = Q_t[start:start + args.batch_size].to(device)
            T_fno_norm_chunks.append(model(chunk).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    fno_total = time.perf_counter() - t0
    fno_per_ms = fno_total / N * 1000

    T_fno_norm = torch.cat(T_fno_norm_chunks, dim=0).numpy()[:, 0]   # (N, H, W)
    T_fno_all  = T_fno_norm * T_std + T_mean                          # denormalise → °C

    # ── Error statistics ──
    diff   = T_fno_all - T_fd_all
    rel_l2 = (
        np.linalg.norm(diff.reshape(N, -1), axis=1)
        / (np.linalg.norm(T_fd_all.reshape(N, -1), axis=1) + 1e-8)
    )
    speedup = solver_total / fno_total

    # ── Summary table ──
    log.info("")
    log.info("=" * 60)
    log.info(f"  Batch size        : {N} chip layouts")
    log.info(f"  FD Solver  total  : {solver_total:.1f}s  ({solver_per_ms:.1f} ms/layout)")
    log.info(f"  FNO        total  : {fno_total*1000:.0f}ms  ({fno_per_ms:.2f} ms/layout)")
    log.info(f"  Speedup           : {speedup:.0f}×")
    log.info(f"  Rel-L2 avg        : {np.mean(rel_l2):.4f}  (max {np.max(rel_l2):.4f})")
    log.info("=" * 60)

    # ── Plot ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe

        # Use the first sample for the figure
        idx  = 0
        Q0   = Q_all[idx]       # (H, W)
        T_fd = T_fd_all[idx]
        T_fn = T_fno_all[idx]
        err  = np.abs(T_fn - T_fd)
        rl2  = float(rel_l2[idx])

        T_ambient = float(T_fd.min())
        vmin_T    = T_ambient
        vmax_T    = float(T_fd.max())

        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        fig.patch.set_facecolor("#0d0d0d")
        fig.subplots_adjust(wspace=0.38, left=0.04, right=0.97, top=0.82, bottom=0.06)

        # Panel 0 — Power map with chip architecture labels
        im0 = axes[0].imshow(Q0, cmap="hot", origin="lower", interpolation="bilinear")
        axes[0].set_title("Power Map  Q(x,y)\n[W/m²]", fontsize=9, color="white", pad=6)
        cb0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cb0.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb0.outline.set_edgecolor("white")

        # Architecture labels
        for fx, fy, label in LAYOUT_LABELS:
            px = fx * (W - 1)
            py = fy * (H - 1)
            txt = axes[0].text(
                px, py, label,
                color="white", fontsize=6.5, ha="center", va="center",
                fontweight="bold",
            )
            txt.set_path_effects([
                pe.Stroke(linewidth=2, foreground="black"),
                pe.Normal(),
            ])

        # Panel 1 — FD solver temperature
        im1 = axes[1].imshow(T_fd, cmap="inferno", origin="lower",
                             vmin=vmin_T, vmax=vmax_T, interpolation="bilinear")
        axes[1].set_title(
            f"FD Solver  T [°C]\n{solver_per_ms:.1f} ms/layout",
            fontsize=9, color="white", pad=6,
        )
        cb1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cb1.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb1.outline.set_edgecolor("white")

        # Panel 2 — FNO prediction temperature
        im2 = axes[2].imshow(T_fn, cmap="inferno", origin="lower",
                             vmin=vmin_T, vmax=vmax_T, interpolation="bilinear")
        axes[2].set_title(
            f"FNO Surrogate  T [°C]\n{fno_per_ms:.2f} ms/layout",
            fontsize=9, color="white", pad=6,
        )
        cb2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cb2.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb2.outline.set_edgecolor("white")

        # Panel 3 — Absolute error
        im3 = axes[3].imshow(err, cmap="RdBu_r", origin="lower", interpolation="bilinear")
        axes[3].set_title(
            f"|Error|  [°C]\nMax: {err.max():.1f}°C · Rel-L2: {rl2*100:.2f}%",
            fontsize=9, color="white", pad=6,
        )
        cb3 = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        cb3.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb3.outline.set_edgecolor("white")

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
            ax.set_facecolor("#0d0d0d")

        fig.suptitle(
            f"Batch of {N} new chip layouts  ·  "
            f"FD Solver: {solver_total:.1f}s  ·  "
            f"FNO (GPU): {fno_total*1000:.0f}ms  ·  "
            f"{speedup:.0f}× faster",
            fontsize=12, fontweight="bold", color="white", y=0.97,
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
        log.info(f"Figure saved → {out}")
        plt.close(fig)

    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/best_fno.pt")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--n_batch",     type=int, default=1000, help="Number of new layouts to compare")
    p.add_argument("--batch_size",  type=int, default=64,   help="GPU batch size for FNO inference")
    p.add_argument("--seed",        type=int, default=9999, help="RNG seed (different from training data)")
    p.add_argument("--output",      default="results/compare.png")
    run_comparison(p.parse_args())
