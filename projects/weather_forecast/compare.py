# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — Toy NWP solver vs AFNO surrogate, side-by-side.

Output: results/comparison.png
  • Each column = one forecast lead time (Day 1 → Day 5)
  • Row 1 = Traditional solver (z500 geopotential height)
  • Row 2 = AFNO prediction
  • Row 3 = Absolute error
  • Bottom = Timing bar chart

Usage
-----
python compare.py --device cuda
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

from solaris.models.afno import AFNO
from solaris.utils import get_logger
from data_gen import make_initial_state, simulate


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_dir, device):
    ckpt_dir = Path(ckpt_dir)
    stats = np.load(ckpt_dir / "norm_stats.npz", allow_pickle=True)
    meta = {k: stats[k].item() for k in stats.files}
    nlat, nlon = int(meta["nlat"]), int(meta["nlon"])
    n_days = int(meta["n_days"])

    ckpt = torch.load(ckpt_dir / "best_afno.pt", map_location=device, weights_only=False)
    model = AFNO(in_channels=3, out_channels=2,
                 img_size=(nlat, nlon), patch_size=4,
                 hidden_size=256, n_layers=6, num_blocks=8).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, meta, nlat, nlon, n_days


def norm(arr, mean, std): return (arr - mean) / std
def denorm(arr, mean, std): return arr * std + mean


# ── AFNO inference ────────────────────────────────────────────────────────────

def afno_forecast(model, z0, t0, lead_days, meta, device):
    nlat, nlon = z0.shape
    n_days = meta["n_days"]

    # Warmup
    dummy = torch.zeros(1, 3, nlat, nlon, device=device)
    with torch.no_grad(): _ = model(dummy)
    if device.type == "cuda": torch.cuda.synchronize()

    z_preds, t_preds = [], []
    t0_wall = time.perf_counter()
    for day in lead_days:
        z0_n = norm(z0, meta["z500_mean"], meta["z500_std"]).astype(np.float32)
        t0_n = norm(t0, meta["t850_mean"], meta["t850_std"]).astype(np.float32)
        lt   = np.full((1, nlat, nlon), day / n_days, dtype=np.float32)
        x = torch.as_tensor(
            np.stack([z0_n, t0_n, lt[0]])[None], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]   # (2, H, W)
        z_preds.append(denorm(pred[0], meta["z500_mean"], meta["z500_std"]))
        t_preds.append(denorm(pred[1], meta["t850_mean"], meta["t850_std"]))
    if device.type == "cuda": torch.cuda.synchronize()
    afno_time = time.perf_counter() - t0_wall
    return np.stack(z_preds), np.stack(t_preds), afno_time


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(z0, t0, z_sol, t_sol, z_afno, t_afno,
                lead_days, solver_time, afno_time, out_path, lat, lon):
    n_leads = len(lead_days)
    speedup = solver_time / afno_time

    # Use z500 for the main visual (most recognisable as a weather map)
    vmin = min(z_sol.min(), z_afno.min())
    vmax = max(z_sol.max(), z_afno.max())
    norm_z = Normalize(vmin=vmin, vmax=vmax)
    err_max = np.abs(z_sol - z_afno).max()
    norm_e  = Normalize(vmin=0, vmax=err_max)

    fig = plt.figure(figsize=(3.5 * n_leads, 14), facecolor="#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[0.6, 1, 1, 1, 0.55],
                              hspace=0.12)

    # ── Row 0: initial state ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(outer[0])
    ax0.set_facecolor("#0d0d0d")
    im0 = ax0.imshow(z0, cmap="RdBu_r", norm=norm_z, origin="upper",
                     extent=[lon[0], lon[-1], lat[0], lat[-1]], aspect="auto")
    ax0.set_title("Initial state — Day 0  (z500 geopotential height)", color="white",
                  fontsize=11, pad=5)
    ax0.set_xlabel("Longitude", color="#aaaaaa", fontsize=8)
    ax0.set_ylabel("Latitude",  color="#aaaaaa", fontsize=8)
    ax0.tick_params(colors="#666666", labelsize=7)
    _add_grid(ax0)
    cb0 = fig.colorbar(im0, ax=ax0, orientation="vertical", pad=0.01, fraction=0.015)
    cb0.set_label("Height (m)", color="white", fontsize=8)
    cb0.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb0.ax.yaxis.get_ticklabels(), color="white")

    # ── Rows 1–3: forecast rows ───────────────────────────────────────────────
    row_data  = [z_sol, z_afno, np.abs(z_sol - z_afno)]
    row_norms = [norm_z, norm_z, norm_e]
    row_cmaps = ["RdBu_r", "RdBu_r", "hot"]
    row_labels = [
        "Traditional solver\n(steps through every hour)",
        "AFNO surrogate\n(our model, AMD GPU)",
        "Absolute error  |solver − AFNO|",
    ]
    row_colors = ["#ef9a9a", "#a5d6a7", "#ffe082"]

    for row in range(3):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_leads, subplot_spec=outer[row + 1], wspace=0.06)
        for col in range(n_leads):
            ax = fig.add_subplot(inner[col])
            ax.set_facecolor("#0d0d0d")
            im = ax.imshow(row_data[row][col], cmap=row_cmaps[row],
                           norm=row_norms[row], origin="upper",
                           extent=[lon[0], lon[-1], lat[0], lat[-1]],
                           aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#333333")
            if row == 0:
                ax.set_title(f"Day {lead_days[col]}", color="white",
                             fontsize=10, pad=4)
            if col == 0:
                ax.set_ylabel(row_labels[row], color=row_colors[row],
                              fontsize=9, labelpad=6)

        # Shared colorbar per row
        cbar_ax = fig.add_axes([0.92, 0.72 - row * 0.22, 0.012, 0.18])
        sm = plt.cm.ScalarMappable(cmap=row_cmaps[row], norm=row_norms[row])
        cb = fig.colorbar(sm, cax=cbar_ax)
        label = "|Error| (m)" if row == 2 else "Height (m)"
        cb.set_label(label, color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # ── Row 4: timing bar ─────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[4])
    ax_bar.set_facecolor("#1a1a1a")
    for sp in ax_bar.spines.values(): sp.set_edgecolor("#444444")

    labels  = ["Traditional solver\n(stepped through every hour)",
               f"AFNO Surrogate\n(our model, AMD GPU)"]
    times   = [solver_time, afno_time]
    colors  = ["#ef9a9a", "#a5d6a7"]
    bars = ax_bar.barh(labels, times, color=colors, height=0.4, edgecolor="#222222")

    for bar, t in zip(bars, times):
        lbl = f"{t:.2f}s" if t >= 0.1 else f"{t*1000:.1f}ms"
        ax_bar.text(bar.get_width() + solver_time * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    lbl, va="center", color="white", fontsize=13, fontweight="bold")

    ax_bar.set_xlim(0, solver_time * 1.3)
    ax_bar.set_xlabel("Wall time (seconds)", color="white", fontsize=10)
    ax_bar.tick_params(colors="white")
    plt.setp(ax_bar.get_xticklabels(), color="white")
    plt.setp(ax_bar.get_yticklabels(), color="white", fontsize=10)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"5-Day Weather Forecast  ·  Toy NWP Solver vs AFNO Surrogate\n"
        f"Solver: {solver_time:.2f}s   |   AFNO: {afno_time*1000:.0f}ms   |   "
        f"Speedup: {speedup:.0f}×   |   Max z500 error: {err_max:.1f} m",
        color="white", fontsize=13, fontweight="bold", y=0.995,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _add_grid(ax):
    ax.set_xticks(range(0, 361, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.grid(color="#333333", linewidth=0.4)
    ax.tick_params(colors="#666666", labelsize=7)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    log = get_logger("compare")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model, meta, nlat, nlon, n_days = load_model(args.checkpoint_dir, device)
    lead_days = list(range(1, n_days + 1))
    log.info(f"AFNO | {model.num_parameters():,} params | {n_days}-day forecast on {nlat}×{nlon} grid")

    from data_gen import make_grid
    lat, lon, _, _ = make_grid(nlat, nlon)

    rng = np.random.default_rng(args.seed)
    z0, t0 = make_initial_state(nlat, nlon, rng=rng)

    # ── Traditional solver ────────────────────────────────────────────────────
    log.info("\nRunning traditional NWP solver …")
    z_sol, t_sol, snap_days, solver_time, n_steps = simulate(
        z0, t0, n_days=n_days,
        n_snapshots=n_days + 1,
        dt_hours=args.dt_hours,
    )
    z_sol = z_sol[1:]   # drop day-0 snapshot
    t_sol = t_sol[1:]
    log.info(f"  Solver: {n_steps} steps ({n_days} days) → {solver_time:.3f}s")

    # ── AFNO ──────────────────────────────────────────────────────────────────
    log.info("Running AFNO surrogate …")
    z_afno, t_afno, afno_time = afno_forecast(model, z0, t0, lead_days, meta, device)
    log.info(f"  AFNO:   {n_days} predictions → {afno_time*1000:.1f}ms")

    speedup  = solver_time / afno_time
    max_err  = np.abs(z_sol - z_afno).max()
    log.info(f"\n{'='*52}")
    log.info(f"  Solver:  {solver_time:.2f}s  ({n_steps} × {args.dt_hours}h steps)")
    log.info(f"  AFNO:    {afno_time*1000:.1f}ms  ({n_days} forward passes)")
    log.info(f"  Speedup: {speedup:.0f}×")
    log.info(f"  Max z500 error: {max_err:.1f} m  (typical z500 range ~1000 m)")

    make_figure(z0, t0, z_sol, t_sol, z_afno, t_afno,
                lead_days, solver_time, afno_time,
                args.output, lat, lon)
    log.info(f"\nFigure saved → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--seed",           type=int,   default=777)
    p.add_argument("--dt_hours",       type=float, default=1.0)
    p.add_argument("--output",         default="results/comparison.png")
    run(p.parse_args())
