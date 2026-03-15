# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare.py — visual comparison: traditional FD solver vs FNO surrogate.

Produces results/comparison.png with:
  • Row 1 – Traditional solver snapshots over time
  • Row 2 – FNO predictions at the same times
  • Row 3 – Absolute error maps
  • Bottom – Bar chart: time taken by each method

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
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from physicsnemo.models.fno import FNO
from physicsnemo.utils import get_logger
from solver import make_initial_field, solve_diffusion, ALPHA_WATER


def load_model_and_stats(ckpt_dir: str, device: torch.device):
    ckpt_dir = Path(ckpt_dir)
    stats = np.load(ckpt_dir / "norm_stats.npz")
    T_ambient = float(stats["T_ambient"])
    t_end     = float(stats["t_end"])
    times     = stats["times"].tolist()

    ckpt = torch.load(ckpt_dir / "best_fno.pt", map_location=device, weights_only=False)
    model = FNO(in_channels=2, out_channels=1, hidden_channels=64,
                n_layers=4, modes=16, dim=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, T_ambient, t_end, times


def fno_predict_sequence(model, T0, times, t_end, T_ambient, device):
    """Run FNO for each requested time and return predictions + total wall time."""
    H, W = T0.shape
    T0_rise = T0 - T_ambient
    peak = float(T0_rise.max()) + 1e-8
    T0_n = (T0_rise / peak).astype(np.float32)
    preds = []

    # GPU warmup (not counted in timing)
    t_norm_dummy = np.full((1, H, W), 0.5, dtype=np.float32)
    x_dummy = torch.as_tensor(
        np.concatenate([T0_n[None, None], t_norm_dummy[None]], axis=1),
        dtype=torch.float32
    ).to(device)
    with torch.no_grad():
        _ = model(x_dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for t in times:
        t_norm = t / t_end
        t_ch = np.full((1, H, W), t_norm, dtype=np.float32)
        x = torch.as_tensor(
            np.concatenate([T0_n[None, None], t_ch[None]], axis=1),
            dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            pred_n = model(x)
        # Inverse transform: rise → absolute temperature
        T_pred = pred_n.cpu().numpy()[0, 0] * peak + T_ambient
        preds.append(T_pred)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fno_time = time.perf_counter() - t0
    return np.stack(preds), fno_time


def make_comparison_figure(T0, solver_snaps, fno_snaps, times,
                           solver_time, fno_time, out_path):
    n_times = len(times)
    T_ambient = float(T0[0, 0])

    # Global colour scale across both methods (fair comparison)
    vmin = T_ambient
    vmax = max(solver_snaps.max(), fno_snaps.max(), T0.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "inferno"

    err_max = np.abs(solver_snaps - fno_snaps).max()
    err_norm = Normalize(vmin=0, vmax=err_max)

    speedup = solver_time / fno_time

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(3.2 * (n_times + 1), 15), facecolor="#0e0e0e")
    fig.patch.set_facecolor("#0e0e0e")

    # 4 image rows + 1 bar-chart row
    outer = gridspec.GridSpec(5, 1, figure=fig,
                              height_ratios=[1, 1, 1, 1, 0.65],
                              hspace=0.08)

    row_labels = ["Initial field", "Traditional solver", "FNO surrogate (ours)", "Absolute error"]
    row_color  = ["#4fc3f7",       "#ef9a9a",            "#a5d6a7",              "#ffe082"]

    col_titles = ["t = 0s"] + [f"t = {t:.1f}s" for t in times]

    axes_grid = []
    for row in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_times + 1, subplot_spec=outer[row], wspace=0.04
        )
        row_axes = []
        for col in range(n_times + 1):
            ax = fig.add_subplot(inner[col])
            ax.set_facecolor("#0e0e0e")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#333333")
            row_axes.append(ax)
        axes_grid.append(row_axes)

    # ── Row 0: initial field ─────────────────────────────────────────────────
    for col in range(n_times + 1):
        ax = axes_grid[0][col]
        if col == 0:
            im = ax.imshow(T0, cmap=cmap, norm=norm, origin="lower", aspect="auto")
            ax.set_title(col_titles[0], color="white", fontsize=10, pad=4)
        else:
            ax.axis("off")
        if col == 0:
            ax.set_ylabel("Initial", color=row_color[0], fontsize=11, labelpad=6)

    # ── Rows 1–3: solver / FNO / error ───────────────────────────────────────
    data_rows = [solver_snaps, fno_snaps, np.abs(solver_snaps - fno_snaps)]
    norms_rows = [norm, norm, err_norm]
    cmaps_rows = [cmap, cmap, "hot"]

    for row, (data, dn, cm) in enumerate(zip(data_rows, norms_rows, cmaps_rows), start=1):
        for col in range(n_times + 1):
            ax = axes_grid[row][col]
            if col == 0:
                ax.axis("off")
            else:
                im = ax.imshow(data[col - 1], cmap=cm, norm=dn,
                               origin="lower", aspect="auto")
                if row == 1:
                    ax.set_title(col_titles[col], color="white", fontsize=10, pad=4)
            if col == 0:
                ax.set_ylabel(row_labels[row], color=row_color[row],
                              fontsize=11, labelpad=6)

    # Shared colourbar (temperature rows)
    cbar_ax = fig.add_axes([0.92, 0.32, 0.012, 0.52])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Temperature (°C)", color="white", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # ── Bottom: timing bar chart ──────────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[4])
    ax_bar.set_facecolor("#1a1a1a")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444444")

    methods = ["Traditional\nFD Solver", f"FNO Surrogate\n(our model, AMD GPU)"]
    timings = [solver_time, fno_time]
    colors  = ["#ef9a9a", "#a5d6a7"]
    bars = ax_bar.barh(methods, timings, color=colors, height=0.45, edgecolor="#222222")

    for bar, t in zip(bars, timings):
        label = f"{t:.3f}s" if t >= 0.01 else f"{t*1000:.1f}ms"
        ax_bar.text(bar.get_width() + solver_time * 0.01, bar.get_y() + bar.get_height() / 2,
                    label, va="center", color="white", fontsize=13, fontweight="bold")

    ax_bar.set_xlim(0, solver_time * 1.25)
    ax_bar.set_xlabel("Wall time (seconds)", color="white", fontsize=11)
    ax_bar.tick_params(colors="white")
    ax_bar.xaxis.label.set_color("white")
    plt.setp(ax_bar.get_xticklabels(), color="white")
    plt.setp(ax_bar.get_yticklabels(), color="white", fontsize=12)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Heat Diffusion in Water  ·  Traditional Solver vs FNO Surrogate\n"
        f"Solver: {solver_time:.2f}s   |   FNO: {fno_time*1000:.1f}ms   |   "
        f"Speedup: {speedup:.0f}×   |   Max error: {err_max:.2f}°C",
        color="white", fontsize=14, fontweight="bold", y=0.995
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return speedup, err_max


def run(args):
    log = get_logger("compare")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    model, T_ambient, t_end, times = load_model_and_stats(args.checkpoint_dir, device)
    log.info(f"Loaded FNO | {model.num_parameters():,} params | simulating {t_end}s of diffusion")
    log.info(f"Time points: {[f'{t:.1f}s' for t in times]}")

    rng = np.random.default_rng(args.seed)
    T0 = make_initial_field(64, 64, rng=rng)

    # ── Traditional solver ────────────────────────────────────────────────────
    log.info("\nRunning traditional FD solver …")
    snaps_all, snap_times, solver_time, n_steps = solve_diffusion(
        T0, t_end=t_end, n_snapshots=len(times) + 1
    )
    solver_snaps = snaps_all[1:]    # drop t=0 frame
    log.info(f"  Solver: {n_steps:,} time steps  →  {solver_time:.3f}s")

    # ── FNO surrogate ─────────────────────────────────────────────────────────
    log.info("Running FNO surrogate …")
    fno_snaps, fno_time = fno_predict_sequence(
        model, T0, times, t_end, T_ambient, device
    )
    log.info(f"  FNO: {len(times)} predictions  →  {fno_time*1000:.2f}ms total")

    # ── Results ───────────────────────────────────────────────────────────────
    speedup = solver_time / fno_time
    max_err = np.abs(solver_snaps - fno_snaps).max()
    log.info(f"\n{'='*50}")
    log.info(f"  Solver:   {solver_time:.3f}s  ({n_steps:,} steps)")
    log.info(f"  FNO:      {fno_time*1000:.2f}ms  ({len(times)} forward passes)")
    log.info(f"  Speedup:  {speedup:.0f}×")
    log.info(f"  Max err:  {max_err:.3f}°C  (out of {T0.max()-T0[0,0]:.1f}°C range)")

    out = Path(args.output)
    speedup, err = make_comparison_figure(
        T0, solver_snaps, fno_snaps, times,
        solver_time, fno_time, out
    )
    log.info(f"\nFigure saved → {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--seed",           type=int, default=2024)
    p.add_argument("--output",         default="results/comparison.png")
    run(p.parse_args())
