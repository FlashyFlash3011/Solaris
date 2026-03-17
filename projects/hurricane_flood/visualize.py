# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Animated 2×2 hurricane flood visualization.

The animation shows four panels updated each frame:
  ┌─────────────────────┬──────────────────────┐
  │  Flood depth [m]    │  Hurricane wind [m/s]│
  │  (Blues + coastline)│  (speed + quiver)    │
  ├─────────────────────┼──────────────────────┤
  │  Uncertainty ±q̂    │  Cumulative extent   │
  │  (conformal bands)  │  (ever-flooded mask) │
  └─────────────────────┴──────────────────────┘

Data source
-----------
Either pass a pre-computed .npz from compare.py (--data), or use --generate
to run a fresh solver + surrogate simulation inline (no checkpoint needed
for --solver-only mode).

Usage
-----
# Standalone: solver only, generates its own data, saves GIF
python visualize.py --generate --solver-only --output results/hurricane_flood.gif

# Full: surrogate + uncertainty (requires trained checkpoint)
python visualize.py --generate --output results/hurricane_flood.gif

# From compare.py output
python visualize.py --data results/compare_data.npz --output results/hurricane_flood.gif

# Interactive window
python visualize.py --generate --solver-only --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Data generation / loading
# ---------------------------------------------------------------------------

def generate_simulation_data(
    args,
    seed: int = 0,
) -> dict:
    """Run a solver + (optionally) surrogate simulation and return visualisation data."""
    import torch
    from solver import (
        random_coastal_bathymetry,
        random_hurricane_track,
        run_hurricane_simulation,
    )

    H, W = args.H, args.W
    rng = np.random.default_rng(seed)
    bathy = random_coastal_bathymetry(H, W, rng=rng)
    track = random_hurricane_track(H, W, n_hours=24, rng=rng)
    result = run_hurricane_simulation(bathy, track, n_hours=24, n_snapshots=args.n_snapshots)

    flood_solver = result["flood"]     # (T, H, W)
    wind_u       = result["wind_u"]
    wind_v       = result["wind_v"]
    times_h      = result["times_h"]

    out = {
        "flood_solver": flood_solver,
        "flood_pred":   flood_solver.copy(),   # default: show solver as "prediction"
        "flood_lower":  None,
        "flood_upper":  None,
        "wind_u":       wind_u,
        "wind_v":       wind_v,
        "bathy":        bathy,
        "times_h":      np.array(times_h),
        "q_hat":        0.0,
    }

    if args.solver_only:
        return out

    # Try to load trained surrogate and roll out
    ckpt_path = Path(args.checkpoint_dir) / "best_coupled.pt"
    if not ckpt_path.exists():
        print(f"[visualize] No checkpoint at {ckpt_path}; showing solver only.")
        return out

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    from compare import load_conformal, load_model, surrogate_rollout
    from solaris.models.conformal import ConformalNeuralOperator

    model, stats = load_model(args.checkpoint_dir, device)
    predictor = load_conformal(args.checkpoint_dir, model, device)

    flood_pred, flood_lower, flood_upper = surrogate_rollout(
        model, predictor, bathy, result, stats, device, H, W
    )

    q_hat = predictor._q_hat.item() if predictor is not None else 0.0
    out.update({
        "flood_pred":  flood_pred,
        "flood_lower": flood_lower,
        "flood_upper": flood_upper,
        "q_hat":       q_hat,
    })
    return out


def load_simulation_data(data_path: str) -> dict:
    """Load pre-computed visualisation data from an .npz file."""
    d = np.load(data_path, allow_pickle=True)
    out = {}
    for k in ["flood_solver", "flood_pred", "wind_u", "wind_v", "bathy", "times_h", "q_hat"]:
        out[k] = d[k] if k in d else None
    out["flood_lower"] = d["flood_lower"] if "flood_lower" in d else None
    out["flood_upper"] = d["flood_upper"] if "flood_upper" in d else None
    if out["q_hat"] is not None:
        out["q_hat"] = float(out["q_hat"])
    return out


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------

def build_figure(H: int, W: int, extent_km: float):
    """Create the 2×2 animated figure with fixed layout."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(13, 9), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25,
                            left=0.06, right=0.96, top=0.88, bottom=0.06)

    axes = {
        "flood":     fig.add_subplot(gs[0, 0]),
        "wind":      fig.add_subplot(gs[0, 1]),
        "unc":       fig.add_subplot(gs[1, 0]),
        "extent":    fig.add_subplot(gs[1, 1]),
    }

    panel_style = dict(facecolor="#1a1a2e")
    tick_style  = dict(colors="0.7", labelsize=7)
    extent_img  = [0, extent_km, 0, extent_km]

    for name, ax in axes.items():
        ax.set_facecolor(panel_style["facecolor"])
        ax.tick_params(axis="both", **tick_style)
        for spine in ax.spines.values():
            spine.set_edgecolor("0.35")
        ax.set_xlabel("km", color="0.7", fontsize=7)
        ax.set_ylabel("km", color="0.7", fontsize=7)

    # Fixed colourbar labels
    for name, title in [
        ("flood", "Flood depth [m]"),
        ("wind",  "Wind speed [m/s]"),
        ("unc",   "Uncertainty ±q̂ [m]"),
        ("extent","Cumulative flood extent"),
    ]:
        axes[name].set_title(title, color="0.9", fontsize=9, pad=4)

    return fig, axes, extent_img


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def run_visualization(
    data: dict,
    output: str,
    fps: int = 4,
    dpi: int = 120,
    interactive: bool = False,
) -> None:
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    H, W  = data["bathy"].shape
    n_frames = len(data["times_h"])
    extent_km = W * 1.0   # 1 km/cell
    q_hat = data["q_hat"] or 0.0

    fig, axes, extent_img = build_figure(H, W, extent_km)

    flood_data   = data["flood_pred"]    # (T, H, W) — surrogate or solver
    solver_data  = data["flood_solver"]  # (T, H, W)
    wind_u       = data["wind_u"]
    wind_v       = data["wind_v"]
    bathy        = data["bathy"]
    times_h      = data["times_h"]
    flood_lower  = data["flood_lower"]
    flood_upper  = data["flood_upper"]

    vmax_flood = float(np.nanmax(flood_data)) if np.nanmax(flood_data) > 0 else 1.0
    vmax_wind  = float(np.nanmax(np.sqrt(wind_u ** 2 + wind_v ** 2)))
    vmax_wind  = max(vmax_wind, 10.0)

    # Cumulative extent mask (updated each frame)
    cumulative_mask = np.zeros((H, W), dtype=bool)

    # ── Initial artists ──────────────────────────────────────────────
    ax_f = axes["flood"]
    ax_w = axes["wind"]
    ax_u = axes["unc"]
    ax_e = axes["extent"]

    # Flood panel
    flood_im = ax_f.imshow(
        flood_data[0], origin="lower", cmap="Blues",
        vmin=0, vmax=vmax_flood, aspect="auto",
        extent=[0, extent_km, 0, extent_km],
    )
    coast_contour = ax_f.contour(
        np.linspace(0, extent_km, W), np.linspace(0, extent_km, H),
        bathy, levels=[0], colors=["#c8a97e"], linewidths=[1.2],
    )
    flood_title = ax_f.set_title("Flood depth [m]  t=0.0h", color="0.9", fontsize=9, pad=4)
    cb_flood = fig.colorbar(flood_im, ax=ax_f, fraction=0.046, pad=0.04)
    cb_flood.ax.tick_params(labelsize=7, colors="0.7")
    cb_flood.outline.set_edgecolor("0.35")

    # Wind panel
    wspd0 = np.sqrt(wind_u[0] ** 2 + wind_v[0] ** 2)
    wind_im = ax_w.imshow(
        wspd0, origin="lower", cmap="YlOrRd",
        vmin=0, vmax=vmax_wind, aspect="auto",
        extent=[0, extent_km, 0, extent_km],
    )
    qs = 6  # quiver step
    # Build coordinate arrays that match the strided slices exactly
    qi = np.arange(0, H, qs)   # row indices of sampled points
    qj = np.arange(0, W, qs)   # col indices
    yw = qi * (extent_km / H)
    xw = qj * (extent_km / W)
    quiv = ax_w.quiver(
        xw, yw,
        wind_u[0][::qs, ::qs],
        wind_v[0][::qs, ::qs],
        scale=vmax_wind * 12, width=0.003,
        color="white", alpha=0.7,
    )
    cb_wind = fig.colorbar(wind_im, ax=ax_w, fraction=0.046, pad=0.04)
    cb_wind.ax.tick_params(labelsize=7, colors="0.7")
    cb_wind.outline.set_edgecolor("0.35")
    wind_title = ax_w.set_title("Hurricane wind [m/s]  t=0.0h", color="0.9", fontsize=9, pad=4)

    # Uncertainty panel
    has_unc = flood_upper is not None
    unc_data0 = np.zeros((H, W)) if not has_unc else np.maximum(flood_upper[0] - flood_data[0], 0.0)
    unc_vmax = float(np.nanmax(unc_data0)) if has_unc and np.nanmax(unc_data0) > 0 else 1.0
    unc_im = ax_u.imshow(
        unc_data0, origin="lower", cmap="Purples",
        vmin=0, vmax=unc_vmax, aspect="auto",
        extent=[0, extent_km, 0, extent_km],
    )
    cb_unc = fig.colorbar(unc_im, ax=ax_u, fraction=0.046, pad=0.04)
    cb_unc.ax.tick_params(labelsize=7, colors="0.7")
    cb_unc.outline.set_edgecolor("0.35")
    if has_unc:
        unc_subtitle = f"90% coverage bound  q̂={q_hat:.2f}"
    else:
        unc_subtitle = "Train model for uncertainty bounds"
    ax_u.set_title(f"Uncertainty ±q̂ [m]\n{unc_subtitle}", color="0.9", fontsize=8, pad=4)

    # Extent panel
    cumulative_mask |= (flood_data[0] > 0.05)
    cmap_ext = mcolors.LinearSegmentedColormap.from_list(
        "extent", ["#1a1a2e", "#e63946"], N=2
    )
    extent_im = ax_e.imshow(
        cumulative_mask.astype(float), origin="lower", cmap=cmap_ext,
        vmin=0, vmax=1, aspect="auto",
        extent=[0, extent_km, 0, extent_km],
    )
    ax_e.contour(
        np.linspace(0, extent_km, W), np.linspace(0, extent_km, H),
        bathy, levels=[0], colors=["#c8a97e"], linewidths=[0.8],
    )
    extent_title = ax_e.set_title(
        f"Cumulative flood extent\n0.0% of land flooded", color="0.9", fontsize=8, pad=4,
    )

    land_cells = float((bathy >= 0).sum())

    # Overall figure title
    fig.suptitle(
        "Hurricane Storm Surge — Neural Surrogate Forecast",
        color="white", fontsize=12, y=0.96,
    )

    # ── Update function ──────────────────────────────────────────────
    def update(frame: int):
        nonlocal cumulative_mask
        t_h = float(times_h[frame])

        # Flood
        flood_im.set_data(flood_data[frame])
        flood_title.set_text(f"Flood depth [m]  t={t_h:.1f}h")

        # Wind
        wspd = np.sqrt(wind_u[frame] ** 2 + wind_v[frame] ** 2)
        wind_im.set_data(wspd)
        quiv.set_UVC(wind_u[frame][::qs, ::qs], wind_v[frame][::qs, ::qs])
        wind_title.set_text(f"Hurricane wind [m/s]  t={t_h:.1f}h")

        # Uncertainty
        if has_unc and frame < len(flood_upper):
            unc = np.maximum(flood_upper[frame] - flood_data[frame], 0.0)
            # Mask to flooded area only
            unc[flood_data[frame] < 0.05] = 0.0
            unc_im.set_data(unc)

        # Cumulative extent
        cumulative_mask |= (flood_data[frame] > 0.05)
        extent_im.set_data(cumulative_mask.astype(float))
        pct = 100.0 * cumulative_mask.sum() / max(land_cells, 1)
        extent_title.set_text(
            f"Cumulative flood extent\n{pct:.1f}% of land flooded"
        )

        return [flood_im, wind_im, quiv, unc_im, extent_im,
                flood_title, wind_title, extent_title]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if interactive:
        plt.show()
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif ext == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        else:
            raise ValueError(f"Unsupported output format: {ext!r} (use .gif or .mp4)")
        anim.save(str(out_path), writer=writer, dpi=dpi)
        print(f"[visualize] Saved → {out_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Animated hurricane flood visualization")
    p.add_argument("--data",           default=None,
                   help="Path to .npz from compare.py. If omitted, uses --generate.")
    p.add_argument("--generate",       action="store_true",
                   help="Generate a fresh simulation (no pre-existing data needed).")
    p.add_argument("--solver-only",    dest="solver_only", action="store_true",
                   help="Show solver output only (no surrogate / uncertainty).")
    p.add_argument("--output",         default="results/hurricane_flood.gif")
    p.add_argument("--fps",            type=int,   default=4)
    p.add_argument("--dpi",            type=int,   default=120)
    p.add_argument("--interactive",    action="store_true")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed",           type=int,   default=7)
    p.add_argument("--H",              type=int,   default=64)
    p.add_argument("--W",              type=int,   default=64)
    p.add_argument("--n_snapshots",    type=int,   default=25)
    args = p.parse_args()

    if args.data is not None:
        data = load_simulation_data(args.data)
    elif args.generate:
        data = generate_simulation_data(args, seed=args.seed)
    else:
        p.error("Provide either --data <path.npz> or --generate")

    run_visualization(data, args.output, fps=args.fps, dpi=args.dpi, interactive=args.interactive)
