# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Animated hurricane flood visualization.

Four panels, each with a terrain background so ocean/land is always visible:

  ┌──────────────────────────┬──────────────────────────┐
  │  FLOODING MAP            │  WIND SPEED              │
  │  terrain + surge overlay │  speed map + storm eye   │
  │  + storm track           │  + wind arrows           │
  ├──────────────────────────┼──────────────────────────┤
  │  STORM INTENSITY         │  TOTAL DAMAGE SO FAR     │
  │  (wind speed / AI conf.) │  cumulative flooded area │
  └──────────────────────────┴──────────────────────────┘

Usage
-----
# Standalone — no trained model needed
python visualize.py --generate --solver-only --output results/hurricane_flood.gif

# With surrogate + AI confidence map
python visualize.py --generate --output results/hurricane_flood.gif

# Interactive window
python visualize.py --generate --solver-only --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make solver / solaris importable from any working directory
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))


# ---------------------------------------------------------------------------
# Terrain RGBA helper
# ---------------------------------------------------------------------------

def make_terrain_rgba(bathy: np.ndarray) -> np.ndarray:
    """Build a static RGBA image of the coastal terrain.

    Ocean cells shade from deep blue (deep) to light teal (shallow).
    Land cells shade from sandy tan (coast) to olive green (higher ground).
    """
    H, W = bathy.shape
    rgba = np.zeros((H, W, 4), dtype=np.float32)

    ocean = bathy < 0
    land  = ~ocean

    # Ocean: deep navy → light teal proportional to depth
    depth_norm = np.clip(-bathy / 35.0, 0.0, 1.0)
    rgba[ocean, 0] = 0.04 + 0.12 * (1 - depth_norm[ocean])
    rgba[ocean, 1] = 0.22 + 0.32 * (1 - depth_norm[ocean])
    rgba[ocean, 2] = 0.48 + 0.38 * (1 - depth_norm[ocean])
    rgba[ocean, 3] = 1.0

    # Land: sandy tan (sea level) → olive green (hills)
    elev_norm = np.clip(bathy / 8.0, 0.0, 1.0)
    rgba[land, 0] = 0.84 - 0.22 * elev_norm[land]
    rgba[land, 1] = 0.73 - 0.08 * elev_norm[land]
    rgba[land, 2] = 0.44 - 0.22 * elev_norm[land]
    rgba[land, 3] = 1.0

    return rgba


# ---------------------------------------------------------------------------
# Storm position along the track
# ---------------------------------------------------------------------------

def get_storm_positions(track, times_h):
    """Return list of (x_km, y_km) for each snapshot time."""
    positions = []
    for t in times_h:
        t_c = float(np.clip(t, 0, len(track) - 1))
        i   = int(t_c)
        f   = t_c - i
        if i + 1 < len(track):
            x = track[i][0] + f * (track[i + 1][0] - track[i][0])
            y = track[i][1] + f * (track[i + 1][1] - track[i][1])
        else:
            x, y = track[-1]
        positions.append((float(x), float(y)))
    return positions


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_simulation_data(args, seed: int = 0) -> dict:
    """Run the SWE solver (and optionally the neural surrogate) and return all data."""
    from solver import (
        random_coastal_bathymetry,
        random_hurricane_track,
        run_hurricane_simulation,
    )

    H, W = args.H, args.W
    rng  = np.random.default_rng(seed)

    bathy  = random_coastal_bathymetry(H, W, rng=rng)
    track  = random_hurricane_track(H, W, n_hours=24, rng=rng)
    result = run_hurricane_simulation(bathy, track, n_hours=24,
                                      n_snapshots=args.n_snapshots)

    times_h   = np.array(result["times_h"])
    storm_pos = get_storm_positions(track, times_h)

    out = {
        "flood_solver": result["flood"],
        "flood_pred":   result["flood"].copy(),   # shown by default (solver = ground truth)
        "flood_lower":  None,
        "flood_upper":  None,
        "wind_u":       result["wind_u"],
        "wind_v":       result["wind_v"],
        "bathy":        bathy,
        "times_h":      times_h,
        "track":        track,
        "storm_pos":    storm_pos,
        "q_hat":        0.0,
    }

    if args.solver_only:
        return out

    # Load surrogate if checkpoint exists
    ckpt_path = Path(args.checkpoint_dir) / "best_coupled.pt"
    if not ckpt_path.exists():
        print(f"[visualize] No checkpoint at {ckpt_path} — showing solver only.")
        return out

    import torch
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    from compare import load_conformal, load_model, surrogate_rollout

    model, stats = load_model(args.checkpoint_dir, device)
    predictor    = load_conformal(args.checkpoint_dir, model, device)
    flood_pred, flood_lower, flood_upper = surrogate_rollout(
        model, predictor, bathy, result, stats, device, H, W
    )
    q_hat = predictor._q_hat.item() if predictor is not None else 0.0
    out.update({"flood_pred": flood_pred, "flood_lower": flood_lower,
                "flood_upper": flood_upper, "q_hat": q_hat})
    return out


def load_simulation_data(data_path: str) -> dict:
    d = np.load(data_path, allow_pickle=True)
    out = {k: d[k] for k in d.files if k not in ("flood_lower", "flood_upper",
                                                    "track", "storm_pos")}
    out["flood_lower"] = d["flood_lower"] if "flood_lower" in d else None
    out["flood_upper"] = d["flood_upper"] if "flood_upper" in d else None
    out["track"]       = d["track"].tolist()     if "track"     in d else None
    out["storm_pos"]   = d["storm_pos"].tolist() if "storm_pos" in d else None
    out["q_hat"]       = float(d["q_hat"]) if "q_hat" in d else 0.0
    return out


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def run_visualization(data: dict, output: str,
                      fps: int = 4, dpi: int = 130,
                      interactive: bool = False) -> None:
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap

    bathy      = data["bathy"]
    H, W       = bathy.shape
    times_h    = data["times_h"]
    n_frames   = len(times_h)
    flood_data = data["flood_pred"]
    wind_u     = data["wind_u"]
    wind_v     = data["wind_v"]
    upper      = data["flood_upper"]
    storm_pos  = data.get("storm_pos") or [(W / 2, -10.0)] * n_frames
    q_hat      = float(data.get("q_hat") or 0.0)
    has_unc    = upper is not None
    domain_km  = float(W)            # 1 cell = 1 km

    terrain_rgba = make_terrain_rgba(bathy)
    land_mask    = bathy >= 0
    land_cells   = float(land_mask.sum())
    cumul_mask   = np.zeros((H, W), dtype=bool)

    vmax_flood = max(float(np.nanmax(flood_data)), 0.5)
    vmax_wind  = max(float(np.nanmax(np.sqrt(wind_u**2 + wind_v**2))), 10.0)

    xs_grid = np.linspace(0, domain_km, W)
    ys_grid = np.linspace(0, domain_km, H)
    ext     = [0, domain_km, 0, domain_km]

    # ── Custom colourmaps ─────────────────────────────────────────────
    # Flood: transparent at 0 → bright cyan → deep blue at max surge
    flood_cmap = LinearSegmentedColormap.from_list("surge", [
        (0.00, (0.00, 0.55, 1.00, 0.00)),
        (0.08, (0.00, 0.70, 1.00, 0.50)),
        (0.35, (0.00, 0.40, 0.90, 0.78)),
        (1.00, (0.05, 0.05, 0.65, 0.95)),
    ])
    # Wind: pale cream → yellow → orange → crimson (matches hurricane categories)
    wind_cmap = LinearSegmentedColormap.from_list(
        "wind", ["#f7f4e8", "#ffe44d", "#ff8800", "#cc1100", "#6b0000"]
    )
    # Damage: transparent → vivid red (land cells that have flooded)
    damage_cmap = LinearSegmentedColormap.from_list(
        "damage", [(0, (0.85, 0.85, 0.85, 0.0)), (1, (0.92, 0.08, 0.08, 0.92))]
    )
    # Confidence: green (AI is sure) → yellow → red (AI is unsure)
    conf_cmap = LinearSegmentedColormap.from_list(
        "conf", ["#1a8c1a", "#aadd00", "#ffcc00", "#ff6600", "#cc0000"]
    )

    # ── Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10), facecolor="#16162a")
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.24,
                           left=0.05, right=0.95, top=0.89, bottom=0.06)
    ax_f = fig.add_subplot(gs[0, 0])   # flooding
    ax_w = fig.add_subplot(gs[0, 1])   # wind
    ax_c = fig.add_subplot(gs[1, 0])   # confidence / intensity
    ax_d = fig.add_subplot(gs[1, 1])   # damage area

    def _style(ax):
        ax.set_facecolor("#16162a")
        ax.tick_params(colors="0.6", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.35")
        ax.set_xlabel("← west   km   east →", color="0.55", fontsize=6.5)
        ax.set_ylabel("← south   km   north →", color="0.55", fontsize=6.5)

    for ax in (ax_f, ax_w, ax_c, ax_d):
        _style(ax)

    def _add_terrain(ax, alpha=1.0):
        ax.imshow(terrain_rgba, origin="lower", extent=ext, aspect="auto",
                  alpha=alpha, zorder=1)

    def _add_coast(ax, lw=1.3, color="white", alpha=0.85):
        ax.contour(xs_grid, ys_grid, bathy, levels=[0],
                   colors=[color], linewidths=[lw], linestyles=["-"],
                   zorder=5, alpha=alpha)

    def _add_colorbar(fig, ax, mappable, label):
        cb = fig.colorbar(mappable, ax=ax, fraction=0.043, pad=0.03)
        cb.set_label(label, color="0.75", fontsize=8)
        cb.ax.tick_params(labelsize=7, colors="0.65")
        cb.outline.set_edgecolor("0.35")
        return cb

    # ── Panel 1 : Flooding map ────────────────────────────────────────
    _add_terrain(ax_f)
    flood_norm = mcolors.Normalize(vmin=0, vmax=vmax_flood)
    flood_im   = ax_f.imshow(flood_data[0], origin="lower", extent=ext,
                              aspect="auto", cmap=flood_cmap,
                              norm=flood_norm, zorder=3)
    _add_coast(ax_f)
    _add_colorbar(fig, ax_f,
                  plt.cm.ScalarMappable(norm=flood_norm, cmap=flood_cmap),
                  "Flood depth (m)")
    flood_title = ax_f.set_title("FLOODING MAP  —  t = 0.0 h",
                                  color="white", fontsize=10, pad=5, fontweight="bold")
    ax_f.text(0.02, 0.02, "White line = coastline   ⊕ = storm eye",
              transform=ax_f.transAxes, color="0.65", fontsize=6.5, zorder=12)

    # ── Panel 2 : Wind speed ──────────────────────────────────────────
    _add_terrain(ax_w, alpha=0.30)
    wind_norm = mcolors.Normalize(vmin=0, vmax=vmax_wind)
    wspd0     = np.sqrt(wind_u[0]**2 + wind_v[0]**2)
    wind_im   = ax_w.imshow(wspd0, origin="lower", extent=ext, aspect="auto",
                             cmap=wind_cmap, norm=wind_norm, alpha=0.90, zorder=3)
    qs = 6
    qi = np.arange(0, H, qs)
    qj = np.arange(0, W, qs)
    yw = qi * (domain_km / H)
    xw = qj * (domain_km / W)
    quiv = ax_w.quiver(xw, yw,
                        wind_u[0][::qs, ::qs], wind_v[0][::qs, ::qs],
                        scale=vmax_wind * 14, width=0.003,
                        color="#ffffffbb", zorder=5)
    _add_coast(ax_w, color="#aaaaaa", alpha=0.6)
    cb_w = _add_colorbar(fig, ax_w,
                          plt.cm.ScalarMappable(norm=wind_norm, cmap=wind_cmap),
                          "Wind speed (m/s)")
    # Mark hurricane-force threshold (33 m/s ≈ Category 1)
    if vmax_wind > 33:
        frac = 33.0 / vmax_wind
        cb_w.ax.axhline(frac, color="white", lw=1.2, ls="--")
        cb_w.ax.text(1.12, frac, "Cat-1\nforce", color="white",
                     fontsize=5.5, va="center", transform=cb_w.ax.transAxes)
    wind_title = ax_w.set_title("HURRICANE WINDS  —  t = 0.0 h",
                                 color="white", fontsize=10, pad=5, fontweight="bold")

    # ── Panel 3 : AI confidence or storm intensity ────────────────────
    _add_terrain(ax_c, alpha=0.45)
    if has_unc:
        unc0     = np.maximum(upper[0] - flood_data[0], 0.0)
        unc_vmax = max(float(np.nanmax(unc0)), 0.1)
        unc_norm = mcolors.Normalize(vmin=0, vmax=unc_vmax)
        conf_im  = ax_c.imshow(unc0, origin="lower", extent=ext, aspect="auto",
                                cmap=conf_cmap, norm=unc_norm, alpha=0.88, zorder=3)
        _add_colorbar(fig, ax_c,
                      plt.cm.ScalarMappable(norm=unc_norm, cmap=conf_cmap),
                      "Uncertainty ± (m)")
        conf_lbl = f"AI CONFIDENCE  —  90% coverage  —  q={q_hat:.1f} m"
        ax_c.text(0.02, 0.02, "Green = AI is confident   Red = AI is unsure",
                  transform=ax_c.transAxes, color="0.65", fontsize=6.5, zorder=12)
    else:
        conf_im  = ax_c.imshow(wspd0, origin="lower", extent=ext, aspect="auto",
                                cmap="plasma", norm=wind_norm, alpha=0.88, zorder=3)
        _add_colorbar(fig, ax_c,
                      plt.cm.ScalarMappable(norm=wind_norm, cmap="plasma"),
                      "Wind speed (m/s)")
        conf_lbl = "STORM INTENSITY"
        ax_c.text(0.02, 0.02, "Brighter = stronger, more dangerous winds",
                  transform=ax_c.transAxes, color="0.65", fontsize=6.5, zorder=12)
    _add_coast(ax_c, color="#aaaaaa", alpha=0.6)
    conf_title = ax_c.set_title(conf_lbl, color="white", fontsize=10,
                                 pad=5, fontweight="bold")

    # ── Panel 4 : Total damage ────────────────────────────────────────
    _add_terrain(ax_d)
    damage_im = ax_d.imshow(cumul_mask.astype(float), origin="lower", extent=ext,
                             aspect="auto", cmap=damage_cmap, vmin=0, vmax=1, zorder=3)
    _add_coast(ax_d)
    damage_title = ax_d.set_title("TOTAL FLOODED AREA  —  0.0% of coast",
                                   color="white", fontsize=10, pad=5, fontweight="bold")
    ax_d.text(0.02, 0.02, "Red = flooded at any point during the storm",
              transform=ax_d.transAxes, color="0.65", fontsize=6.5, zorder=12)

    # ── Main title & progress bar ─────────────────────────────────────
    main_title = fig.suptitle(
        "Hurricane Storm Surge  ·  Hour 0 / 24",
        color="white", fontsize=13, fontweight="bold", y=0.95,
    )
    ax_bar = fig.add_axes([0.05, 0.015, 0.90, 0.016])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_facecolor("#2a2a42")
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    ax_bar.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_bar.text(0.5, 0.5, "simulation progress", ha="center", va="center",
                color="0.45", fontsize=6, transform=ax_bar.transAxes)
    bar_fill = mpatches.Rectangle((0, 0), 0.0, 1,
                                   facecolor="#4fc3f7", edgecolor="none")
    ax_bar.add_patch(bar_fill)

    # ── Mutable eye-marker containers (removed and redrawn each frame) ──
    eyes = {ax: [None] for ax in (ax_f, ax_w, ax_c, ax_d)}
    # Track line containers
    track_lines = {ax: [None] for ax in (ax_f, ax_d)}

    def _place_eye(ax, x_km, y_km, container):
        if container[0] is not None:
            try:
                container[0].remove()
            except Exception:
                pass
        c = mpatches.Circle((x_km, y_km), 3.2, fill=False,
                             edgecolor="white", lw=2.0, zorder=10)
        ax.add_patch(c)
        ax.plot(x_km, y_km, "+", color="white", ms=7, lw=1.6, zorder=11)
        container[0] = c

    def _draw_track(ax, container, frame):
        if container[0] is not None:
            try:
                container[0].remove()
            except Exception:
                pass
            container[0] = None
        if storm_pos and frame > 0:
            xs = [p[0] for p in storm_pos[:frame + 1]]
            ys = [p[1] for p in storm_pos[:frame + 1]]
            ln, = ax.plot(xs, ys, "--", color="white", lw=1.1, alpha=0.55, zorder=8)
            container[0] = ln

    # ── Animation update ──────────────────────────────────────────────
    def update(frame: int):
        nonlocal cumul_mask
        t_h        = float(times_h[frame])
        sx, sy     = storm_pos[frame]
        prog       = frame / max(n_frames - 1, 1)
        wspd_frame = np.sqrt(wind_u[frame]**2 + wind_v[frame]**2)
        max_wspd   = float(wspd_frame.max())

        # Panel 1 — flood depth overlay on terrain
        flood_im.set_data(flood_data[frame])
        flood_title.set_text(
            f"FLOODING MAP  —  t = {t_h:.1f} h"
            + (f"  |  surge peak {float(flood_data[frame].max()):.1f} m"
               if flood_data[frame].max() > 0.05 else "")
        )
        _place_eye(ax_f, sx, sy, eyes[ax_f])
        _draw_track(ax_f, track_lines[ax_f], frame)

        # Panel 2 — wind speed + quiver
        wind_im.set_data(wspd_frame)
        quiv.set_UVC(wind_u[frame][::qs, ::qs], wind_v[frame][::qs, ::qs])
        wind_title.set_text(f"HURRICANE WINDS  —  t = {t_h:.1f} h  —  max {max_wspd:.0f} m/s")
        _place_eye(ax_w, sx, sy, eyes[ax_w])

        # Panel 3 — AI confidence or storm intensity
        if has_unc and frame < len(upper):
            unc = np.maximum(upper[frame] - flood_data[frame], 0.0)
            unc[flood_data[frame] < 0.02] = 0.0
            conf_im.set_data(unc)
        else:
            conf_im.set_data(wspd_frame)
        _place_eye(ax_c, sx, sy, eyes[ax_c])

        # Panel 4 — cumulative flood extent
        cumul_mask = cumul_mask | (flood_data[frame] > 0.05)
        damage_im.set_data(cumul_mask.astype(float))
        pct = 100.0 * float(cumul_mask.sum()) / max(land_cells, 1)
        km2 = float(cumul_mask.sum())    # 1 cell = 1 km²
        damage_title.set_text(
            f"TOTAL FLOODED AREA  —  {pct:.1f}%  ({km2:.0f} km2)"
        )
        _place_eye(ax_d, sx, sy, eyes[ax_d])
        _draw_track(ax_d, track_lines[ax_d], frame)

        # Progress bar + main title
        bar_fill.set_width(prog)
        main_title.set_text(f"Hurricane Storm Surge  ·  Hour {t_h:.1f} / {times_h[-1]:.0f}")

        return [flood_im, wind_im, conf_im, damage_im,
                flood_title, wind_title, conf_title, damage_title, main_title]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    if interactive:
        plt.show()
    else:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sfx = out_path.suffix.lower()
        if sfx == ".gif":
            anim.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
        elif sfx == ".mp4":
            anim.save(str(out_path),
                      writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=dpi)
        else:
            raise ValueError(f"Use .gif or .mp4, got {sfx!r}")
        print(f"[visualize] Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default=None)
    p.add_argument("--generate",       action="store_true")
    p.add_argument("--solver-only",    dest="solver_only", action="store_true")
    p.add_argument("--output",         default="results/hurricane_flood.gif")
    p.add_argument("--fps",            type=int,   default=4)
    p.add_argument("--dpi",            type=int,   default=130)
    p.add_argument("--interactive",    action="store_true")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--seed",           type=int,   default=7)
    p.add_argument("--H",              type=int,   default=64)
    p.add_argument("--W",              type=int,   default=64)
    p.add_argument("--n_snapshots",    type=int,   default=25)
    args = p.parse_args()

    if args.data:
        data = load_simulation_data(args.data)
    elif args.generate:
        data = generate_simulation_data(args, seed=args.seed)
    else:
        p.error("Provide --data <path.npz> or --generate")

    run_visualization(data, args.output, fps=args.fps, dpi=args.dpi,
                      interactive=args.interactive)
