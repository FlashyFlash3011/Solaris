# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
visualize.py — animated 3-D slice views: solver vs FNO surrogate.

Runs a single simulation through both the 3-D FD solver and the trained FNO,
then produces an animated matplotlib figure with three rows of slice panels:

    Row 1  XY plane  at z = Nz//2  —  Solver T  |  FNO T
    Row 2  XZ plane  at y = Ny//2  —  Solver T  |  FNO T
    Row 3  YZ plane  at x = Nx//2  —  Solver T  |  FNO T

A ``FuncAnimation`` cycles through all snapshot times (title shows ms).
Saved to ``results/chip_thermal_3d.gif``.

Usage
-----
python3.11 projects/chip_thermal/visualize.py --device cpu
python3.11 projects/chip_thermal/visualize.py --device cuda --save results/out.gif
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from solaris.models.fno import FNO
from solaris.utils import get_logger
from solver import random_power_map_3d, solve_heat_3d, T_AMBIENT_3D


def load_model(ckpt_path: str, device: torch.device) -> FNO:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden   = ckpt.get("hidden_channels", 32)
    n_layers = ckpt.get("n_layers", 4)
    modes    = ckpt.get("modes", 8)
    model = FNO(
        in_channels=2, out_channels=1,
        hidden_channels=hidden, n_layers=n_layers,
        modes=modes, dim=3,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_fno(model, Q: np.ndarray, times: np.ndarray, t_end: float,
            T_scale_global: float, device: torch.device) -> np.ndarray:
    """Run the FNO for all snapshot times and return denormalised T."""
    Nx, Ny, Nz = Q.shape
    n_times = len(times)
    peak_Q = float(Q.max()) + 1e-8
    Q_norm = Q / peak_Q

    Q_t = torch.as_tensor(Q_norm[None], dtype=torch.float32)         # (1,Nx,Ny,Nz)
    t_norms = (times / t_end).astype(np.float32)
    t_ch = torch.tensor(
        t_norms[:, None, None, None, None] * np.ones((n_times, 1, Nx, Ny, Nz),
                                                      dtype=np.float32)
    )
    Q_rep = Q_t.unsqueeze(0).expand(n_times, -1, -1, -1, -1)          # (n_t,1,Nx,Ny,Nz)
    inp = torch.cat([Q_rep, t_ch], dim=1).to(device)                   # (n_t,2,Nx,Ny,Nz)

    with torch.no_grad():
        out = model(inp).cpu().numpy()[:, 0]                           # (n_t,Nx,Ny,Nz)

    return out * T_scale_global + T_AMBIENT_3D   # denormalise → °C


def main(args):
    log = get_logger("visualize")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")

    # ── Load model and norm stats ────────────────────────────────────────────
    stats = np.load(Path(args.checkpoint).parent / "norm_stats_3d.npz")
    T_ambient      = float(stats["T_ambient"])
    t_end          = float(stats["t_end"])
    T_scale_global = float(stats.get("T_scale_global", np.array(1.0)))
    n_times   = args.n_times

    model = load_model(args.checkpoint, device)
    log.info(f"Loaded FNO-3D | params={model.num_parameters():,}")

    times = np.linspace(t_end / n_times, t_end, n_times)   # (n_times,)

    # ── Run solver ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    Nx, Ny, Nz = 32, 32, 16
    Q = random_power_map_3d(Nx, Ny, Nz, rng=rng)
    log.info("Running 3-D FD solver …")
    snaps_ref, _, wall = solve_heat_3d(Q, t_end=t_end, n_snapshots=n_times)
    log.info(f"Solver done in {wall:.2f}s")

    # ── Run FNO ──────────────────────────────────────────────────────────────
    log.info("Running FNO-3D …")
    import time; t0 = time.perf_counter()
    snaps_fno = run_fno(model, Q, times, t_end, T_scale_global, device)
    log.info(f"FNO done in {(time.perf_counter()-t0)*1000:.1f}ms")

    # ── Build animation ──────────────────────────────────────────────────────
    mid_x = Nx // 2
    mid_y = Ny // 2
    mid_z = Nz // 2

    # Colour limits: common across solver + FNO, across all times
    T_min = min(snaps_ref.min(), snaps_fno.min())
    T_max = max(snaps_ref.max(), snaps_fno.max())

    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    row_labels = [
        f"XY  z={mid_z}",
        f"XZ  y={mid_y}",
        f"YZ  x={mid_x}",
    ]
    col_labels = ["FD Solver", "FNO-3D"]

    def get_slices(snaps_arr, t_idx):
        T = snaps_arr[t_idx]
        return [
            T[:, :, mid_z],   # XY
            T[:, mid_y, :],   # XZ
            T[mid_x, :, :],   # YZ
        ]

    # Initialise with first frame
    ims = []
    for row in range(3):
        for col, snaps_arr in enumerate([snaps_ref, snaps_fno]):
            ax = axes[row, col]
            sl = get_slices(snaps_arr, 0)[row]
            im = ax.imshow(
                sl.T, cmap="inferno", origin="lower", aspect="auto",
                vmin=T_min, vmax=T_max,
                interpolation="nearest",
            )
            ax.set_title(f"{col_labels[col]}  {row_labels[row]}", fontsize=9)
            ax.set_xlabel("x / y" if row < 2 else "y")
            ax.set_ylabel("y / z" if row == 0 else ("z" if row == 1 else "z"))
            ims.append(im)
            if col == 1:
                plt.colorbar(im, ax=ax, label="°C")

    suptitle = fig.suptitle(
        f"Chip Thermal 3-D  |  t = {times[0]*1e3:.2f} ms",
        fontsize=11, fontweight="bold",
    )

    def update(frame):
        slices_ref = get_slices(snaps_ref, frame)
        slices_fno = get_slices(snaps_fno, frame)
        idx = 0
        for row in range(3):
            for col, slices in enumerate([slices_ref, slices_fno]):
                ims[idx].set_data(slices[row].T)
                idx += 1
        suptitle.set_text(
            f"Chip Thermal 3-D  |  t = {times[frame]*1e3:.2f} ms"
        )
        return ims + [suptitle]

    anim = FuncAnimation(
        fig, update, frames=n_times, interval=400, blit=False,
    )

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving animation → {out}")
    anim.save(str(out), writer=PillowWriter(fps=3), dpi=100)
    log.info("Done.")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Animate 3-D chip-thermal solver vs FNO")
    p.add_argument("--checkpoint", default="checkpoints/best_fno_3d.pt")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--n_times",    type=int, default=8,   help="Snapshots (match training)")
    p.add_argument("--seed",       type=int, default=2024, help="Power-map RNG seed")
    p.add_argument("--save",       default="results/chip_thermal_3d.gif")
    main(p.parse_args())
