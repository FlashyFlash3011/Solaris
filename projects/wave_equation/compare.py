# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Compare FD wave solver vs trained FNO surrogate.

Evaluates the task the model was trained on: predict u one snapshot interval
ahead from (u, v, t).  Top row: solver; bottom row: FNO.

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
from solver import random_gaussian_ic, solve_wave, solve_wave_snapshots  # noqa: E402
from train import build_timestep_channel  # noqa: E402

from solaris.models.fno import FNO
from solaris.utils import get_logger


def compare(args):
    log = get_logger("wave_compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / "best_wave_fno.pt"
    stats_path = ckpt_dir / "norm_stats.npz"

    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}. Run train.py first.")
        return

    stats = np.load(stats_path)
    in_mean, in_std = float(stats["in_mean"]), float(stats["in_std"])
    tgt_mean, tgt_std = float(stats["tgt_mean"]), float(stats["tgt_std"])

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=ckpt.get("hidden", 64),
        n_layers=ckpt.get("n_layers", 4),
        modes=ckpt.get("modes", 16),
        dim=2,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    log.info(f"Loaded FNO from {ckpt_path}")

    # Reproduce the exact training task: one snapshot interval ahead.
    H, W = ckpt.get("H", 64), ckpt.get("W", 64)
    c = ckpt.get("c", 1.0)
    dt = ckpt.get("dt", 5e-4)
    n_steps = ckpt.get("n_steps", 400)
    n_snapshots = ckpt.get("n_snapshots", 10)
    steps_per_snap = max(1, n_steps // (n_snapshots - 1))
    dt_snap = steps_per_snap * dt

    def predict(u: np.ndarray, v: np.ndarray, t_phys: float) -> np.ndarray:
        t_ch = build_timestep_channel(t_phys, H, W).numpy()
        inp = np.concatenate([u[None], v[None], t_ch], axis=0)[None]  # (1,3,H,W)
        inp_norm = (inp - in_mean) / in_std
        with torch.no_grad():
            out = model(torch.as_tensor(inp_norm, dtype=torch.float32).to(device))
        return out.squeeze().cpu().numpy() * tgt_std + tgt_mean

    rng = np.random.default_rng(1234)
    solver_times, fno_times, l2_errors = [], [], []

    for _ in range(args.n_test):
        u0, v0 = random_gaussian_ic(H, W, rng=rng)

        t0 = time.perf_counter()
        u_gt, _ = solve_wave(u0, v0, c=c, dt=dt, n_steps=steps_per_snap)
        solver_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred = predict(u0, v0, 0.0)
        fno_times.append(time.perf_counter() - t0)

        l2 = np.linalg.norm(pred - u_gt) / (np.linalg.norm(u_gt) + 1e-8)
        l2_errors.append(l2)

    solver_ms = np.mean(solver_times) * 1000
    fno_ms = np.mean(fno_times) * 1000
    speedup = solver_ms / (fno_ms + 1e-9)
    mean_l2 = np.mean(l2_errors)

    log.info(f"Solver:  {solver_ms:.1f} ms/sample")
    log.info(f"FNO:     {fno_ms:.2f} ms/sample  ({speedup:.1f}× speedup)")
    log.info(f"Mean rel-L2 error: {mean_l2:.4f}")

    # Visualisation — wave propagation snapshots
    try:
        import matplotlib.pyplot as plt

        rng2 = np.random.default_rng(9999)
        u0_vis, v0_vis = random_gaussian_ic(H, W, rng=rng2)
        n_show = 4
        u_snaps, v_snaps = solve_wave_snapshots(
            u0_vis,
            v0_vis,
            c=c,
            dt=dt,
            n_steps=steps_per_snap * n_show,
            n_snapshots=n_show + 1,
        )

        fig, axes = plt.subplots(2, n_show, figsize=(14, 6))
        vmax = np.abs(u_snaps).max()
        kw = dict(cmap="seismic", vmin=-vmax, vmax=vmax, origin="lower")

        # Top: solver at t_{k+1}; bottom: FNO one-step prediction from t_k
        for col in range(n_show):
            axes[0, col].imshow(u_snaps[col + 1], **kw)
            axes[0, col].set_title(f"Solver t={(col + 1) * dt_snap:.3f}")
            axes[0, col].axis("off")
            pr = predict(u_snaps[col], v_snaps[col], col * dt_snap)
            axes[1, col].imshow(pr, **kw)
            axes[1, col].set_title(f"FNO t={(col + 1) * dt_snap:.3f}")
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("Solver", rotation=90, labelpad=4)
        axes[1, 0].set_ylabel("FNO", rotation=90, labelpad=4)
        plt.suptitle(f"Wave Equation: FD vs FNO  (mean rel-L2={mean_l2:.3f})")
        plt.tight_layout()
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out), dpi=120)
        log.info(f"Saved {out}")
    except ImportError:
        log.warning("matplotlib not installed — skipping visualisation")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--n_test", type=int, default=20)
    p.add_argument("--output", default="comparison.png")
    compare(p.parse_args())
