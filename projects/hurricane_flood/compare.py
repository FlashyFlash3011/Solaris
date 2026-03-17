# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark the neural surrogate against the SWE solver on held-out hurricanes.

For each test scenario:
  1. Run the SWE solver to get ground-truth flood snapshots
  2. Autoregressively roll out the neural surrogate
  3. Compare flood depth accuracy (relative L2) and wall-clock speed

Generates results/compare.png — a 5-column figure showing:
  bathymetry | solver flood | surrogate flood | absolute error | uncertainty ±q̂

Usage
-----
python compare.py                          # 10 scenarios, default checkpoint
python compare.py --n 5 --n_plot 2        # fewer scenarios
python compare.py --no_conformal          # skip uncertainty panel
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solver import (
    random_coastal_bathymetry,
    random_hurricane_track,
    run_hurricane_simulation,
)
from train import FloodExtractor, apply_normalisation, build_model, compute_norm_stats

from solaris.metrics import relative_l2_error
from solaris.models.conformal import ConformalNeuralOperator
from solaris.utils import get_logger


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str, device: torch.device):
    """Load CoupledOperator from checkpoint.

    Returns (model, norm_stats_dict).
    """
    ckpt_dir  = Path(checkpoint_dir)
    ckpt_path = ckpt_dir / "best_coupled.pt"
    stats_path = ckpt_dir / "norm_stats.npz"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. Run train.py first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild model from saved hyper-params
    extra = ckpt.get("extra", {})

    class _Args:
        hidden_flood = extra.get("hidden_flood", 64)
        hidden_wind  = extra.get("hidden_wind",  48)
        n_layers     = extra.get("n_layers",      4)
        modes        = extra.get("modes",         16)

    model = build_model(_Args())
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    stats = dict(np.load(stats_path))
    return model, stats


def load_conformal(checkpoint_dir: str, model, device: torch.device) -> Optional[ConformalNeuralOperator]:
    """Load ConformalNeuralOperator wrapping the flood extractor."""
    path = Path(checkpoint_dir) / "conformal_predictor.pt"
    if not path.exists():
        return None
    predictor = ConformalNeuralOperator(FloodExtractor(model))
    predictor.load_state_dict(torch.load(path, map_location=device))
    predictor.to(device)
    predictor.eval()
    # _calibrated is not a buffer so won't survive state_dict; restore from _q_hat
    if not predictor._q_hat.isinf():
        predictor._calibrated = True
    return predictor


# ---------------------------------------------------------------------------
# Surrogate rollout
# ---------------------------------------------------------------------------

def surrogate_rollout(
    model,
    predictor: Optional[ConformalNeuralOperator],
    bathy: np.ndarray,
    result_solver: dict,
    stats: dict,
    device: torch.device,
    H: int,
    W: int,
):
    """Autoregressively roll out the surrogate over all snapshots.

    Feeds predicted (h, wind) back as inputs at each step.

    Returns
    -------
    flood_pred   : np.ndarray (n_snapshots, H, W)
    flood_lower  : np.ndarray or None (n_snapshots, H, W)
    flood_upper  : np.ndarray or None
    """
    n_snap = len(result_solver["times_h"])
    bathy_t = torch.as_tensor(bathy[None, None], dtype=torch.float32)  # (1,1,H,W)

    # Denorm helpers
    fi_mean = torch.as_tensor(stats["fi_mean"], dtype=torch.float32).to(device)
    fi_std  = torch.as_tensor(stats["fi_std"],  dtype=torch.float32).to(device)
    ft_mean = torch.as_tensor(stats["ft_mean"], dtype=torch.float32).to(device)
    ft_std  = torch.as_tensor(stats["ft_std"],  dtype=torch.float32).to(device)
    wi_mean = torch.as_tensor(stats["wi_mean"], dtype=torch.float32).to(device)
    wi_std  = torch.as_tensor(stats["wi_std"],  dtype=torch.float32).to(device)
    wt_mean = torch.as_tensor(stats["wt_mean"], dtype=torch.float32).to(device)
    wt_std  = torch.as_tensor(stats["wt_std"],  dtype=torch.float32).to(device)

    def norm_flood_in(h_arr, bathy_arr, wu_arr, wv_arr, t_norm_val):
        """Build and normalise a (1,5,H,W) flood input tensor."""
        h_log = np.log1p(np.maximum(h_arr, 0.0))
        fi = np.stack([h_log[None], bathy_arr[None], wu_arr[None], wv_arr[None],
                       np.full((1, H, W), t_norm_val)], axis=1).astype(np.float32)
        fi_t = torch.as_tensor(fi).to(device)
        return (fi_t - fi_mean) / fi_std

    def norm_wind_in(wu_arr, wv_arr, t_norm_val):
        wi = np.stack([wu_arr[None], wv_arr[None],
                       np.full((1, H, W), t_norm_val)], axis=1).astype(np.float32)
        wi_t = torch.as_tensor(wi).to(device)
        return (wi_t - wi_mean) / wi_std

    def denorm_flood(ft_norm):
        """Denormalise and invert log1p. Returns numpy (H,W)."""
        ft_phys = ft_norm * ft_std + ft_mean
        return np.expm1(np.maximum(ft_phys.cpu().numpy()[0, 0], 0.0))

    def denorm_wind(wt_norm):
        wt_phys = wt_norm * wt_std + wt_mean
        return wt_phys.cpu().numpy()[0, 0], wt_phys.cpu().numpy()[0, 1]

    # Initialise with solver's t=0 state
    h_cur  = result_solver["flood"][0]
    wu_cur = result_solver["wind_u"][0]
    wv_cur = result_solver["wind_v"][0]

    flood_pred  = [h_cur.copy()]
    flood_lower = [None]
    flood_upper = [None]

    with torch.no_grad():
        for t_idx in range(n_snap - 1):
            t_norm = t_idx / max(n_snap - 2, 1)
            fi = norm_flood_in(h_cur, bathy, wu_cur, wv_cur, t_norm)
            wi = norm_wind_in(wu_cur, wv_cur, t_norm)

            outputs = model({"flood": fi, "wind": wi})
            h_pred_norm = outputs["flood"]
            w_pred_norm = outputs["wind"]

            h_next = denorm_flood(h_pred_norm)
            wu_next, wv_next = denorm_wind(w_pred_norm)

            # Conformal bounds on flood depth
            if predictor is not None:
                combined = torch.cat([fi, wi], dim=1)  # (1, 8, H, W)
                lower_n, upper_n, _ = predictor.predict(combined)
                lower = denorm_flood(lower_n)
                upper = denorm_flood(upper_n)
            else:
                lower = upper = None

            flood_pred.append(np.maximum(h_next, 0.0))
            flood_lower.append(lower)
            flood_upper.append(upper)

            h_cur  = h_next
            wu_cur = wu_next
            wv_cur = wv_next

    return (
        np.stack(flood_pred),
        np.stack(flood_lower[1:]) if flood_lower[1] is not None else None,
        np.stack(flood_upper[1:]) if flood_upper[1] is not None else None,
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = get_logger("compare")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    model, stats = load_model(args.checkpoint_dir, device)
    predictor = None if args.no_conformal else load_conformal(args.checkpoint_dir, model, device)

    H, W = 64, 64
    rng = np.random.default_rng(args.seed)

    results_to_plot, speedups, l2s = [], [], []

    for i in range(args.n):
        bathy = random_coastal_bathymetry(H, W, rng=rng)
        track = random_hurricane_track(H, W, n_hours=24, rng=rng)

        # ── Solver ──
        t0 = time.perf_counter()
        result_solver = run_hurricane_simulation(bathy, track, n_hours=24, n_snapshots=25)
        t_solver = time.perf_counter() - t0

        # ── Surrogate ──
        t0 = time.perf_counter()
        flood_pred, flood_lower, flood_upper = surrogate_rollout(
            model, predictor, bathy, result_solver, stats, device, H, W
        )
        t_surr = time.perf_counter() - t0

        speedup = t_solver / max(t_surr, 1e-6)
        speedups.append(speedup)

        # Rel-L2 at peak flood snapshot (skip t=0)
        solver_flood = result_solver["flood"]
        peak_t = int(np.argmax(solver_flood[1:].max(axis=(1, 2)))) + 1
        pred_t  = torch.as_tensor(flood_pred[peak_t][None, None])
        truth_t = torch.as_tensor(solver_flood[peak_t][None, None])
        l2 = relative_l2_error(pred_t, truth_t).item()
        l2s.append(l2)

        log.info(
            f"  Scenario {i+1:2d}:  solver={t_solver:.2f}s  "
            f"surrogate={t_surr:.3f}s  "
            f"speedup={speedup:.1f}×  "
            f"rel-L2={l2:.4f}"
        )

        if i < args.n_plot:
            results_to_plot.append({
                "bathy": bathy,
                "solver": solver_flood,
                "pred": flood_pred,
                "lower": flood_lower,
                "upper": flood_upper,
                "peak_t": peak_t,
                "times_h": result_solver["times_h"],
            })

    log.info(
        f"\nSummary ({args.n} scenarios):\n"
        f"  Speedup:  {np.mean(speedups):.1f}× (mean)  {np.max(speedups):.1f}× (max)\n"
        f"  Rel-L2:   {np.mean(l2s):.4f} (mean)  {np.max(l2s):.4f} (max)"
    )

    # ── Figure ────────────────────────────────────────────────────────
    n_plot = len(results_to_plot)
    n_cols = 4 if args.no_conformal else 5
    fig, axes = plt.subplots(n_plot, n_cols, figsize=(n_cols * 4, n_plot * 3.5))
    if n_plot == 1:
        axes = axes[None, :]

    col_titles = ["Bathymetry", "Solver flood", "Surrogate flood", "Abs error"]
    if not args.no_conformal:
        col_titles.append("Uncertainty ±q̂")

    for row, res in enumerate(results_to_plot):
        bathy = res["bathy"]
        peak_t = res["peak_t"]
        t_h = res["times_h"][peak_t]

        vmax_flood = max(res["solver"][peak_t].max(), res["pred"][peak_t].max(), 0.01)
        err = np.abs(res["pred"][peak_t] - res["solver"][peak_t])

        # Col 0: bathymetry
        ax = axes[row, 0]
        im = ax.imshow(bathy, origin="lower", cmap="terrain")
        ax.set_title(col_titles[0] if row == 0 else "", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="m")

        # Col 1: solver flood
        ax = axes[row, 1]
        im = ax.imshow(res["solver"][peak_t], origin="lower", cmap="Blues",
                       vmin=0, vmax=vmax_flood)
        ax.contour(bathy, levels=[0], colors="gray", linewidths=0.6)
        ax.set_title(f"{col_titles[1]}\nt={t_h:.1f}h" if row == 0 else f"t={t_h:.1f}h", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="m")

        # Col 2: surrogate flood
        ax = axes[row, 2]
        im = ax.imshow(res["pred"][peak_t], origin="lower", cmap="Blues",
                       vmin=0, vmax=vmax_flood)
        ax.contour(bathy, levels=[0], colors="gray", linewidths=0.6)
        ax.set_title(f"{col_titles[2]}\nrel-L2={l2s[row]:.4f}" if row == 0 else f"rel-L2={l2s[row]:.4f}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="m")

        # Col 3: absolute error
        ax = axes[row, 3]
        im = ax.imshow(err, origin="lower", cmap="Reds", vmin=0)
        ax.set_title(col_titles[3] if row == 0 else "", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, label="m")

        # Col 4: uncertainty half-width
        if not args.no_conformal and res["upper"] is not None:
            ax = axes[row, 4]
            # Width = upper - pred, shown only over flooded cells
            unc = (res["upper"][peak_t - 1] - res["pred"][peak_t])
            unc = np.maximum(unc, 0.0)
            flood_mask = res["pred"][peak_t] > 0.01
            unc[~flood_mask] = 0.0
            im = ax.imshow(unc, origin="lower", cmap="Purples", vmin=0)
            ax.set_title(col_titles[4] if row == 0 else "", fontsize=9)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, label="m")

    plt.suptitle(
        f"Hurricane Flood: Solver vs CoupledOperator  "
        f"({np.mean(speedups):.0f}× speedup, rel-L2={np.mean(l2s):.3f})",
        fontsize=11,
    )
    plt.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    log.info(f"Saved comparison figure → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Benchmark hurricane flood surrogate vs SWE solver")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--n",              type=int, default=10)
    p.add_argument("--n_plot",         type=int, default=3)
    p.add_argument("--seed",           type=int, default=999)
    p.add_argument("--output",         default="results/compare.png")
    p.add_argument("--no_conformal",   action="store_true")
    run_comparison(p.parse_args())
