# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
calibrate.py — Conformal calibration of a trained chip-thermal model.

Wraps the best trained model in ConformalNeuralOperator to produce statistically
rigorous uncertainty intervals: P(T_true ∈ [T_low, T_high]) ≥ 1-α for a new
layout drawn from the same distribution as the calibration layouts.

The guarantee needs the calibration layouts to be an ordinary random sample
(exchangeable with future layouts), so they are freshly generated and used
as-is — never filtered by how well the model does on them.  Coverage is then
checked on a separate set of test layouts.

Usage
-----
# Calibrate best FNO
python calibrate.py --model fno --device cuda

# Calibrate best residual corrector
python calibrate.py --model residual --device cuda

# Use a specific checkpoint
python calibrate.py --checkpoint checkpoints/best_residual.pt --model residual
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solver import T_AMBIENT, chip_floorplan_power_map, solve_heat_fd
from train import build_model, make_coarse_solver

from solaris.models.conformal import ConformalNeuralOperator
from solaris.utils import get_logger

# ─── Model loading ───────────────────────────────────────────────────────────


def load_model(
    ckpt_path: Path,
    model_type: str,
    device: torch.device,
    Q_mean: float,
    Q_std: float,
    T_mean: float,
    T_std: float,
):
    """Reconstruct model from checkpoint and load weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hidden = ckpt.get("hidden_channels", 64)
    n_layers = ckpt.get("n_layers", 4)
    modes = ckpt.get("modes", 16)

    coarse_solver = None
    if model_type == "residual":
        coarse_solver = make_coarse_solver(Q_mean, Q_std, T_mean, T_std, coarse_factor=4)
    model = build_model(
        model_type,
        hidden,
        n_layers,
        modes,
        coarse_solver=coarse_solver,
        boundary_value=(T_AMBIENT - T_mean) / T_std,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def calibrate(args):
    log = get_logger("calibrate")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")

    # ── Load norm stats ──
    stats_path = Path(args.checkpoint).parent / "norm_stats.npz"
    stats = np.load(stats_path)
    Q_mean, Q_std = float(stats["Q_mean"]), float(stats["Q_std"])
    T_mean, T_std = float(stats["T_mean"]), float(stats["T_std"])

    # ── Load model ──
    model = load_model(Path(args.checkpoint), args.model, device, Q_mean, Q_std, T_mean, T_std)
    log.info(f"Loaded {args.model} model  |  params: {model.num_parameters():,}")

    # ── Generate a fresh random calibration set ──
    log.info(f"Generating {args.cal_size} calibration layouts …")
    rng = np.random.default_rng(args.seed)
    H = W = args.resolution
    Q_cal = np.stack([chip_floorplan_power_map(H, W, rng) for _ in range(args.cal_size)])
    T_cal = np.stack([solve_heat_fd(Q_cal[i])[0] for i in range(args.cal_size)])
    Q_t = torch.as_tensor(((Q_cal - Q_mean) / Q_std)[:, None], dtype=torch.float32)
    T_t = torch.as_tensor(((T_cal - T_mean) / T_std)[:, None], dtype=torch.float32)

    # ── Calibrate ConformalNeuralOperator ──
    conformal = ConformalNeuralOperator(model)
    q_hat = conformal.calibrate(Q_t.to(device), T_t.to(device), alpha=args.alpha, batch_size=32)
    log.info(
        f"Calibrated q̂ = {q_hat:.4f}  (α={args.alpha}, coverage target ≥ {1 - args.alpha:.0%})"
    )

    # ── Coverage report on held-out test samples ──
    log.info(f"Generating {args.test_size} test layouts for coverage verification …")
    rng2 = np.random.default_rng(args.seed + 999)
    Q_test = np.stack([chip_floorplan_power_map(H, W, rng2) for _ in range(args.test_size)])
    T_test = np.stack([solve_heat_fd(Q_test[i])[0] for i in range(args.test_size)])
    Q_test_n = (Q_test - Q_mean) / Q_std
    T_test_n = (T_test - T_mean) / T_std
    Q_test_t = torch.as_tensor(Q_test_n[:, None], dtype=torch.float32).to(device)
    T_test_t = torch.as_tensor(T_test_n[:, None], dtype=torch.float32).to(device)

    report = conformal.coverage_report(Q_test_t, T_test_t, batch_size=32)
    log.info("")
    log.info("=" * 60)
    log.info(f"  Model              : {args.model}")
    log.info(f"  Calibration set    : {args.cal_size} random layouts")
    log.info(f"  Coverage target    : ≥ {1 - args.alpha:.0%}  (α={args.alpha})")
    log.info(f"  Empirical coverage : {report['coverage']:.1%}")
    log.info(f"  Interval width     : {report['mean_interval_width']:.4f} (normalised)")
    log.info(f"  Interval width     : {report['mean_interval_width'] * T_std:.2f} °C")
    log.info(f"  q̂ threshold        : {report['q_hat']:.4f}")
    log.info("=" * 60)

    # ── Save conformal artefacts ──
    ckpt_dir = Path(args.checkpoint).parent
    ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    save_path = ckpt_dir / f"conformal_{args.model}.pt"
    torch.save(
        {
            "q_hat": report["q_hat"],
            "alpha": args.alpha,
            "cal_size": args.cal_size,
            "coverage": report["coverage"],
            "model_type": args.model,
            "hidden_channels": ckpt_meta.get("hidden_channels", 64),
            "n_layers": ckpt_meta.get("n_layers", 4),
            "modes": ckpt_meta.get("modes", 16),
            "resolution": args.resolution,
        },
        save_path,
    )
    log.info(f"Conformal artefacts saved → {save_path}")

    # ── Optional: uncertainty visualisation ──
    try:
        _visualise(conformal, Q_test_t[:1], T_test_t[:1], T_std, T_mean, args)
    except Exception as exc:
        log.warning(f"Visualisation skipped: {exc}")


def _visualise(conformal, Q_t, T_true_t, T_std, T_mean, args):
    import matplotlib.pyplot as plt

    lo, hi, pt = conformal.predict(Q_t)
    lo_c = lo[0, 0].cpu().numpy() * T_std + T_mean
    hi_c = hi[0, 0].cpu().numpy() * T_std + T_mean
    pt_c = pt[0, 0].cpu().numpy() * T_std + T_mean
    tru_c = T_true_t[0, 0].cpu().numpy() * T_std + T_mean
    width = hi_c - lo_c

    vmin, vmax = tru_c.min(), tru_c.max()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.subplots_adjust(wspace=0.38, left=0.04, right=0.97, top=0.82, bottom=0.06)

    panels = [
        (tru_c, "inferno", vmin, vmax, "Ground Truth T [°C]"),
        (pt_c, "inferno", vmin, vmax, "Predicted T [°C]"),
        (np.abs(pt_c - tru_c), "RdBu_r", None, None, "|Error| [°C]"),
        (
            width,
            "viridis",
            None,
            None,
            f"Interval Width [°C]\n(α={args.alpha}, ≥{1 - args.alpha:.0%} coverage)",
        ),
    ]
    for ax, (data, cmap, vlo, vhi, title) in zip(axes, panels):
        kw = dict(cmap=cmap, origin="lower", interpolation="bilinear")
        if vlo is not None:
            kw.update(vmin=vlo, vmax=vhi)
        im = ax.imshow(data, **kw)
        ax.set_title(title, fontsize=9, color="white", pad=6)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb.outline.set_edgecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_facecolor("#0d0d0d")

    fig.suptitle(
        f"Conformal UQ — {args.model} model  ·  "
        f"q̂ = {conformal._q_hat.item():.4f}  ·  α = {args.alpha}",
        fontsize=12,
        fontweight="bold",
        color="white",
        y=0.97,
    )
    out = Path(args.checkpoint).parent.parent / "results" / f"calibrate_{args.model}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    import logging

    logging.getLogger("calibrate").info(f"Calibration figure → {out}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fno", choices=["fno", "constrained", "residual"])
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to checkpoints/best_{model}.pt)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--cal_size", type=int, default=200, help="Number of random calibration layouts")
    p.add_argument(
        "--test_size",
        type=int,
        default=200,
        help="Number of test samples for coverage verification",
    )
    p.add_argument(
        "--alpha", type=float, default=0.1, help="Miscoverage rate (0.1 → 90%% coverage guarantee)"
    )
    p.add_argument("--seed", type=int, default=7777)
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(Path(__file__).parent / f"checkpoints/best_{args.model}.pt")

    calibrate(args)
