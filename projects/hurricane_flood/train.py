# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train a CoupledOperator surrogate for hurricane storm surge.

Architecture
------------
  flood_op  — ConstrainedFNO (conservative)   5→1 channels
  wind_op   — ConstrainedFNO (divergence_free) 3→2 channels
  model     — CoupledOperator (learned coupling, 2 rounds)

After training, a ConformalNeuralOperator is calibrated on a held-out
split to give 90% coverage uncertainty bounds on flood depth predictions.

Usage
-----
# Quick CPU test
python train.py --n_sims 20 --epochs 5

# Full run
python train.py --n_sims 500 --epochs 80 --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
# Add repo root so 'solaris' is importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from solver import (
    random_coastal_bathymetry,
    random_hurricane_track,
    run_hurricane_simulation,
)

from solaris.models.conformal import ConformalNeuralOperator
from solaris.models.constrained_fno import ConstrainedFNO
from solaris.models.coupled import CoupledOperator
from solaris.utils import get_logger, save_checkpoint
from solaris.utils.training import EarlyStopping, GradientClipper


# ---------------------------------------------------------------------------
# FloodExtractor — bridges CoupledOperator ↔ ConformalNeuralOperator
# ---------------------------------------------------------------------------

class FloodExtractor(nn.Module):
    """Wraps CoupledOperator so ConformalNeuralOperator can wrap it.

    Input:  (B, 8, H, W)  — flood_in[:5] concat wind_in[5:8]
    Output: (B, 1, H, W)  — flood depth prediction only
    """

    def __init__(self, coupled_model: nn.Module) -> None:
        super().__init__()
        self.model = coupled_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flood_in = x[:, :5]
        wind_in  = x[:, 5:]
        outputs  = self.model({"flood": flood_in, "wind": wind_in})
        return outputs["flood"]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    n_sims: int,
    H: int,
    W: int,
    n_hours: int,
    n_snapshots: int,
    seed: int = 42,
) -> dict:
    """Run n_sims hurricane simulations and extract one-step transition pairs.

    Returns
    -------
    dict with keys:
        'flood_in'   (N, 5, H, W)  [h_t, bathy, wu_t, wv_t, t_norm]
        'flood_tgt'  (N, 1, H, W)  [h_{t+1}]
        'wind_in'    (N, 3, H, W)  [wu_t, wv_t, t_norm]
        'wind_tgt'   (N, 2, H, W)  [wu_{t+1}, wv_{t+1}]

    N = n_sims × (n_snapshots − 1)
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    flood_ins, flood_tgts = [], []
    wind_ins, wind_tgts   = [], []

    log.info(f"Generating {n_sims} hurricane simulations ({H}×{W}) …")
    for i in range(n_sims):
        bathy = random_coastal_bathymetry(H, W, rng=rng)
        track = random_hurricane_track(H, W, n_hours=n_hours, rng=rng)
        max_wind = rng.uniform(35.0, 70.0)
        result = run_hurricane_simulation(
            bathy, track,
            n_hours=n_hours,
            n_snapshots=n_snapshots,
            max_wind_ms=max_wind,
        )

        n_snap = len(result["times_h"])
        # Broadcast bathy to (1, H, W) for concatenation
        bathy_tile = bathy[None]  # (1, H, W)

        for t_idx in range(n_snap - 1):
            t_norm = t_idx / max(n_snap - 2, 1)
            t_map  = np.full((1, H, W), t_norm, dtype=np.float32)

            h_t   = result["flood"][t_idx][None]      # (1,H,W) flood depth
            h_t1  = result["flood"][t_idx + 1][None]  # (1,H,W)
            wu_t  = result["wind_u"][t_idx][None]      # (1,H,W)
            wv_t  = result["wind_v"][t_idx][None]
            wu_t1 = result["wind_u"][t_idx + 1][None]
            wv_t1 = result["wind_v"][t_idx + 1][None]

            flood_ins.append(np.concatenate([h_t, bathy_tile, wu_t, wv_t, t_map], axis=0))
            flood_tgts.append(h_t1)
            wind_ins.append(np.concatenate([wu_t, wv_t, t_map], axis=0))
            wind_tgts.append(np.concatenate([wu_t1, wv_t1], axis=0))

        if (i + 1) % max(1, n_sims // 5) == 0:
            log.info(f"  {i+1}/{n_sims} sims — {time.perf_counter()-t0:.1f}s")

    log.info(f"Dataset: {len(flood_ins)} pairs  ({time.perf_counter()-t0:.1f}s)")
    return {
        "flood_in":  np.stack(flood_ins).astype(np.float32),
        "flood_tgt": np.stack(flood_tgts).astype(np.float32),
        "wind_in":   np.stack(wind_ins).astype(np.float32),
        "wind_tgt":  np.stack(wind_tgts).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_norm_stats(data: dict) -> dict:
    """Compute per-channel mean/std for all fields. h uses log1p first."""
    fi = data["flood_in"].copy()
    ft = data["flood_tgt"].copy()

    # log1p transform on h channels (channel 0 of flood_in and flood_tgt)
    fi[:, 0] = np.log1p(fi[:, 0])
    ft[:, 0] = np.log1p(ft[:, 0])

    def _stats(arr, axis):
        mean = arr.mean(axis=axis, keepdims=True)
        std  = arr.std(axis=axis,  keepdims=True) + 1e-8
        return mean.astype(np.float32), std.astype(np.float32)

    spatial_axes = (0, 2, 3)
    fi_mean, fi_std = _stats(fi,               spatial_axes)
    ft_mean, ft_std = _stats(ft,               spatial_axes)
    wi_mean, wi_std = _stats(data["wind_in"],  spatial_axes)
    wt_mean, wt_std = _stats(data["wind_tgt"], spatial_axes)

    # t_norm channel (index 4 of flood_in, index 2 of wind_in) — already [0,1]
    fi_mean[0, 4], fi_std[0, 4] = 0.0, 1.0
    wi_mean[0, 2], wi_std[0, 2] = 0.0, 1.0

    return {
        "fi_mean": fi_mean, "fi_std": fi_std,
        "ft_mean": ft_mean, "ft_std": ft_std,
        "wi_mean": wi_mean, "wi_std": wi_std,
        "wt_mean": wt_mean, "wt_std": wt_std,
    }


def apply_normalisation(data: dict, stats: dict) -> dict:
    """Normalise data in-place (log1p on h channels, then z-score)."""
    fi = data["flood_in"].copy()
    ft = data["flood_tgt"].copy()
    fi[:, 0] = np.log1p(fi[:, 0])
    ft[:, 0] = np.log1p(ft[:, 0])

    return {
        "flood_in":  (fi - stats["fi_mean"]) / stats["fi_std"],
        "flood_tgt": (ft - stats["ft_mean"]) / stats["ft_std"],
        "wind_in":   (data["wind_in"]  - stats["wi_mean"]) / stats["wi_std"],
        "wind_tgt":  (data["wind_tgt"] - stats["wt_mean"]) / stats["wt_std"],
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def flood_loss(
    pred_flood: torch.Tensor,
    tgt_flood:  torch.Tensor,
    pred_wind:  torch.Tensor,
    tgt_wind:   torch.Tensor,
    alpha_wind: float = 0.5,
    alpha_cons: float = 0.1,
) -> torch.Tensor:
    """Combined MSE loss for flood depth + wind field.

    Includes a soft conservation penalty as an extra signal (the hard
    constraint in ConstrainedFNO handles most of this, but the penalty
    catches any residual from the log1p denormalisation).
    """
    mse = nn.functional.mse_loss
    L_flood = mse(pred_flood, tgt_flood)
    L_wind  = mse(pred_wind, tgt_wind)

    # Soft conservation: sum of predicted flood ≈ sum of target flood
    sum_pred = pred_flood.sum(dim=(-2, -1))
    sum_tgt  = tgt_flood.sum(dim=(-2, -1))
    L_cons = ((sum_pred - sum_tgt) ** 2 / (sum_tgt ** 2 + 1e-6)).mean()

    return L_flood + alpha_wind * L_wind + alpha_cons * L_cons


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args) -> CoupledOperator:
    flood_op = ConstrainedFNO(
        in_channels=5,
        out_channels=1,
        hidden_channels=args.hidden_flood,
        n_layers=args.n_layers,
        modes=args.modes,
        constraint="conservative",
    )
    wind_op = ConstrainedFNO(
        in_channels=3,
        out_channels=2,            # divergence_free requires exactly 2
        hidden_channels=args.hidden_wind,
        n_layers=args.n_layers,
        modes=args.modes,
        constraint="divergence_free",
    )
    return CoupledOperator(
        operators={"flood": flood_op, "wind": wind_op},
        coupling_channels={"flood": 1, "wind": 2},
        coupling_mode="learned",
        n_coupling_steps=2,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args) -> None:
    log = get_logger("hurricane_train")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    # ── Dataset ────────────────────────────────────────────────────────
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        raw = {k: d[k] for k in ["flood_in", "flood_tgt", "wind_in", "wind_tgt"]}
    else:
        raw = generate_dataset(
            args.n_sims, args.H, args.W, args.n_hours, args.n_snapshots, seed=42,
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **raw)
        log.info(f"Saved dataset → {cache}")

    N = len(raw["flood_in"])
    log.info(f"Total samples: {N}  flood_in shape: {raw['flood_in'].shape}")

    # ── Normalisation ──────────────────────────────────────────────────
    stats = compute_norm_stats(raw)
    np.savez(ckpt_dir / "norm_stats.npz", **stats)
    norm = apply_normalisation(raw, stats)

    # ── Splits: 80% train / 10% val / 10% calibration ─────────────────
    n_train = int(N * 0.80)
    n_val   = int(N * 0.10)

    def _split(arr, start, end):
        return torch.as_tensor(arr[start:end], dtype=torch.float32)

    fi_t  = torch.as_tensor(norm["flood_in"],  dtype=torch.float32)
    ft_t  = torch.as_tensor(norm["flood_tgt"], dtype=torch.float32)
    wi_t  = torch.as_tensor(norm["wind_in"],   dtype=torch.float32)
    wt_t  = torch.as_tensor(norm["wind_tgt"],  dtype=torch.float32)

    # Pack flood + wind inputs for conformal (ConformalNeuralOperator needs single tensor)
    combined = torch.cat([fi_t, wi_t], dim=1)  # (N, 8, H, W)

    train_ds = TensorDataset(
        fi_t[:n_train], ft_t[:n_train], wi_t[:n_train], wt_t[:n_train],
    )
    val_ds = TensorDataset(
        fi_t[n_train:n_train+n_val], ft_t[n_train:n_train+n_val],
        wi_t[n_train:n_train+n_val], wt_t[n_train:n_train+n_val],
    )
    cal_combined = combined[n_train+n_val:].to(device)
    cal_flood_tgt = ft_t[n_train+n_val:].to(device)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,                pin_memory=pin)
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Cal: {N - n_train - n_val}")

    # ── Model ──────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    n_flood = sum(p.numel() for p in model.operators["flood"].parameters())
    n_wind  = sum(p.numel() for p in model.operators["wind"].parameters())
    log.info(f"CoupledOperator  flood={n_flood:,}  wind={n_wind:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(patience=15, min_delta=1e-5, mode="min")
    clipper    = GradientClipper(max_norm=1.0)

    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        tr_loss = 0.0
        for fi, ft, wi, wt in train_loader:
            fi, ft, wi, wt = fi.to(device), ft.to(device), wi.to(device), wt.to(device)
            optimizer.zero_grad()
            outputs = model({"flood": fi, "wind": wi})
            loss = flood_loss(outputs["flood"], ft, outputs["wind"], wt,
                              args.alpha_wind, args.alpha_cons)
            loss.backward()
            clipper(model)
            optimizer.step()
            tr_loss += loss.item() * len(fi)
        tr_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fi, ft, wi, wt in val_loader:
                fi, ft, wi, wt = fi.to(device), ft.to(device), wi.to(device), wt.to(device)
                outputs = model({"flood": fi, "wind": wi})
                val_loss += flood_loss(outputs["flood"], ft, outputs["wind"], wt,
                                       args.alpha_wind, args.alpha_cons).item() * len(fi)
        val_loss /= len(val_ds)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={tr_loss:.3e}  val={val_loss:.3e}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({time.perf_counter()-t0:.0f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best_coupled.pt",
                model, optimizer, scheduler, epoch, val_loss,
                extra={"H": args.H, "W": args.W,
                       "hidden_flood": args.hidden_flood,
                       "hidden_wind": args.hidden_wind,
                       "n_layers": args.n_layers,
                       "modes": args.modes},
            )

        if early_stop.step(val_loss):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Training done. Best val loss: {best_val:.4e}  → {ckpt_dir}/best_coupled.pt")

    # ── Conformal calibration ──────────────────────────────────────────
    log.info("Calibrating ConformalNeuralOperator on held-out set …")
    model.eval()
    predictor = ConformalNeuralOperator(FloodExtractor(model))
    predictor.to(device)

    q_hat = predictor.calibrate(cal_combined, cal_flood_tgt, alpha=args.conformal_alpha)
    log.info(f"Conformal q_hat = {q_hat:.4f}  (90% coverage on flood depth, normalised units)")

    # Coverage report
    report = predictor.coverage_report(cal_combined, cal_flood_tgt)
    log.info(
        f"Coverage: {report['coverage']:.3f}  "
        f"interval width: {report['mean_interval_width']:.4f}"
    )

    torch.save(predictor.state_dict(), ckpt_dir / "conformal_predictor.pt")
    log.info(f"Saved conformal predictor → {ckpt_dir}/conformal_predictor.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CoupledOperator hurricane flood surrogate")
    p.add_argument("--device",          default="cpu")
    p.add_argument("--H",               type=int,   default=64)
    p.add_argument("--W",               type=int,   default=64)
    p.add_argument("--n_sims",          type=int,   default=200)
    p.add_argument("--n_hours",         type=int,   default=24)
    p.add_argument("--n_snapshots",     type=int,   default=25)
    p.add_argument("--epochs",          type=int,   default=80)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--hidden_flood",    type=int,   default=64)
    p.add_argument("--hidden_wind",     type=int,   default=48)
    p.add_argument("--n_layers",        type=int,   default=4)
    p.add_argument("--modes",           type=int,   default=16)
    p.add_argument("--alpha_wind",      type=float, default=0.5)
    p.add_argument("--alpha_cons",      type=float, default=0.1)
    p.add_argument("--conformal_alpha", type=float, default=0.1)
    p.add_argument("--cache",           default="data/hurricane_dataset.npz")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    train(p.parse_args())
