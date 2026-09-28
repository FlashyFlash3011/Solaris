# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Train a CoupledOperator surrogate for hurricane storm surge.

Architecture
------------
  flood_op  — FloodOperator: ConstrainedFNO("nonnegative") × land mask   6→1 channels
  wind_op   — FNO                                                     3→2 channels
  model     — CoupledOperator (learned coupling, 2 rounds)

Physics enforced exactly (for any weights, also after coupling):
  * flood depth ≥ 0 everywhere
  * flood depth = 0 on open-ocean cells (only land floods)

Data are split by *storm* into train / val / calibration / test (70/10/10/10),
so no storm contributes samples to more than one split.  A
ConformalNeuralOperator is calibrated on the calibration storms and its
coverage is reported on the held-out test storms.

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
# Add repo root so 'solaris' is importable when running from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from solver import (
    random_coastal_bathymetry,
    random_hurricane_track,
    run_hurricane_simulation,
)

from solaris.datapipes import cached_arrays, split_by_group
from solaris.models import FNO
from solaris.models.conformal import ConformalNeuralOperator
from solaris.models.constrained_fno import ConstrainedFNO
from solaris.models.coupled import CoupledOperator
from solaris.utils import get_logger, save_checkpoint
from solaris.utils.training import EarlyStopping, GradientClipper

# Input channel layout
FLOOD_CHANNELS = ("h", "bathy", "wind_u", "wind_v", "t", "land")  # flood_in
WIND_CHANNELS = ("wind_u", "wind_v", "t")  # wind_in
LAND = FLOOD_CHANNELS.index("land")
N_FLOOD_IN, N_WIND_IN = len(FLOOD_CHANNELS), len(WIND_CHANNELS)
DATA_FORMAT = 2  # bump when the cached array layout changes


# ---------------------------------------------------------------------------
# Physics-constrained flood operator
# ---------------------------------------------------------------------------


class FloodOperator(nn.Module):
    """Flood-depth operator with two exact physical constraints.

    Wraps a ``ConstrainedFNO(constraint="nonnegative")`` (depth ≥ 0) and
    multiplies its output by the land mask carried in input channel ``LAND``
    (depth = 0 on ocean cells).  :meth:`apply_constraint` re-imposes both after
    the output has been modified, which :class:`CoupledOperator` calls after
    coupling.
    """

    def __init__(
        self, hidden_channels: int, n_layers: int, modes: int, norm: str | None = None
    ) -> None:
        super().__init__()
        self.fno = ConstrainedFNO(
            in_channels=N_FLOOD_IN,
            out_channels=1,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            modes=modes,
            constraint="nonnegative",
            norm=norm,
        )

    @staticmethod
    def _land(x: torch.Tensor) -> torch.Tensor:
        return x[:, LAND : LAND + 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fno(x) * self._land(x)

    def apply_constraint(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return F.relu(out) * self._land(x)


# ---------------------------------------------------------------------------
# FloodExtractor — bridges CoupledOperator ↔ ConformalNeuralOperator
# ---------------------------------------------------------------------------


class FloodExtractor(nn.Module):
    """Wraps CoupledOperator so ConformalNeuralOperator can wrap it.

    Input:  (B, N_FLOOD_IN + N_WIND_IN, H, W) — flood_in then wind_in channels
    Output: (B, 1, H, W) — flood depth prediction only
    """

    def __init__(self, coupled_model: nn.Module) -> None:
        super().__init__()
        self.model = coupled_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model({"flood": x[:, :N_FLOOD_IN], "wind": x[:, N_FLOOD_IN:]})
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
    data_format: int = DATA_FORMAT,
) -> dict:
    """Run n_sims hurricane simulations and extract one-step transition pairs.

    Returns
    -------
    dict with keys:
        'flood_in'   (N, 6, H, W)  [h_t, bathy, wu_t, wv_t, t_norm, land]
        'flood_tgt'  (N, 1, H, W)  [h_{t+1}]
        'wind_in'    (N, 3, H, W)  [wu_t, wv_t, t_norm]
        'wind_tgt'   (N, 2, H, W)  [wu_{t+1}, wv_{t+1}]
        'sim'        (N,)          index of the storm each sample comes from

    N = n_sims × (n_snapshots − 1)
    """
    log = get_logger("generate")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    flood_ins, flood_tgts, wind_ins, wind_tgts, sims = [], [], [], [], []

    log.info(f"Generating {n_sims} hurricane simulations ({H}×{W}) …")
    for i in range(n_sims):
        bathy = random_coastal_bathymetry(H, W, rng=rng)
        track = random_hurricane_track(H, W, n_hours=n_hours, rng=rng)
        max_wind = rng.uniform(35.0, 70.0)
        result = run_hurricane_simulation(
            bathy,
            track,
            n_hours=n_hours,
            n_snapshots=n_snapshots,
            max_wind_ms=max_wind,
        )

        n_snap = len(result["times_h"])
        bathy_tile = bathy[None]  # (1, H, W)
        land_tile = (bathy >= 0.0).astype(np.float32)[None]  # solver's land mask

        for t_idx in range(n_snap - 1):
            t_norm = t_idx / max(n_snap - 2, 1)
            t_map = np.full((1, H, W), t_norm, dtype=np.float32)

            h_t = result["flood"][t_idx][None]
            h_t1 = result["flood"][t_idx + 1][None]
            wu_t = result["wind_u"][t_idx][None]
            wv_t = result["wind_v"][t_idx][None]
            wu_t1 = result["wind_u"][t_idx + 1][None]
            wv_t1 = result["wind_v"][t_idx + 1][None]

            flood_ins.append(
                np.concatenate([h_t, bathy_tile, wu_t, wv_t, t_map, land_tile], axis=0)
            )
            flood_tgts.append(h_t1)
            wind_ins.append(np.concatenate([wu_t, wv_t, t_map], axis=0))
            wind_tgts.append(np.concatenate([wu_t1, wv_t1], axis=0))
            sims.append(i)

        if (i + 1) % max(1, n_sims // 5) == 0:
            log.info(f"  {i + 1}/{n_sims} sims — {time.perf_counter() - t0:.1f}s")

    log.info(f"Dataset: {len(flood_ins)} pairs  ({time.perf_counter() - t0:.1f}s)")
    return {
        "flood_in": np.stack(flood_ins).astype(np.float32),
        "flood_tgt": np.stack(flood_tgts).astype(np.float32),
        "wind_in": np.stack(wind_ins).astype(np.float32),
        "wind_tgt": np.stack(wind_tgts).astype(np.float32),
        "sim": np.asarray(sims, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def compute_norm_stats(data: dict, idx: np.ndarray) -> dict:
    """Per-channel statistics from the training samples ``idx`` only.

    Flood depth (input channel 0 and the target) is only *scaled* — divided by
    its training RMS over flooded cells, never shifted — so zero depth stays
    exactly zero and the non-negativity / ocean constraints hold in normalised
    units too.  The time and land-mask channels are left as they are.
    """
    fi, wi, wt = data["flood_in"][idx], data["wind_in"][idx], data["wind_tgt"][idx]

    def _stats(arr):
        return (
            arr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)[0],
            (arr.std(axis=(0, 2, 3), keepdims=True) + 1e-8).astype(np.float32)[0],
        )

    fi_mean, fi_std = _stats(fi)
    wi_mean, wi_std = _stats(wi)
    wt_mean, wt_std = _stats(wt)

    h = np.concatenate([fi[:, 0].ravel(), data["flood_tgt"][idx].ravel()])
    wet = h[h > 0]
    h_scale = np.float32(np.sqrt((wet**2).mean()) if wet.size else 1.0)

    for ch in (FLOOD_CHANNELS.index("t"), LAND):
        fi_mean[ch], fi_std[ch] = 0.0, 1.0
    fi_mean[0], fi_std[0] = 0.0, h_scale
    wi_mean[WIND_CHANNELS.index("t")], wi_std[WIND_CHANNELS.index("t")] = 0.0, 1.0

    return {
        "fi_mean": fi_mean,
        "fi_std": fi_std,
        "h_scale": h_scale,
        "wi_mean": wi_mean,
        "wi_std": wi_std,
        "wt_mean": wt_mean,
        "wt_std": wt_std,
    }


def apply_normalisation(data: dict, stats: dict) -> dict:
    return {
        "flood_in": (data["flood_in"] - stats["fi_mean"]) / stats["fi_std"],
        "flood_tgt": data["flood_tgt"] / stats["h_scale"],
        "wind_in": (data["wind_in"] - stats["wi_mean"]) / stats["wi_std"],
        "wind_tgt": (data["wind_tgt"] - stats["wt_mean"]) / stats["wt_std"],
    }


def split_storms(sim: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """70 / 10 / 10 / 10 split of whole storms into train / val / cal / test."""
    if len(np.unique(sim)) < 10:
        raise ValueError("need at least 10 storms (--n_sims) for a train/val/cal/test split")
    # Each split_by_group call returns (kept, held_out) index arrays into its input.
    train, held = split_by_group(sim, val_fraction=0.3, seed=seed)
    val_pos, calib_test_pos = split_by_group(sim[held], val_fraction=2 / 3, seed=seed + 1)
    calib_test = held[calib_test_pos]
    cal_pos, test_pos = split_by_group(sim[calib_test], val_fraction=0.5, seed=seed + 2)
    return {
        "train": train,
        "val": held[val_pos],
        "cal": calib_test[cal_pos],
        "test": calib_test[test_pos],
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def build_model(args) -> CoupledOperator:
    norm = getattr(args, "norm", "instance")  # checkpoints without "norm" predate the option
    flood_op = FloodOperator(args.hidden_flood, args.n_layers, args.modes, norm)
    wind_op = FNO(
        in_channels=N_WIND_IN,
        out_channels=2,
        hidden_channels=args.hidden_wind,
        n_layers=args.n_layers,
        modes=args.modes,
        dim=2,
        norm=norm,
    )
    return CoupledOperator(
        operators={"flood": flood_op, "wind": wind_op},
        coupling_channels={"flood": 1, "wind": 2},
        coupling_mode="learned",
        n_coupling_steps=2,
    )


def flood_loss(pred_flood, tgt_flood, pred_wind, tgt_wind, alpha_wind: float = 0.5):
    """MSE on flood depth plus a weighted MSE on the wind field."""
    return F.mse_loss(pred_flood, tgt_flood) + alpha_wind * F.mse_loss(pred_wind, tgt_wind)


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
    data_cfg = {
        "n_sims": args.n_sims,
        "H": args.H,
        "W": args.W,
        "n_hours": args.n_hours,
        "n_snapshots": args.n_snapshots,
        "seed": 42,
        "data_format": DATA_FORMAT,
    }
    raw = cached_arrays(args.cache, data_cfg, lambda: generate_dataset(**data_cfg))
    log.info(f"Dataset: {len(raw['flood_in'])} samples (cache: {args.cache})")

    # ── Splits by storm, normalisation from training storms only ──────
    splits = split_storms(raw["sim"])
    stats = compute_norm_stats(raw, splits["train"])
    np.savez(ckpt_dir / "norm_stats.npz", **stats)
    norm = apply_normalisation(raw, stats)
    tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in norm.items()}

    def subset(name):
        i = torch.as_tensor(splits[name])
        return [tensors[k][i] for k in ("flood_in", "flood_tgt", "wind_in", "wind_tgt")]

    def packed(name):  # (flood_in | wind_in) for the conformal wrapper, and flood target
        fi, ft, wi, _ = subset(name)
        return torch.cat([fi, wi], dim=1), ft

    log.info(
        "Storm split  train/val/cal/test = "
        + "/".join(str(len(np.unique(raw["sim"][splits[k]]))) for k in splits)
    )

    pin = device.type == "cuda"
    train_ds, val_ds = TensorDataset(*subset("train")), TensorDataset(*subset("val"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, pin_memory=pin)

    # ── Model ──────────────────────────────────────────────────────────
    model = build_model(args).to(device)
    n_flood = sum(p.numel() for p in model.operators["flood"].parameters())
    n_wind = sum(p.numel() for p in model.operators["wind"].parameters())
    log.info(f"CoupledOperator  flood={n_flood:,}  wind={n_wind:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(patience=15, min_delta=1e-5, mode="min")
    clipper = GradientClipper(max_norm=1.0)

    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for fi, ft, wi, wt in train_loader:
            fi, ft, wi, wt = fi.to(device), ft.to(device), wi.to(device), wt.to(device)
            optimizer.zero_grad()
            outputs = model({"flood": fi, "wind": wi})
            loss = flood_loss(outputs["flood"], ft, outputs["wind"], wt, args.alpha_wind)
            loss.backward()
            clipper(model)
            optimizer.step()
            tr_loss += loss.item() * len(fi)
        tr_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fi, ft, wi, wt in val_loader:
                fi, ft, wi, wt = fi.to(device), ft.to(device), wi.to(device), wt.to(device)
                outputs = model({"flood": fi, "wind": wi})
                val_loss += flood_loss(
                    outputs["flood"], ft, outputs["wind"], wt, args.alpha_wind
                ).item() * len(fi)
        val_loss /= len(val_ds)

        log.info(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={tr_loss:.3e}  val={val_loss:.3e}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({time.perf_counter() - t0:.0f}s)"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / "best_coupled.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                extra={
                    "H": args.H,
                    "W": args.W,
                    "n_hours": args.n_hours,
                    "n_snapshots": args.n_snapshots,
                    "hidden_flood": args.hidden_flood,
                    "hidden_wind": args.hidden_wind,
                    "n_layers": args.n_layers,
                    "modes": args.modes,
                    "norm": args.norm,
                },
            )

        if early_stop.step(val_loss):
            log.info(f"Early stopping at epoch {epoch}")
            break

    log.info(f"Training done. Best val loss: {best_val:.4e}  → {ckpt_dir}/best_coupled.pt")

    # ── Conformal calibration (cal storms) + coverage (test storms) ──────
    ckpt = torch.load(ckpt_dir / "best_coupled.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    predictor = ConformalNeuralOperator(FloodExtractor(model)).to(device)

    cal_x, cal_y = packed("cal")
    q_hat = predictor.calibrate(cal_x, cal_y, alpha=args.conformal_alpha)
    log.info(
        f"Conformal q̂ = {q_hat:.4f} (normalised) = {q_hat * float(stats['h_scale']):.3f} m  "
        f"target coverage ≥ {1 - args.conformal_alpha:.0%}"
    )

    test_x, test_y = packed("test")
    report = predictor.coverage_report(test_x, test_y)
    log.info(
        f"Held-out test storms: coverage {report['coverage']:.3f}  "
        f"mean interval width {report['mean_interval_width'] * float(stats['h_scale']):.3f} m"
    )

    torch.save(predictor.state_dict(), ckpt_dir / "conformal_predictor.pt")
    log.info(f"Saved conformal predictor → {ckpt_dir}/conformal_predictor.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train CoupledOperator hurricane flood surrogate")
    p.add_argument("--device", default="cpu")
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--n_sims", type=int, default=200)
    p.add_argument("--n_hours", type=int, default=24)
    p.add_argument("--n_snapshots", type=int, default=25)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_flood", type=int, default=64)
    p.add_argument("--hidden_wind", type=int, default=48)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument(
        "--norm",
        default="none",
        choices=["none", "instance"],
        help="Normalisation inside Fourier blocks (none = reference FNO)",
    )
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--alpha_wind", type=float, default=0.5)
    p.add_argument("--conformal_alpha", type=float, default=0.1)
    p.add_argument("--cache", default="data/hurricane_dataset.npz")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    args = p.parse_args()
    args.norm = None if args.norm == "none" else args.norm
    train(args)
