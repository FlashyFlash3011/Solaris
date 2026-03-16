# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Constrained FNO demo — Darcy flow with conservation enforcement.

Trains a ConstrainedFNO on synthetic Darcy flow data and verifies that the
conservation constraint (total pressure integral ≈ total source integral) is
satisfied exactly, even before training converges — demonstrating that the
hard constraint operates at the architecture level, not just as a loss penalty.

Usage
-----
    python examples/train_constrained_darcy.py
    python examples/train_constrained_darcy.py --device cuda --epochs 30
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import solaris
from solaris.models import ConstrainedFNO, FNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger


def synthetic_darcy(n: int, res: int, seed: int = 0):
    """Synthetic Darcy flow: random permeability k → pressure p.
    Uses a blurred random field as a proxy (same as existing FNO Darcy example).
    """
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    K_all, P_all = [], []
    for _ in range(n):
        k = np.exp(rng.standard_normal((res, res)))
        k = gaussian_filter(k, sigma=res / 16)
        k = k / k.mean()
        p = gaussian_filter(k * rng.standard_normal((res, res)), sigma=res / 8)
        K_all.append(k)
        P_all.append(p)
    K = np.stack(K_all)[:, None].astype(np.float32)
    P = np.stack(P_all)[:, None].astype(np.float32)
    return K, P


def check_conservation(model, loader, device):
    """Measure how well the model conserves total pressure integral."""
    violations = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            src_int = x[:, 0].sum(dim=[-2, -1])
            out_int = pred[:, 0].sum(dim=[-2, -1])
            rel_err = ((src_int - out_int).abs() / (src_int.abs() + 1e-8))
            violations.append(rel_err.cpu())
    return torch.cat(violations).mean().item()


def train(args):
    log = get_logger("constrained_darcy")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}  |  PhysicsNeMo {solaris.__version__}")

    K, P = synthetic_darcy(args.n_train + args.n_val, args.res)
    K_t = torch.as_tensor(K)
    P_t = torch.as_tensor(P)

    # Normalise
    K_t = (K_t - K_t.mean()) / (K_t.std() + 1e-8)
    P_t = (P_t - P_t.mean()) / (P_t.std() + 1e-8)

    train_ds = TensorDataset(K_t[: args.n_train], P_t[: args.n_train])
    val_ds = TensorDataset(K_t[args.n_train :], P_t[args.n_train :])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Constrained FNO (conservation constraint) ──
    cmodel = ConstrainedFNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
        modes=args.modes, constraint="conservative",
    ).to(device)

    # ── Baseline FNO (no constraint) ──
    baseline = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
        modes=args.modes, dim=2,
    ).to(device)

    log.info(f"ConstrainedFNO params: {cmodel.num_parameters():,}")
    log.info(f"Baseline FNO params:   {baseline.num_parameters():,}")

    def run_epoch(model, loader, opt=None):
        is_train = opt is not None
        model.train(is_train)
        total_loss, total_l2 = 0.0, 0.0
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)
                if is_train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_loss += loss.item() * len(x)
                total_l2 += relative_l2_error(pred, y).item() * len(x)
        n = len(loader.dataset)
        return total_loss / n, total_l2 / n

    opt_c = torch.optim.AdamW(cmodel.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_b = torch.optim.AdamW(baseline.parameters(), lr=args.lr, weight_decay=1e-4)
    sch_c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_c, T_max=args.epochs)
    sch_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)

    log.info("\n{:<6} {:>10} {:>10} {:>12} {:>12}".format(
        "Epoch", "Cstr-L2", "Base-L2", "Cstr-Viol", "Base-Viol"
    ))
    log.info("-" * 56)

    for ep in range(1, args.epochs + 1):
        run_epoch(cmodel, train_dl, opt_c)
        run_epoch(baseline, train_dl, opt_b)
        sch_c.step()
        sch_b.step()

        _, c_l2 = run_epoch(cmodel, val_dl)
        _, b_l2 = run_epoch(baseline, val_dl)
        c_viol = check_conservation(cmodel, val_dl, device)
        b_viol = check_conservation(baseline, val_dl, device)

        log.info(f"{ep:6d} {c_l2:10.4f} {b_l2:10.4f} {c_viol:12.2e} {b_viol:12.2e}")

    log.info(
        "\nKey result: ConstrainedFNO conservation violation should be ~0 "
        "regardless of epoch; baseline violation only decreases with training."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=4)
    train(p.parse_args())
