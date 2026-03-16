# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
MultiScaleFNO demo — multi-frequency Darcy flow.

Trains MultiScaleFNO and standard FNO on a synthetic Darcy flow problem that
contains *both* large-scale smooth structure and fine-scale sharp features.
This dataset is specifically designed to stress-test high-frequency accuracy,
where standard FNO is known to struggle.

Usage
-----
    python examples/train_multiscale_fno.py
    python examples/train_multiscale_fno.py --device cuda --epochs 40
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaris.models import FNO, MultiScaleFNO
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger


def multiscale_darcy(n: int, res: int, seed: int = 0):
    """Synthetic Darcy flow with deliberately multi-scale structure.

    The permeability field combines:
    - A smooth large-scale component   (low-frequency: big blobs)
    - A medium-scale random field      (mid-frequency: eddies)
    - Sharp interface inclusions       (high-frequency: step discontinuities)

    This multi-scale structure makes it hard for standard FNO to learn well.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    K_all, P_all = [], []

    for _ in range(n):
        # Low-frequency background
        k_low = gaussian_filter(rng.standard_normal((res, res)), sigma=res / 8)
        # Mid-frequency texture
        k_mid = gaussian_filter(rng.standard_normal((res, res)), sigma=res / 32)
        # High-frequency sharp inclusions (binary)
        k_high = (rng.random((res, res)) > 0.85).astype(float) * 3.0

        k = np.exp(k_low * 0.6 + k_mid * 0.3 + k_high * 0.1)
        k = k / k.mean()

        # Pressure field via blurred forcing
        p = gaussian_filter(k * rng.standard_normal((res, res)), sigma=res / 12)
        K_all.append(k.astype(np.float32))
        P_all.append(p.astype(np.float32))

    K = np.stack(K_all)[:, None]
    P = np.stack(P_all)[:, None]
    return K, P


def train(args):
    log = get_logger("multiscale_fno")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}")

    K, P = multiscale_darcy(args.n_train + args.n_val, args.res)
    K_t = torch.as_tensor(K)
    P_t = torch.as_tensor(P)
    K_t = (K_t - K_t.mean()) / (K_t.std() + 1e-8)
    P_t = (P_t - P_t.mean()) / (P_t.std() + 1e-8)

    train_ds = TensorDataset(K_t[: args.n_train], P_t[: args.n_train])
    val_ds = TensorDataset(K_t[args.n_train :], P_t[args.n_train :])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # ── MultiScaleFNO ──
    ms_model = MultiScaleFNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
        n_scales=3, max_modes=args.modes,
    ).to(device)

    # ── Baseline FNO ──
    fno = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
        modes=args.modes, dim=2,
    ).to(device)

    log.info(f"MultiScaleFNO params: {ms_model.num_parameters():,}")
    log.info(f"Baseline FNO params:  {fno.num_parameters():,}")

    def train_one(model, name):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf")
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                loss = nn.functional.mse_loss(model(x), y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()

            model.eval()
            val_l2 = 0.0
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    val_l2 += relative_l2_error(model(x), y).item() * len(x)
            val_l2 /= len(val_ds)
            best = min(best, val_l2)
            if ep % max(1, args.epochs // 5) == 0:
                log.info(f"  [{name}] ep {ep:3d}/{args.epochs}  val-rel-L2={val_l2:.4f}")
        return best

    log.info("\n=== Training MultiScaleFNO ===")
    ms_l2 = train_one(ms_model, "MultiScaleFNO")

    log.info("\n=== Training Baseline FNO ===")
    fno_l2 = train_one(fno, "FNO")

    log.info("\n" + "=" * 50)
    log.info(f"MultiScaleFNO best val rel-L2: {ms_l2:.4f}")
    log.info(f"Baseline FNO  best val rel-L2: {fno_l2:.4f}")
    improvement = (fno_l2 - ms_l2) / fno_l2 * 100
    log.info(f"Relative improvement on multi-scale data: {improvement:+.1f}%")
    log.info("(Higher gain expected on data with sharp high-frequency inclusions)")
    log.info("=" * 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n_train", type=int, default=400)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=4)
    train(p.parse_args())
