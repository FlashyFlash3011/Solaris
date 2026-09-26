# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
ConstrainedFNO demo — heat diffusion with exact conservation of total heat.

On a periodic domain the heat equation  ∂u/∂t = κ∇²u  conserves the total
heat ∫u exactly: it only spreads heat around.  We learn the map
u(0) → u(T) with two models:

* a standard FNO, which conserves heat only approximately (and only as well
  as it has been trained);
* ``ConstrainedFNO(constraint="conservative")``, which conserves it exactly
  for any weights — from the very first epoch.

Ground truth is computed exactly in Fourier space: û(k, T) = û(k, 0)·e^{−κ|k|²T}.

Usage
-----
    python examples/train_conservative_diffusion.py
    python examples/train_conservative_diffusion.py --device cuda --epochs 30
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from solaris.metrics import relative_l2_error
from solaris.models import FNO, ConstrainedFNO
from solaris.utils import get_logger


def periodic_diffusion(n: int, res: int, kappa: float = 2e-3, t_end: float = 1.0, seed: int = 0):
    """Random smooth initial fields and their exact diffused state at t_end."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(res, d=1.0 / res)  # integer wavenumbers, domain [0, 2π)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    decay = np.exp(-kappa * k2 * t_end * (2 * np.pi) ** 2 / 4)

    u0 = rng.standard_normal((n, res, res))
    u0 = np.real(np.fft.ifft2(np.fft.fft2(u0) * np.exp(-k2 / 40.0)))  # smooth IC
    u0 += rng.uniform(0.5, 2.0, size=(n, 1, 1))  # non-zero total heat
    uT = np.real(np.fft.ifft2(np.fft.fft2(u0) * decay))
    return u0[:, None].astype(np.float32), uT[:, None].astype(np.float32)


def conservation_violation(model, loader, device):
    """Mean |Σ prediction − Σ input| / Σ|input| over the loader."""
    model.eval()
    vals = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pred = model(x)
            diff = (pred.sum(dim=(1, 2, 3)) - x.sum(dim=(1, 2, 3))).abs()
            vals.append((diff / x.abs().sum(dim=(1, 2, 3))).cpu())
    return torch.cat(vals).mean().item()


def train(args):
    log = get_logger("conservative_diffusion")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}")

    U0, UT = periodic_diffusion(args.n_train + args.n_val, args.res)
    # One shared affine scaling for input and target: Σ(a·u + b) = a·Σu + b·N, so
    # conserving the scaled total is the same as conserving the physical one.
    mean, std = U0.mean(), U0.std() + 1e-8
    U0_t = torch.as_tensor((U0 - mean) / std)
    UT_t = torch.as_tensor((UT - mean) / std)

    train_dl = DataLoader(
        TensorDataset(U0_t[: args.n_train], UT_t[: args.n_train]),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(U0_t[args.n_train :], UT_t[args.n_train :]), batch_size=args.batch_size
    )

    common = dict(hidden_channels=args.hidden, n_layers=args.n_layers, modes=args.modes, norm=None)
    cmodel = ConstrainedFNO(1, 1, constraint="conservative", **common).to(device)
    baseline = FNO(1, 1, dim=2, **common).to(device)

    def run_epoch(model, loader, opt=None):
        is_train = opt is not None
        model.train(is_train)
        total_l2 = 0.0
        with torch.set_grad_enabled(is_train):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                if is_train:
                    loss = nn.functional.mse_loss(pred, y)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                total_l2 += relative_l2_error(pred, y).item() * len(x)
        return total_l2 / len(loader.dataset)

    opts = {
        m: torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
        for m in (cmodel, baseline)
    }

    log.info(
        "{:<6} {:>10} {:>10} {:>12} {:>12}".format(
            "Epoch", "Cstr-L2", "Base-L2", "Cstr-Viol", "Base-Viol"
        )
    )
    for ep in range(1, args.epochs + 1):
        for model, opt in opts.items():
            run_epoch(model, train_dl, opt)
        c_l2, b_l2 = run_epoch(cmodel, val_dl), run_epoch(baseline, val_dl)
        c_viol = conservation_violation(cmodel, val_dl, device)
        b_viol = conservation_violation(baseline, val_dl, device)
        log.info(f"{ep:6d} {c_l2:10.4f} {b_l2:10.4f} {c_viol:12.2e} {b_viol:12.2e}")

    log.info(
        "ConstrainedFNO's violation is at float rounding level from epoch 1; "
        "the baseline's depends on how well it has been trained."
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
