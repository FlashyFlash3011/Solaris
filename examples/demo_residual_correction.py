# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Neural Residual Corrector demo — chip thermal surrogate with data efficiency.

Demonstrates that NeuralResidualCorrector achieves lower error than a pure
FNO surrogate when trained on the SAME amount of data.  Key insight: the
residual (difference between coarse FD solution and truth) is small and smooth,
so the network learns it much more easily.

Experiment
----------
1. Generate chip thermal data (power map → temperature) using FD solver.
2. Train a PURE FNO surrogate (standard approach).
3. Train a RESIDUAL CORRECTOR — uses coarse FD (20 iter) + neural correction.
4. Compare validation rel-L2 error at matched parameter counts.

Usage
-----
    python examples/demo_residual_correction.py
    python examples/demo_residual_correction.py --device cuda --n_train 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from project directories
sys.path.insert(0, str(Path(__file__).parent.parent / "projects" / "chip_thermal"))

from solaris.metrics import relative_l2_error
from solaris.models import FNO, NeuralResidualCorrector
from solaris.utils import get_logger


def make_coarse_solver(
    n_iter: int, Q_mean: float, Q_std: float, T_mean: float, T_std: float, T_ambient: float = 40.0
):
    """Build a deliberately under-converged FD heat solver as a torch callable.

    Runs ``n_iter`` Jacobi sweeps of ``-∇²T = Q`` (``T = T_ambient`` on the
    boundary, grid spacing ``h = 1/(H-1)`` — the same discretisation as
    ``projects/chip_thermal/solver.py``) instead of solving to convergence.  The
    result captures the large-scale physics; the neural corrector learns the
    remaining error.

    The callable maps *normalised* power maps to *normalised* temperatures so
    the coarse solution lives in the same space as the training targets.
    """

    def coarse_solver(x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) normalised power map → (B, 1, H, W) normalised temperature."""
        Q = x.detach() * Q_std + Q_mean
        H = Q.shape[-2]
        h2 = (1.0 / (H - 1)) ** 2
        rise = torch.zeros_like(Q)
        for _ in range(n_iter):
            nb = (
                rise[..., :-2, 1:-1]
                + rise[..., 2:, 1:-1]
                + rise[..., 1:-1, :-2]
                + rise[..., 1:-1, 2:]
            )
            interior = (nb + h2 * Q[..., 1:-1, 1:-1]) / 4.0
            rise = torch.nn.functional.pad(interior, (1, 1, 1, 1))  # boundary rise = 0
        return (rise + T_ambient - T_mean) / T_std

    return coarse_solver


def generate_data(n: int, res: int, seed: int = 0):
    """Generate (power_map, temperature) pairs with the chip_thermal FD solver."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "projects" / "chip_thermal"))
    from solver import chip_floorplan_power_map, solve_heat_fd

    rng = np.random.default_rng(seed)
    Q_all, T_all = [], []
    log = get_logger("data")
    log.info(f"Generating {n} thermal samples …")
    for i in range(n):
        Q = chip_floorplan_power_map(res, res, rng=rng)
        T, _ = solve_heat_fd(Q)
        Q_all.append(Q)
        T_all.append(T)
        if (i + 1) % max(1, n // 5) == 0:
            log.info(f"  {i + 1}/{n}")
    return np.stack(Q_all)[:, None].astype(np.float32), np.stack(T_all)[:, None].astype(np.float32)


def run(args):
    log = get_logger("residual_demo")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}")

    Q_arr, T_arr = generate_data(args.n_train + args.n_val, args.res)

    def norm(arr):
        m, s = arr.mean(), arr.std() + 1e-8
        return (arr - m) / s, m, s

    Q_norm, Q_mean, Q_std = norm(Q_arr)
    T_norm, T_mean, T_std = norm(T_arr)

    Q_t = torch.as_tensor(Q_norm)
    T_t = torch.as_tensor(T_norm)
    train_ds = TensorDataset(Q_t[: args.n_train], T_t[: args.n_train])
    val_ds = TensorDataset(Q_t[args.n_train :], T_t[args.n_train :])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Coarse solver: a few hundred Jacobi sweeps (far from converged) ──
    coarse = make_coarse_solver(
        args.coarse_iters, float(Q_mean), float(Q_std), float(T_mean), float(T_std)
    )

    # ── Model A: pure FNO surrogate ──
    fno = FNO(
        in_channels=1,
        out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        modes=args.modes,
        dim=2,
    ).to(device)

    # ── Model B: NeuralResidualCorrector ──
    corrector = NeuralResidualCorrector(
        solver=coarse,
        in_channels=1,
        out_channels=1,
        solver_out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
        modes=args.modes,
    ).to(device)

    log.info(f"Pure FNO params:       {fno.num_parameters():,}")
    log.info(f"Residual corrector params: {corrector.num_parameters():,}")

    def train_model(model, name):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_l2 = float("inf")
        for ep in range(1, args.epochs + 1):
            model.train()
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = nn.functional.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
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
            best_l2 = min(best_l2, val_l2)
            if ep % max(1, args.epochs // 5) == 0:
                log.info(f"  [{name}] ep {ep:3d}/{args.epochs}  val-rel-L2={val_l2:.4f}")
        return best_l2

    log.info("\n=== Training Pure FNO ===")
    fno_l2 = train_model(fno, "FNO")

    log.info("\n=== Training Residual Corrector ===")
    corr_l2 = train_model(corrector, "Corrector")

    log.info("\n" + "=" * 50)
    log.info(f"Pure FNO surrogate   best val rel-L2: {fno_l2:.4f}")
    log.info(f"Residual Corrector   best val rel-L2: {corr_l2:.4f}")
    improvement = (fno_l2 - corr_l2) / fno_l2 * 100
    log.info(f"Relative improvement: {improvement:+.1f}%")
    log.info("=" * 50)

    # Correction diagnostics
    x_sample, _ = next(iter(val_dl))
    diag = corrector.correction_diagnostics(x_sample.to(device))
    log.info("\nCorrection magnitude analysis:")
    log.info(f"  Coarse solution norm:  {diag['coarse_norm']:.4f}")
    log.info(f"  Neural correction norm:{diag['correction_norm']:.4f}")
    log.info(f"  Relative correction:   {diag['relative_correction']:.4f}")
    log.info("(Relative correction < 0.1 means the net only needs to fix <10% of the solution)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--coarse_iters", type=int, default=200)
    run(p.parse_args())
