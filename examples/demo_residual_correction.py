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

from solaris.models import FNO, NeuralResidualCorrector
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger


def make_coarse_solver(n_iter: int, tol: float = 1e-3):
    """Build a coarse Gauss-Seidel FD heat solver as a torch-compatible callable.

    The solver runs only ``n_iter`` iterations — intentionally stopping early
    to produce an approximate solution that the neural corrector will improve.
    """
    try:
        from solver import solve_heat
    except ImportError:
        # Fallback: trivial Laplacian smoother if chip_thermal not on path
        def solve_heat(Q, max_iter=10, tol=1e-3):
            import torch.nn.functional as F_
            T = Q.clone()
            for _ in range(max_iter):
                T = F_.avg_pool2d(
                    T.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1
                ).squeeze()
            return T, 0, max_iter

    def coarse_solver(x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) power map tensor → (B, 1, H, W) coarse temperature."""
        device = x.device
        results = []
        x_np = x.detach().cpu().numpy()
        for i in range(len(x_np)):
            Q = x_np[i, 0]
            T, _, _ = solve_heat(Q, max_iter=n_iter, tol=tol)
            results.append(T)
        T_arr = np.stack(results)[:, None].astype(np.float32)
        return torch.as_tensor(T_arr, device=device)

    return coarse_solver


def generate_data(n: int, res: int, seed: int = 0):
    """Generate (power_map, temperature) pairs.  Returns normalised float32 tensors."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "projects" / "chip_thermal"))
        from solver import random_power_map, solve_heat
    except ImportError:
        log = get_logger("data")
        log.warning("chip_thermal solver not found — using synthetic Gaussian data")
        rng = np.random.default_rng(seed)
        from scipy.ndimage import gaussian_filter
        Q_all, T_all = [], []
        for _ in range(n):
            Q = np.abs(rng.standard_normal((res, res))).astype(np.float32)
            Q = gaussian_filter(Q, sigma=res / 16).astype(np.float32)
            T = gaussian_filter(Q, sigma=res / 8).astype(np.float32)
            Q_all.append(Q)
            T_all.append(T)
        Q_arr = np.stack(Q_all)[:, None]
        T_arr = np.stack(T_all)[:, None]
        return Q_arr, T_arr

    rng = np.random.default_rng(seed)
    Q_all, T_all = [], []
    log = get_logger("data")
    log.info(f"Generating {n} thermal samples …")
    for i in range(n):
        Q = random_power_map(res, res, rng=rng)
        T, _, _ = solve_heat(Q, max_iter=10_000, tol=1e-4)
        Q_all.append(Q)
        T_all.append(T)
        if (i + 1) % max(1, n // 5) == 0:
            log.info(f"  {i+1}/{n}")
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

    Q_norm, _, _ = norm(Q_arr)
    T_norm, T_mean, T_std = norm(T_arr)

    Q_t = torch.as_tensor(Q_norm)
    T_t = torch.as_tensor(T_norm)
    train_ds = TensorDataset(Q_t[: args.n_train], T_t[: args.n_train])
    val_ds = TensorDataset(Q_t[args.n_train :], T_t[args.n_train :])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Coarse solver (20 iterations) ──
    coarse = make_coarse_solver(n_iter=20)

    # ── Model A: pure FNO surrogate ──
    fno = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
        modes=args.modes, dim=2,
    ).to(device)

    # ── Model B: NeuralResidualCorrector ──
    corrector = NeuralResidualCorrector(
        solver=coarse,
        in_channels=1, out_channels=1, solver_out_channels=1,
        hidden_channels=args.hidden, n_layers=args.n_layers,
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
    log.info(f"\nCorrection magnitude analysis:")
    log.info(f"  Coarse solution norm:  {diag['coarse_norm']:.4f}")
    log.info(f"  Neural correction norm:{diag['correction_norm']:.4f}")
    log.info(f"  Relative correction:   {diag['relative_correction']:.4f}")
    log.info(
        "(Relative correction < 0.1 means the net only needs to fix <10% of the solution)"
    )


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
    run(p.parse_args())
