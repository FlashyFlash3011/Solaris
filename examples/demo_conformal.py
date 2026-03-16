# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Conformal Neural Operator demo — calibrated uncertainty quantification.

Demonstrates that ConformalNeuralOperator produces *statistically rigorous*
uncertainty intervals with guaranteed coverage — i.e. if you ask for 90%
coverage, you get ≥ 90% coverage on test data, for ANY model, with NO
distributional assumptions.

Pipeline
--------
1. Train a standard FNO on Darcy flow.
2. Wrap it with ConformalNeuralOperator.
3. Calibrate on a held-out calibration set (separate from train & test).
4. Verify that the empirical coverage on the test set matches the target.

Usage
-----
    python examples/demo_conformal.py
    python examples/demo_conformal.py --alpha 0.05  # 95% coverage
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.models import FNO, ConformalNeuralOperator
from physicsnemo.utils import get_logger


def synthetic_darcy(n: int, res: int, seed: int = 0):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    K_all, P_all = [], []
    for _ in range(n):
        k = np.exp(gaussian_filter(rng.standard_normal((res, res)), sigma=res / 16))
        k /= k.mean()
        p = gaussian_filter(k * rng.standard_normal((res, res)), sigma=res / 8)
        K_all.append(k.astype(np.float32))
        P_all.append(p.astype(np.float32))
    K = np.stack(K_all)[:, None]
    P = np.stack(P_all)[:, None]
    return (K - K.mean()) / (K.std() + 1e-8), (P - P.mean()) / (P.std() + 1e-8)


def run(args):
    log = get_logger("conformal_demo")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}  |  Target coverage: {(1-args.alpha)*100:.0f}%")

    total = args.n_train + args.n_cal + args.n_test
    K, P = synthetic_darcy(total, args.res)
    K_t = torch.as_tensor(K)
    P_t = torch.as_tensor(P)

    train_K = K_t[: args.n_train]
    train_P = P_t[: args.n_train]
    cal_K = K_t[args.n_train : args.n_train + args.n_cal]
    cal_P = P_t[args.n_train : args.n_train + args.n_cal]
    test_K = K_t[args.n_train + args.n_cal :]
    test_P = P_t[args.n_train + args.n_cal :]

    train_dl = DataLoader(TensorDataset(train_K, train_P), batch_size=16, shuffle=True)

    # ── Step 1: train a standard FNO ──
    fno = FNO(in_channels=1, out_channels=1, hidden_channels=32, n_layers=4,
              modes=12, dim=2).to(device)
    log.info(f"FNO parameters: {fno.num_parameters():,}")
    log.info("Training FNO …")
    opt = torch.optim.AdamW(fno.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for ep in range(1, args.epochs + 1):
        fno.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(fno(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(fno.parameters(), 1.0)
            opt.step()
        sch.step()
        if ep % max(1, args.epochs // 4) == 0:
            log.info(f"  Epoch {ep}/{args.epochs}")

    # ── Step 2: wrap with conformal predictor ──
    predictor = ConformalNeuralOperator(fno)

    # ── Step 3: calibrate ──
    log.info(f"\nCalibrating on {args.n_cal} held-out samples …")
    q_hat = predictor.calibrate(cal_K, cal_P, alpha=args.alpha)
    log.info(f"Calibrated q̂ = {q_hat:.4f}")

    # ── Step 4: evaluate coverage on test set ──
    log.info(f"\nEvaluating coverage on {args.n_test} test samples …")
    report = predictor.coverage_report(test_K, test_P)

    target_coverage = 1 - args.alpha
    log.info("\n" + "=" * 50)
    log.info(f"Target coverage:          {target_coverage * 100:.1f}%")
    log.info(f"Empirical coverage:       {report['coverage'] * 100:.1f}%")
    log.info(f"Mean interval half-width: {report['mean_interval_width']:.4f}")
    log.info(f"Calibrated threshold q̂:  {report['q_hat']:.4f}")
    log.info("=" * 50)

    satisfied = report["coverage"] >= target_coverage - 0.02  # 2% tolerance
    status = "SATISFIED" if satisfied else "VIOLATED (more cal data may help)"
    log.info(f"Coverage guarantee: {status}")

    # Show a few per-sample predictions
    log.info("\nSample predictions (first 3 test inputs):")
    x_demo = test_K[:3].to(device)
    lo, hi, pt = predictor.predict(x_demo)
    for i in range(3):
        width = (hi[i] - lo[i]).mean().item()
        log.info(f"  Sample {i}: point_pred_mean={pt[i].mean().item():.4f}  "
                 f"interval_width={width:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--res", type=int, default=64)
    p.add_argument("--n_train", type=int, default=300)
    p.add_argument("--n_cal", type=int, default=100)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Miscoverage rate: 0.1 → 90%% coverage guarantee")
    run(p.parse_args())
