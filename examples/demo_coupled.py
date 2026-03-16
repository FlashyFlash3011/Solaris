# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
CoupledOperator demo — multi-physics thermal + flow coupling.

Demonstrates that a CoupledOperator with learned coupling achieves lower error
than running two independent neural operators — because temperature affects
fluid viscosity (thermal-fluid coupling) and vice versa.

The synthetic problem
---------------------
Two coupled 2-D fields on a 64×64 grid:
  - T(x,y):   temperature field driven by a heat source Q(x,y)
  - ψ(x,y):   stream function driven by buoyancy (proportional to ∇T)

In reality T drives ψ (hot fluid rises) and ψ advects T (flow redistributes
heat).  Two independent operators miss this loop; the CoupledOperator learns it.

Usage
-----
    python examples/demo_coupled.py
    python examples/demo_coupled.py --device cuda --epochs 30
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.models import FNO, CoupledOperator
from physicsnemo.metrics import relative_l2_error
from physicsnemo.utils import get_logger


def generate_coupled_data(n: int, res: int, seed: int = 0):
    """Synthetic coupled thermal-buoyancy dataset.

    Input:  Q(x,y)  — random heat source field  (1 channel)
    Output: T(x,y)  — temperature               (1 channel)
            psi(x,y)— stream function            (1 channel)

    Both output fields depend on the input *and on each other*:
        T = smooth(Q)  + coupling_term(psi)
        psi = laplacian_inv(dT/dy) * buoyancy_strength
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    Q_all, T_all, Psi_all = [], [], []

    buoyancy = 0.3  # coupling strength

    for _ in range(n):
        Q = rng.standard_normal((res, res)).astype(np.float32)
        Q = gaussian_filter(Q, sigma=res / 16)

        # Temperature: driven by source + buoyancy feedback
        T = gaussian_filter(Q, sigma=res / 8)

        # Stream function: driven by vertical temperature gradient (buoyancy)
        dT_dy = np.gradient(T, axis=0)
        psi = gaussian_filter(dT_dy, sigma=res / 12) * buoyancy

        # Thermal-flow feedback: flow advects temperature
        dpsi_dx = np.gradient(psi, axis=1)
        T = T + 0.1 * gaussian_filter(dpsi_dx, sigma=2)

        Q_all.append(Q); T_all.append(T); Psi_all.append(psi)

    Q = np.stack(Q_all)[:, None].astype(np.float32)
    T = np.stack(T_all)[:, None].astype(np.float32)
    Psi = np.stack(Psi_all)[:, None].astype(np.float32)

    def norm(arr):
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    return norm(Q), norm(T), norm(Psi)


def run(args):
    log = get_logger("coupled_demo")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}")

    Q, T, Psi = generate_coupled_data(args.n_train + args.n_val, args.res)
    Q_t = torch.as_tensor(Q)
    T_t = torch.as_tensor(T)
    Psi_t = torch.as_tensor(Psi)

    train_ds = TensorDataset(Q_t[: args.n_train], T_t[: args.n_train], Psi_t[: args.n_train])
    val_ds   = TensorDataset(Q_t[args.n_train :], T_t[args.n_train :], Psi_t[args.n_train :])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    hidden, modes, layers = args.hidden, args.modes, args.n_layers

    # ── Model A: two independent FNOs ──
    fno_T   = FNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                  n_layers=layers, modes=modes, dim=2).to(device)
    fno_Psi = FNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                  n_layers=layers, modes=modes, dim=2).to(device)

    # ── Model B: CoupledOperator with learned coupling ──
    coupled = CoupledOperator(
        operators={
            "thermal": FNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                           n_layers=layers, modes=modes, dim=2),
            "flow":    FNO(in_channels=1, out_channels=1, hidden_channels=hidden,
                           n_layers=layers, modes=modes, dim=2),
        },
        coupling_channels={"thermal": 1, "flow": 1},
        coupling_mode="learned",
        n_coupling_steps=2,
    ).to(device)

    ind_params = fno_T.num_parameters() + fno_Psi.num_parameters()
    log.info(f"Independent operators total params: {ind_params:,}")
    log.info(f"CoupledOperator total params:       {coupled.num_parameters():,}")

    def train_independent():
        opt = torch.optim.AdamW(
            list(fno_T.parameters()) + list(fno_Psi.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf")
        for ep in range(1, args.epochs + 1):
            fno_T.train(); fno_Psi.train()
            for q, t, psi in train_dl:
                q, t, psi = q.to(device), t.to(device), psi.to(device)
                loss = nn.functional.mse_loss(fno_T(q), t) + \
                       nn.functional.mse_loss(fno_Psi(q), psi)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(fno_T.parameters()) + list(fno_Psi.parameters()), 1.0
                )
                opt.step()
            sch.step()

            fno_T.eval(); fno_Psi.eval()
            vl2 = 0.0
            with torch.no_grad():
                for q, t, psi in val_dl:
                    q, t, psi = q.to(device), t.to(device), psi.to(device)
                    vl2 += (relative_l2_error(fno_T(q), t) +
                            relative_l2_error(fno_Psi(q), psi)).item() / 2 * len(q)
            vl2 /= len(val_ds)
            best = min(best, vl2)
            if ep % max(1, args.epochs // 4) == 0:
                log.info(f"  [Independent] ep {ep}/{args.epochs}  val-rel-L2={vl2:.4f}")
        return best

    def train_coupled():
        opt = torch.optim.AdamW(coupled.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf")
        for ep in range(1, args.epochs + 1):
            coupled.train()
            for q, t, psi in train_dl:
                q, t, psi = q.to(device), t.to(device), psi.to(device)
                out = coupled({"thermal": q, "flow": q})
                loss = nn.functional.mse_loss(out["thermal"], t) + \
                       nn.functional.mse_loss(out["flow"], psi)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(coupled.parameters(), 1.0)
                opt.step()
            sch.step()

            coupled.eval()
            vl2 = 0.0
            with torch.no_grad():
                for q, t, psi in val_dl:
                    q, t, psi = q.to(device), t.to(device), psi.to(device)
                    out = coupled({"thermal": q, "flow": q})
                    vl2 += (relative_l2_error(out["thermal"], t) +
                            relative_l2_error(out["flow"], psi)).item() / 2 * len(q)
            vl2 /= len(val_ds)
            best = min(best, vl2)
            if ep % max(1, args.epochs // 4) == 0:
                log.info(f"  [Coupled]      ep {ep}/{args.epochs}  val-rel-L2={vl2:.4f}")
        return best

    log.info("\n=== Training Independent FNOs ===")
    ind_l2 = train_independent()

    log.info("\n=== Training CoupledOperator ===")
    coup_l2 = train_coupled()

    log.info("\n" + "=" * 50)
    log.info(f"Independent operators  best val rel-L2: {ind_l2:.4f}")
    log.info(f"CoupledOperator        best val rel-L2: {coup_l2:.4f}")
    improvement = (ind_l2 - coup_l2) / ind_l2 * 100
    log.info(f"Relative improvement from coupling: {improvement:+.1f}%")

    # Show learned coupling strengths
    W = coupled.coupling_strengths()
    if W is not None:
        log.info("\nLearned coupling strengths W[i→j]:")
        names = ["thermal", "flow"]
        for i, n_i in enumerate(names):
            for j, n_j in enumerate(names):
                if i != j:
                    log.info(f"  {n_j} → {n_i}: {W[i, j].item():.4f}")
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
    run(p.parse_args())
