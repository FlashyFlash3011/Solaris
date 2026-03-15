#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""Example: Solve 2-D Poisson equation with a Physics-Informed Neural Network.

    -∇²u = f(x,y)  on Ω = [0,1]²
     u = 0          on ∂Ω

Run:
    python examples/train_pinn_poisson.py
    python examples/train_pinn_poisson.py --device cuda   # ROCm GPU
"""

import argparse
import torch
import torch.nn as nn

from physicsnemo.models.mlp import FullyConnected
from physicsnemo.utils import get_logger


def f(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RHS forcing: f = 2π² sin(πx)sin(πy) so u = sin(πx)sin(πy)."""
    return 2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def exact_u(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def pde_residual(model: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    """Compute -∇²u - f at collocation points."""
    xy = xy.requires_grad_(True)
    u = model(xy)
    grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(grad_u[:, 0].sum(), xy, create_graph=True)[0][:, 0]
    u_yy = torch.autograd.grad(grad_u[:, 1].sum(), xy, create_graph=True)[0][:, 1]
    x, y = xy[:, 0], xy[:, 1]
    return -(u_xx + u_yy) - f(x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_collocation", type=int, default=2000)
    parser.add_argument("--n_boundary", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    log = get_logger("pinn_poisson")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    log.info(f"Device: {device}")

    model = FullyConnected(in_features=2, out_features=1, hidden_features=64, n_layers=5, activation="tanh")
    model = model.to(device)
    log.info(f"Model parameters: {model.num_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # Interior collocation points
        xy_int = torch.rand(args.n_collocation, 2, device=device)
        res = pde_residual(model, xy_int)
        loss_pde = (res**2).mean()

        # Boundary (u=0 on all four edges)
        t = torch.rand(args.n_boundary, 1, device=device)
        zeros = torch.zeros_like(t)
        ones = torch.ones_like(t)
        xy_b = torch.cat([
            torch.cat([t, zeros], dim=1),  # bottom
            torch.cat([t, ones], dim=1),   # top
            torch.cat([zeros, t], dim=1),  # left
            torch.cat([ones, t], dim=1),   # right
        ])
        loss_bc = (model(xy_b) ** 2).mean()

        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0:
            # L2 error on a grid
            with torch.no_grad():
                g = torch.linspace(0, 1, 50, device=device)
                xx, yy = torch.meshgrid(g, g, indexing="ij")
                xy_test = torch.stack([xx.flatten(), yy.flatten()], dim=1)
                u_pred = model(xy_test).squeeze(-1)
                u_ref = exact_u(xy_test[:, 0], xy_test[:, 1])
                l2_err = ((u_pred - u_ref)**2).mean().sqrt().item()
            log.info(f"Epoch {epoch:5d} | PDE loss={loss_pde.item():.3e} | BC loss={loss_bc.item():.3e} | L2 error={l2_err:.4f}")

    log.info("Done.")


if __name__ == "__main__":
    main()
