# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Pseudo-spectral solver for 2-D incompressible Navier-Stokes in the
vorticity-streamfunction formulation.

    ∂ω/∂t + u·∇ω = ν ∇²ω
    ∇²ψ = −ω          (Poisson equation for streamfunction)
    u = ∂ψ/∂y,  v = −∂ψ/∂x

The solver uses:
  - Spectral Poisson solver for ψ via torch.fft.rfft2
  - RK4 time integration
  - Periodic boundary conditions
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _wavenumbers(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.fft.fftfreq(H, d=1.0 / H, device=device).reshape(H, 1)
    ky = torch.fft.rfftfreq(W, d=1.0 / W, device=device).reshape(1, W // 2 + 1)
    return kx, ky


def _poisson_solve(omega_hat: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Solve ∇²ψ = −ω in Fourier space."""
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.0  # avoid division by zero at DC (set ψ_DC = 0)
    psi_hat = omega_hat / k2
    psi_hat[..., 0, 0] = 0.0  # zero mean streamfunction
    return psi_hat


def _rhs(omega: torch.Tensor, nu: float, kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """Evaluate ∂ω/∂t = −u·∇ω + ν∇²ω."""
    H, W = omega.shape
    omega_hat = torch.fft.rfft2(omega)

    psi_hat = _poisson_solve(omega_hat, kx, ky)

    # Velocity field in physical space
    u = torch.fft.irfft2(1j * ky * psi_hat, s=(H, W))   # u = ∂ψ/∂y
    v = torch.fft.irfft2(-1j * kx * psi_hat, s=(H, W))  # v = −∂ψ/∂x

    # Vorticity gradients
    dw_dx = torch.fft.irfft2(1j * kx * omega_hat, s=(H, W))
    dw_dy = torch.fft.irfft2(1j * ky * omega_hat, s=(H, W))

    # Diffusion
    k2 = kx ** 2 + ky ** 2
    diff = torch.fft.irfft2(-nu * k2 * omega_hat, s=(H, W))

    return -u * dw_dx - v * dw_dy + diff


def solve_ns(
    omega0: torch.Tensor,
    nu: float = 1e-3,
    dt: float = 0.01,
    n_steps: int = 100,
    n_snapshots: int = 10,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Integrate 2-D vorticity equation via RK4.

    Parameters
    ----------
    omega0 : Tensor  shape (H, W)
        Initial vorticity field.
    nu : float
        Kinematic viscosity.
    dt : float
        Time step.
    n_steps : int
        Total integration steps.
    n_snapshots : int
        Number of snapshots to save (including t=0).
    device : torch.device, optional

    Returns
    -------
    Tensor  shape (n_snapshots, H, W)
        Vorticity snapshots at evenly spaced intervals.
    """
    if device is None:
        device = omega0.device
    omega = omega0.to(device=device, dtype=torch.float64)
    H, W = omega.shape
    kx, ky = _wavenumbers(H, W, device)

    save_every = max(1, n_steps // (n_snapshots - 1))
    snapshots = [omega.float().clone()]

    for step in range(n_steps):
        k1 = _rhs(omega, nu, kx, ky)
        k2 = _rhs(omega + 0.5 * dt * k1, nu, kx, ky)
        k3 = _rhs(omega + 0.5 * dt * k2, nu, kx, ky)
        k4 = _rhs(omega + dt * k3, nu, kx, ky)
        omega = omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if (step + 1) % save_every == 0 and len(snapshots) < n_snapshots:
            snapshots.append(omega.float().clone())

    # Ensure we have exactly n_snapshots
    while len(snapshots) < n_snapshots:
        snapshots.append(snapshots[-1])

    return torch.stack(snapshots[:n_snapshots])  # (n_snapshots, H, W)


def random_vorticity_ic(
    H: int = 64,
    W: int = 64,
    n_modes: int = 8,
    rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a random initial vorticity field from low-frequency Fourier modes.

    Parameters
    ----------
    H, W : int
    n_modes : int
        Number of Fourier modes per direction to include.
    rng : numpy Generator, optional

    Returns
    -------
    Tensor  shape (H, W)
    """
    if rng is None:
        rng = np.random.default_rng()
    if device is None:
        device = torch.device("cpu")

    omega = np.zeros((H, W))
    for kx in range(1, n_modes + 1):
        for ky in range(1, n_modes + 1):
            amp = rng.standard_normal() / (kx ** 2 + ky ** 2) ** 0.5
            phi = rng.uniform(0, 2 * np.pi)
            xs = np.linspace(0, 2 * np.pi, W, endpoint=False)
            ys = np.linspace(0, 2 * np.pi, H, endpoint=False)
            XX, YY = np.meshgrid(xs, ys)
            omega += amp * np.sin(kx * XX + ky * YY + phi)

    return torch.tensor(omega, dtype=torch.float32, device=device)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    omega0 = random_vorticity_ic(64, 64, n_modes=6, rng=rng)
    snaps = solve_ns(omega0, nu=1e-3, dt=0.01, n_steps=50, n_snapshots=6)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i, ax in enumerate(axes):
        im = ax.imshow(snaps[i].numpy(), cmap="RdBu_r", origin="lower")
        ax.set_title(f"t={i}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig("ns_vorticity.png", dpi=120)
    print("Saved ns_vorticity.png")
