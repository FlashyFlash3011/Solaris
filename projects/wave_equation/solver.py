# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Finite-difference solver for the 2-D scalar wave equation.

    ∂²u/∂t² = c² ∇²u

Rewritten as a first-order system:
    ∂u/∂t = v
    ∂v/∂t = c² ∇²u

with Dirichlet BCs: u = 0 on ∂Ω.
"""

from typing import Optional, Tuple

import numpy as np
import torch


def _laplacian_fd(u: np.ndarray, dx: float) -> np.ndarray:
    """5-point Laplacian with zero-padding (Dirichlet BCs enforced externally)."""
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
    ) / dx ** 2
    return lap


def solve_wave(
    u0: np.ndarray,
    v0: np.ndarray,
    c: float = 1.0,
    dx: float = 1.0 / 63,
    dt: float = 1e-3,
    n_steps: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the wave equation via leapfrog (Verlet) scheme.

    Parameters
    ----------
    u0 : ndarray  shape (H, W)
        Initial displacement field.
    v0 : ndarray  shape (H, W)
        Initial velocity field.
    c : float
        Wave speed.
    dx : float
        Spatial grid spacing (assumed uniform).
    dt : float
        Time step (must satisfy CFL: c*dt/dx < 1/√2).
    n_steps : int
        Number of time steps.

    Returns
    -------
    u : ndarray  shape (H, W)  — final displacement
    v : ndarray  shape (H, W)  — final velocity
    """
    u = u0.copy().astype(np.float64)
    v = v0.copy().astype(np.float64)
    H, W = u.shape

    cfl = c * dt / dx
    if cfl > 1.0 / np.sqrt(2):
        import warnings
        warnings.warn(f"CFL number {cfl:.3f} > 1/√2 — solution may be unstable.", stacklevel=2)

    for _ in range(n_steps):
        lap_u = _laplacian_fd(u, dx)
        v_new = v + dt * c ** 2 * lap_u
        u_new = u + dt * v_new
        # Dirichlet BCs
        u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
        v_new[0, :] = v_new[-1, :] = v_new[:, 0] = v_new[:, -1] = 0.0
        u, v = u_new, v_new

    return u.astype(np.float32), v.astype(np.float32)


def solve_wave_snapshots(
    u0: np.ndarray,
    v0: np.ndarray,
    c: float = 1.0,
    dx: float = 1.0 / 63,
    dt: float = 1e-3,
    n_steps: int = 200,
    n_snapshots: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like ``solve_wave`` but returns evenly spaced snapshots of (u, v).

    Returns
    -------
    u_snaps : ndarray  shape (n_snapshots, H, W)
    v_snaps : ndarray  shape (n_snapshots, H, W)
    """
    u = u0.copy().astype(np.float64)
    v = v0.copy().astype(np.float64)

    save_every = max(1, n_steps // (n_snapshots - 1))
    u_snaps, v_snaps = [u.astype(np.float32)], [v.astype(np.float32)]

    for step in range(n_steps):
        lap_u = _laplacian_fd(u, dx)
        v_new = v + dt * c ** 2 * lap_u
        u_new = u + dt * v_new
        u_new[0, :] = u_new[-1, :] = u_new[:, 0] = u_new[:, -1] = 0.0
        v_new[0, :] = v_new[-1, :] = v_new[:, 0] = v_new[:, -1] = 0.0
        u, v = u_new, v_new

        if (step + 1) % save_every == 0 and len(u_snaps) < n_snapshots:
            u_snaps.append(u.astype(np.float32))
            v_snaps.append(v.astype(np.float32))

    while len(u_snaps) < n_snapshots:
        u_snaps.append(u_snaps[-1])
        v_snaps.append(v_snaps[-1])

    return np.stack(u_snaps[:n_snapshots]), np.stack(v_snaps[:n_snapshots])


def random_gaussian_ic(
    H: int = 64,
    W: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random Gaussian-blob initial displacement, zero initial velocity.

    Returns
    -------
    u0, v0 : ndarray  shape (H, W)
    """
    if rng is None:
        rng = np.random.default_rng()

    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    XX, YY = np.meshgrid(xs, ys)

    n_blobs = rng.integers(1, 4)
    u0 = np.zeros((H, W))
    for _ in range(n_blobs):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        sigma = rng.uniform(0.05, 0.15)
        amp = rng.uniform(-1.0, 1.0)
        u0 += amp * np.exp(-((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * sigma ** 2))

    # Enforce Dirichlet BCs on IC
    u0[0, :] = u0[-1, :] = u0[:, 0] = u0[:, -1] = 0.0
    v0 = np.zeros_like(u0)
    return u0.astype(np.float32), v0.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    u0, v0 = random_gaussian_ic(64, 64, rng=rng)
    u_snaps, _ = solve_wave_snapshots(u0, v0, c=1.0, dt=5e-4, n_steps=400, n_snapshots=6)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i, ax in enumerate(axes):
        im = ax.imshow(u_snaps[i], cmap="seismic", origin="lower",
                       vmin=-u_snaps.max(), vmax=u_snaps.max())
        ax.set_title(f"t={i}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("wave_snapshots.png", dpi=120)
    print("Saved wave_snapshots.png")
