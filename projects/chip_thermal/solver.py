# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Finite-difference solver for 2-D steady-state heat conduction.

    -k ∇²T = Q(x,y)    on Ω = [0,1]²
    T = T_ambient       on ∂Ω

This is the *baseline* — accurate but slow. We use it to generate training
data and as the ground-truth reference in the comparison.
"""

import time
from typing import Optional, Tuple

import numpy as np


def solve_heat(
    power_map: np.ndarray,
    k: float = 1.0,
    T_ambient: float = 25.0,
    max_iter: int = 20_000,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, int, float]:
    """Solve steady-state 2-D heat equation via Gauss-Seidel iteration.

    Parameters
    ----------
    power_map : ndarray, shape (H, W)
        Heat source density [W/m²] at each grid cell.
    k : float
        Thermal conductivity.
    T_ambient : float
        Dirichlet boundary condition temperature.
    max_iter : int
        Maximum number of Gauss-Seidel sweeps.
    tol : float
        Convergence tolerance (max absolute change per sweep).

    Returns
    -------
    T : ndarray, shape (H, W)  — temperature field
    iters : int                — iterations taken
    wall_time : float          — seconds elapsed
    """
    H, W = power_map.shape
    h = 1.0 / (H - 1)          # grid spacing (square domain)
    rhs = (h ** 2 / k) * power_map

    # Initialise to ambient everywhere, boundaries stay fixed
    T = np.full((H, W), T_ambient, dtype=np.float64)

    t0 = time.perf_counter()
    for it in range(max_iter):
        T_old = T.copy()
        # Interior update (Gauss-Seidel)
        T[1:-1, 1:-1] = 0.25 * (
            T[:-2, 1:-1] + T[2:, 1:-1] +
            T[1:-1, :-2] + T[1:-1, 2:] +
            rhs[1:-1, 1:-1]
        )
        # Enforce boundaries
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_ambient
        if np.max(np.abs(T - T_old)) < tol:
            break
    elapsed = time.perf_counter() - t0
    return T, it + 1, elapsed


def random_power_map(
    H: int = 64,
    W: int = 64,
    n_hotspots: int = 6,
    max_power: float = 100.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic chip power map with Gaussian hot-spots.

    Parameters
    ----------
    H, W : int
        Grid resolution.
    n_hotspots : int
        Number of heat-generating regions (CPU cores, memory, etc.).
    max_power : float
        Peak power density [W/m²].
    rng : numpy Generator, optional
        For reproducibility.

    Returns
    -------
    power_map : ndarray, shape (H, W)
    """
    if rng is None:
        rng = np.random.default_rng()
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    XX, YY = np.meshgrid(xs, ys)
    Q = np.zeros((H, W))
    for _ in range(n_hotspots):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        sx = rng.uniform(0.03, 0.12)
        sy = rng.uniform(0.03, 0.12)
        amp = rng.uniform(0.2, 1.0) * max_power
        Q += amp * np.exp(-((XX - cx)**2 / (2*sx**2) + (YY - cy)**2 / (2*sy**2)))
    return Q.astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    Q = random_power_map(64, 64, rng=rng)
    T, iters, elapsed = solve_heat(Q, max_iter=20_000, tol=1e-5)

    print(f"Converged in {iters} iterations | {elapsed:.3f}s")
    print(f"T range: {T.min():.1f} — {T.max():.1f} °C")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(Q, cmap="hot", origin="lower")
    axes[0].set_title("Power map Q [W/m²]")
    plt.colorbar(axes[0].images[0], ax=axes[0])
    axes[1].imshow(T, cmap="inferno", origin="lower")
    axes[1].set_title("Temperature T [°C]")
    plt.colorbar(axes[1].images[0], ax=axes[1])
    plt.tight_layout()
    plt.savefig("solver_demo.png", dpi=120)
    print("Saved solver_demo.png")
