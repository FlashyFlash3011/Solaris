# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Finite-difference solver for 2-D steady-state heat conduction.

    -k ∇²T = Q(x,y)    on Ω = [0,1]²
    T = T_ambient       on ∂Ω  (heat-sink / package boundary)

Models a silicon die seen from above: heat flows laterally to the package
boundary (Dirichlet T=40 °C).  Power density Q [W/m²] is normalised so
that a fully-loaded core produces a peak junction temperature of ~95 °C,
consistent with modern processor thermal design points.
"""

import time
from typing import Optional, Tuple

import numpy as np


# Realistic chip thermal reference point
T_JUNCTION_MAX = 95.0    # °C  peak junction temperature (thermal design point)
T_AMBIENT      = 40.0    # °C  package / heat-sink boundary temperature


def solve_heat(
    power_map: np.ndarray,
    k: float = 1.0,
    T_ambient: float = T_AMBIENT,
    max_iter: int = 50_000,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, int, float]:
    """Solve steady-state 2-D heat equation via Gauss-Seidel iteration.

    Parameters
    ----------
    power_map : ndarray, shape (H, W)
        Heat source density [W/m²] at each grid cell.
    k : float
        Effective thermal conductivity (normalised units).
    T_ambient : float
        Dirichlet boundary temperature [°C].
    max_iter : int
        Maximum Gauss-Seidel sweeps.
    tol : float
        Convergence tolerance.

    Returns
    -------
    T         : ndarray, shape (H, W)  [°C]
    iters     : int
    wall_time : float  [s]
    """
    H, W = power_map.shape
    h = 1.0 / (H - 1)
    rhs = (h ** 2 / k) * power_map

    T = np.full((H, W), T_ambient, dtype=np.float64)

    t0 = time.perf_counter()
    for it in range(max_iter):
        T_old = T.copy()
        T[1:-1, 1:-1] = 0.25 * (
            T[:-2, 1:-1] + T[2:, 1:-1] +
            T[1:-1, :-2] + T[1:-1, 2:] +
            rhs[1:-1, 1:-1]
        )
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_ambient
        if np.max(np.abs(T - T_old)) < tol:
            break
    elapsed = time.perf_counter() - t0
    return T, it + 1, elapsed


def chip_floorplan_power_map(
    H: int = 128,
    W: int = 128,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a power map from a realistic quad-core processor floorplan.

    Each call draws a random workload scenario — cores may be idle, partially
    loaded, or fully active; memory bandwidth and cache activity vary
    independently.  Power values are calibrated so a fully-loaded chip reaches
    ~95 °C junction temperature (the industry thermal design point).

    Die layout (normalised [0, 1]²):

        ┌──────────────────────────────────────┐
        │  I/O pad ring  (perimeter)           │
        │  ┌────────┬────────┬───────────┐┌──┐│
        │  │  Core 0│  Core 1│           ││  ││
        │  │  +L2$  │  +L2$  │  L3 Cache ││MC││
        │  ├────────┼────────┤   (LLC)   ││ 0││
        │  │  Core 2│  Core 3│           │├──┤│
        │  │  +L2$  │  +L2$  │           ││  ││
        │  └────────┴────────┴───────────┘│MC││
        │                                 │ 1││
        │                                 └──┘│
        └──────────────────────────────────────┘

    Parameters
    ----------
    H, W : int
        Grid resolution.
    rng : numpy Generator, optional

    Returns
    -------
    power_map : ndarray, shape (H, W)  [W/m²  normalised]
    """
    if rng is None:
        rng = np.random.default_rng()

    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    XX, YY = np.meshgrid(xs, ys)
    Q = np.zeros((H, W), dtype=np.float64)

    def fill(x0, x1, y0, y1, power):
        mask = (XX >= x0) & (XX < x1) & (YY >= y0) & (YY < y1)
        Q[mask] += power

    # Background leakage (subthreshold + gate leakage across whole die)
    Q += 120.0

    # I/O pad ring (bump bonds, ESD cells, perimeter SerDes)
    io = 0.05
    io_p = rng.uniform(200, 450)
    fill(0,    1,    0,    io,   io_p)
    fill(0,    1,  1-io,   1,   io_p)
    fill(0,   io,   io,  1-io,  io_p)
    fill(1-io,  1,   io,  1-io,  io_p)

    # 4 CPU cores — compute fabric (high power) wrapped by L2 cache (medium)
    # Per-core utilisation is independent (asymmetric workloads, e.g. 1T, 2T, AVX)
    core_util = rng.uniform(0.05, 1.0, size=4)
    pad = 0.04  # L2 shell thickness

    cores = [
        (0.06, 0.28, 0.52, 0.90),   # Core 0  top-left
        (0.29, 0.46, 0.52, 0.90),   # Core 1  top-right
        (0.06, 0.28, 0.10, 0.48),   # Core 2  bottom-left
        (0.29, 0.46, 0.10, 0.48),   # Core 3  bottom-right
    ]
    for i, (x0, x1, y0, y1) in enumerate(cores):
        u = core_util[i]
        fill(x0, x1, y0, y1,                          550 * u)   # L2 cache
        fill(x0+pad, x1-pad, y0+pad, y1-pad,         2200 * u)   # compute fabric

    # Shared L3 cache / uncore (right of core cluster)
    l3_util = rng.uniform(0.10, 0.85)
    fill(0.48, 0.80, 0.10, 0.90, 480 * l3_util)

    # Dual memory controllers (far right, stacked)
    mc_util = rng.uniform(0.05, 0.95, size=2)
    fill(0.82, 0.94, 0.52, 0.90, 1100 * mc_util[0])
    fill(0.82, 0.94, 0.10, 0.48, 1100 * mc_util[1])

    return Q.clip(0).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    Q = chip_floorplan_power_map(128, 128, rng=rng)
    T, iters, elapsed = solve_heat(Q)

    print(f"Converged in {iters} iterations | {elapsed:.3f}s")
    print(f"Q range: {Q.min():.1f} – {Q.max():.1f} W/m²")
    print(f"T range: {T.min():.1f} – {T.max():.1f} °C")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(Q, cmap="hot", origin="lower", interpolation="bilinear")
    axes[0].set_title("Chip power map  [W/m²]", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(T, cmap="inferno", origin="lower", interpolation="bilinear")
    axes[1].set_title(f"Temperature  [°C]   ({elapsed:.2f}s)", fontsize=11)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle("Chip Thermal — Steady-State FD Solver", fontweight="bold")
    plt.tight_layout()
    plt.savefig("solver_demo.png", dpi=140)
    print("Saved solver_demo.png")
