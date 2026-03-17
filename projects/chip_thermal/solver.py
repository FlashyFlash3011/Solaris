# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Finite-difference solvers for chip heat conduction.

2-D steady-state (legacy):
    -k ∇²T = Q(x,y)    on Ω = [0,1]²
    T = T_ambient       on ∂Ω

3-D transient (new):
    ρCp ∂T/∂t = k ∇²T + Q(x,y,z)
    Domain: 1 mm × 1 mm × 0.5 mm,  Grid: 32 × 32 × 16
    IC:  T = 25 °C everywhere
    BC:  Dirichlet T = 25 °C on bottom (z=0) and all four lateral faces;
         Neumann ∂T/∂z = 0 (adiabatic) on top (z = Nz-1)
"""

import time
from typing import Optional, Tuple

import numpy as np


# ── 2-D steady-state ──────────────────────────────────────────────────────────

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


# Alias for code that prefers the explicit name
solve_heat_2d = solve_heat


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


# Alias
random_power_map_2d = random_power_map


# ── 3-D transient ─────────────────────────────────────────────────────────────

# Silicon thermal properties
ALPHA_SILICON = 8.8e-5   # m²/s  thermal diffusivity
RHO_CP_SILICON = 1.63e6  # J/(m³·K)  volumetric heat capacity (k/α = 148/8.8e-5)
T_AMBIENT_3D = 25.0      # °C  IC + Dirichlet BC

# Domain
LX = 1e-3   # 1 mm
LY = 1e-3   # 1 mm
LZ = 5e-4   # 0.5 mm


def random_power_map_3d(
    Nx: int = 32,
    Ny: int = 32,
    Nz: int = 16,
    n_hotspots: int = 5,
    max_power: float = 1e9,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a 3-D chip power map with Gaussian hot-spots.

    Hot-spots are positioned mostly in the lower half of the die (near the
    active layer), with random amplitude, centre, and width.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Grid dimensions.
    n_hotspots : int
        Number of Gaussian heat sources.
    max_power : float
        Peak power density [W/m³].
    rng : numpy Generator, optional

    Returns
    -------
    power_map : ndarray, shape (Nx, Ny, Nz)  [W/m³]
    """
    if rng is None:
        rng = np.random.default_rng()
    xs = np.linspace(0, 1, Nx)
    ys = np.linspace(0, 1, Ny)
    zs = np.linspace(0, 1, Nz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    Q = np.zeros((Nx, Ny, Nz))
    for _ in range(n_hotspots):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        cz = rng.uniform(0.0, 0.6)   # hot-spots in lower layers (active silicon)
        sx = rng.uniform(0.05, 0.20)
        sy = rng.uniform(0.05, 0.20)
        sz = rng.uniform(0.10, 0.40)
        amp = rng.uniform(0.2, 1.0) * max_power
        Q += amp * np.exp(
            -(  (XX - cx)**2 / (2 * sx**2)
              + (YY - cy)**2 / (2 * sy**2)
              + (ZZ - cz)**2 / (2 * sz**2))
        )
    return Q.astype(np.float32)


def solve_heat_3d(
    power_map: np.ndarray,
    t_end: float = 0.01,
    n_snapshots: int = 8,
    alpha: float = ALPHA_SILICON,
    Lx: float = LX,
    Ly: float = LY,
    Lz: float = LZ,
    T_ambient: float = T_AMBIENT_3D,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Explicit finite-difference solver for 3-D transient heat conduction.

    Solves:
        ρCp ∂T/∂t = k ∇²T + Q(x,y,z)

    with CFL-safe timestep  dt = 0.4 · min(dx,dy,dz)² / (6α).

    Boundary conditions
    -------------------
    - Dirichlet T = T_ambient: z = 0 (bottom / heat sink) and all four
      lateral faces (x = 0/Nx-1, y = 0/Ny-1).
    - Neumann ∂T/∂z = 0 (adiabatic) on top face z = Nz-1.

    Parameters
    ----------
    power_map : ndarray, shape (Nx, Ny, Nz)  [W/m³]
    t_end : float      simulation end time [s]  (default 10 ms)
    n_snapshots : int  number of output frames, evenly spaced in (0, t_end]
    alpha : float      thermal diffusivity [m²/s]
    Lx, Ly, Lz : float  physical domain size [m]
    T_ambient : float  initial and boundary temperature [°C]

    Returns
    -------
    snapshots : ndarray, shape (n_snapshots, Nx, Ny, Nz)  [°C]
    times     : ndarray, shape (n_snapshots,)              [s]
    wall_time : float                                       [s]
    """
    Nx, Ny, Nz = power_map.shape
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    h_min = min(dx, dy, dz)
    dt = 0.4 * h_min**2 / (6.0 * alpha)
    n_steps = max(1, int(np.ceil(t_end / dt)))
    dt = t_end / n_steps   # adjust so we land exactly on t_end

    rx = alpha * dt / dx**2
    ry = alpha * dt / dy**2
    rz = alpha * dt / dz**2
    assert rx + ry + rz <= 0.5 + 1e-9, (
        f"CFL violated: rx+ry+rz = {rx+ry+rz:.4f}  (must be ≤ 0.5)"
    )

    # Source term  ∂T/∂t += Q / (ρCp)
    Q_src = power_map.astype(np.float64) / RHO_CP_SILICON   # [K/s]

    T = np.full((Nx, Ny, Nz), T_ambient, dtype=np.float64)

    # Map: simulation step → snapshot index
    snap_at: dict = {}
    for k in range(n_snapshots):
        s = min(int(round((k + 1) * n_steps / n_snapshots)), n_steps)
        snap_at[s] = k   # last writer wins on ties (negligible impact)

    snapshots: list = [None] * n_snapshots
    times = np.linspace(t_end / n_snapshots, t_end, n_snapshots)

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        # Interior Laplacian (shape: Nx-2, Ny-2, Nz-2)
        dT = (
            rx * (T[2:,  1:-1, 1:-1] + T[:-2, 1:-1, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1]) +
            ry * (T[1:-1, 2:,  1:-1] + T[1:-1, :-2, 1:-1] - 2 * T[1:-1, 1:-1, 1:-1]) +
            rz * (T[1:-1, 1:-1, 2:]  + T[1:-1, 1:-1, :-2] - 2 * T[1:-1, 1:-1, 1:-1]) +
            Q_src[1:-1, 1:-1, 1:-1] * dt
        )
        T[1:-1, 1:-1, 1:-1] += dT

        # Neumann on top face  (∂T/∂z = 0 → ghost cell = adjacent interior)
        T[:, :, -1] = T[:, :, -2]

        # Dirichlet on bottom + lateral faces
        T[0, :, :]  = T[-1, :, :]  = T_ambient
        T[:, 0, :]  = T[:, -1, :]  = T_ambient
        T[:, :,  0] = T_ambient

        if step in snap_at:
            k = snap_at[step]
            snapshots[k] = T.copy().astype(np.float32)

    wall_time = time.perf_counter() - t0

    # Safety: fill any un-captured slots with final state
    for k in range(n_snapshots):
        if snapshots[k] is None:
            snapshots[k] = T.copy().astype(np.float32)

    return np.stack(snapshots), times, wall_time


# ── Smoke-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== 2-D steady-state smoke test ===")
    rng = np.random.default_rng(42)
    Q2 = random_power_map(64, 64, rng=rng)
    T2, iters, elapsed = solve_heat(Q2, max_iter=20_000, tol=1e-5)
    print(f"Converged in {iters} iterations | {elapsed:.3f}s")
    print(f"T range: {T2.min():.1f} — {T2.max():.1f} °C")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(Q2, cmap="hot", origin="lower")
    axes[0].set_title("Power map Q [W/m²]")
    plt.colorbar(axes[0].images[0], ax=axes[0])
    axes[1].imshow(T2, cmap="inferno", origin="lower")
    axes[1].set_title("Temperature T [°C]")
    plt.colorbar(axes[1].images[0], ax=axes[1])
    plt.tight_layout()
    plt.savefig("solver_demo.png", dpi=120)
    print("Saved solver_demo.png")

    print("\n=== 3-D transient smoke test (32×32×16, 10 ms) ===")
    rng3 = np.random.default_rng(7)
    Q3 = random_power_map_3d(32, 32, 16, rng=rng3)
    print(f"Power map: min={Q3.min():.2e}  max={Q3.max():.2e} W/m³")
    snaps, times, wall = solve_heat_3d(Q3, t_end=0.01, n_snapshots=8)
    print(f"Solved {snaps.shape[0]} snapshots in {wall:.2f}s")
    print(f"T at t={times[0]*1e3:.1f}ms : {snaps[0].min():.2f}–{snaps[0].max():.2f} °C")
    print(f"T at t={times[-1]*1e3:.1f}ms: {snaps[-1].min():.2f}–{snaps[-1].max():.2f} °C")

    # Quick slice plot at final time
    mid_z = snaps.shape[-1] // 2
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    axes2[0].imshow(Q3[:, :, mid_z], cmap="hot", origin="lower")
    axes2[0].set_title(f"Q slice at z={mid_z} [W/m³]")
    plt.colorbar(axes2[0].images[0], ax=axes2[0])
    axes2[1].imshow(snaps[-1, :, :, mid_z], cmap="inferno", origin="lower")
    axes2[1].set_title(f"T at t={times[-1]*1e3:.1f}ms, z={mid_z} [°C]")
    plt.colorbar(axes2[1].images[0], ax=axes2[1])
    plt.tight_layout()
    plt.savefig("solver_3d_demo.png", dpi=120)
    print("Saved solver_3d_demo.png")
