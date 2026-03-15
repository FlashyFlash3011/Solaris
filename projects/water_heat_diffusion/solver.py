# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Finite-difference solver for 2-D heat diffusion in water.

    ∂T/∂t = α ∇²T        (heat equation)

    α = 1.43e-7 m²/s      thermal diffusivity of water
    Domain: 10cm × 10cm square
    BC: fixed ambient temperature on all edges

The solver steps forward in tiny increments (required for stability).
This is the "traditional way" — correct but slow.
"""

import time
from typing import List, Tuple

import numpy as np

# Water thermal diffusivity [m²/s]
ALPHA_WATER = 1.43e-7

# Simulation domain: 1mm × 1mm  (microfluidics / water droplet scale)
# At this scale the CFL timestep is ~1.7e-4 s, giving thousands of steps
# for a sub-second simulation — exactly what makes the solver slow.
DOMAIN_SIZE = 1e-3  # metres


def stable_dt(dx: float, alpha: float, safety: float = 0.4) -> float:
    """CFL-stable timestep for explicit finite differences."""
    return safety * dx**2 / (4 * alpha)


def solve_diffusion(
    T0: np.ndarray,
    t_end: float,
    alpha: float = ALPHA_WATER,
    n_snapshots: int = 8,
) -> Tuple[np.ndarray, List[float], float, int]:
    """Simulate heat diffusion from initial field T0 up to time t_end.

    Parameters
    ----------
    T0 : ndarray (H, W)     initial temperature field [°C]
    t_end : float           total simulation time [s]
    alpha : float           thermal diffusivity [m²/s]
    n_snapshots : int       how many evenly-spaced frames to return

    Returns
    -------
    snapshots : ndarray (n_snapshots, H, W)
    snapshot_times : list[float]
    wall_time : float       seconds of compute
    n_steps : int           number of time steps taken
    """
    H, W = T0.shape
    dx = DOMAIN_SIZE / (W - 1)
    dt = stable_dt(dx, alpha)
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps           # adjust so we land exactly on t_end

    r = alpha * dt / dx**2         # diffusion number (≤ 0.25 for stability)

    T = T0.copy().astype(np.float64)
    T_amb = T[0, 0]                # boundary value = initial edge temp

    snap_indices = set(int(round(i * (n_steps - 1) / (n_snapshots - 1)))
                       for i in range(n_snapshots))
    snapshots, snap_times = [], []

    t0_wall = time.perf_counter()
    for step in range(n_steps):
        if step in snap_indices:
            snapshots.append(T.copy().astype(np.float32))
            snap_times.append(step * dt)

        lap = (
            T[:-2, 1:-1] + T[2:, 1:-1] +
            T[1:-1, :-2] + T[1:-1, 2:] -
            4 * T[1:-1, 1:-1]
        )
        T[1:-1, 1:-1] += r * lap
        T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_amb

    if len(snapshots) < n_snapshots:
        snapshots.append(T.copy().astype(np.float32))
        snap_times.append(t_end)

    wall_time = time.perf_counter() - t0_wall
    return np.stack(snapshots[:n_snapshots]), snap_times[:n_snapshots], wall_time, n_steps


def make_initial_field(
    H: int = 64,
    W: int = 64,
    T_ambient: float = 20.0,
    n_hotspots: int = 3,
    T_hot: float = 80.0,
    spot_radius: float = 0.06,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Create an initial temperature field with Gaussian hot-spots.

    Represents e.g. a syringe of hot water injected into a cool tank.

    Parameters
    ----------
    H, W         grid resolution
    T_ambient    background temperature [°C]
    n_hotspots   number of heat sources
    T_hot        peak temperature at hot-spot centres [°C]
    spot_radius  Gaussian sigma as fraction of domain size
    """
    if rng is None:
        rng = np.random.default_rng()
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    XX, YY = np.meshgrid(xs, ys)
    T = np.full((H, W), T_ambient, dtype=np.float32)
    for _ in range(n_hotspots):
        cx = rng.uniform(0.2, 0.8)
        cy = rng.uniform(0.2, 0.8)
        sigma = rng.uniform(spot_radius * 0.5, spot_radius * 1.5)
        amp = rng.uniform(0.5, 1.0) * (T_hot - T_ambient)
        T += amp * np.exp(-((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2))
    # Fix boundary to ambient
    T[0, :] = T[-1, :] = T[:, 0] = T[:, -1] = T_ambient
    return T


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    T0 = make_initial_field(128, 128, rng=rng)
    snaps, times, wall, steps = solve_diffusion(T0, t_end=0.5, n_snapshots=6)
    print(f"Simulated {steps:,} steps in {wall:.2f}s")
    print(f"Snapshot times: {[f'{t:.1f}s' for t in times]}")
    print(f"T range at t=0:    {snaps[0].min():.1f} – {snaps[0].max():.1f} °C")
    print(f"T range at t_end:  {snaps[-1].min():.1f} – {snaps[-1].max():.1f} °C")
