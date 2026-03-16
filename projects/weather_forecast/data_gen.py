# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Synthetic atmospheric field generator.

Produces physically-plausible 2-D weather fields by superimposing:
  - Large-scale wave patterns (Rossby-wave-like)
  - Jet stream structure (mid-latitude band)
  - Travelling pressure systems (cyclones / anticyclones)
  - Small-scale turbulent perturbations

Two prognostic variables are simulated:
  z500  — 500 hPa geopotential height [m]   (tracks pressure systems)
  t850  — 850 hPa temperature          [°C] (tracks warm/cold fronts)

Time evolution uses a simple barotropic advection model:
  ∂z/∂t = -U·∇z + diffusion
where U is a prescribed background wind derived from the jet stream.
This is NOT real NWP — it is a cheap toy model that produces fields
that look like weather maps so the demo is visually meaningful.
"""

import time
from typing import Tuple

import numpy as np


# ── Grid ─────────────────────────────────────────────────────────────────────

def make_grid(nlat: int = 64, nlon: int = 128):
    """Return (lat, lon) arrays in degrees and radian coordinate grids."""
    lat = np.linspace(-90, 90, nlat)      # degrees
    lon = np.linspace(0, 360, nlon, endpoint=False)
    LON, LAT = np.meshgrid(lon, lat)
    phi = np.deg2rad(LAT)   # latitude in radians
    lam = np.deg2rad(LON)   # longitude in radians
    return lat, lon, phi, lam


# ── Initial conditions ────────────────────────────────────────────────────────

def make_initial_state(
    nlat: int = 64,
    nlon: int = 128,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic-looking initial (z500, t850) pair.

    Returns
    -------
    z500 : (nlat, nlon)  geopotential height [m],  typical range 5000–6000 m
    t850 : (nlat, nlon)  temperature         [°C], typical range -40 – +25 °C
    """
    if rng is None:
        rng = np.random.default_rng()

    _, _, phi, lam = make_grid(nlat, nlon)

    # ── z500: background + wave perturbations ────────────────────────────────
    # Background: decreases toward poles (cold air = lower heights)
    z_bg = 5500 + 400 * np.sin(phi)   # 5100 at S pole, 5900 at N pole

    # Planetary waves (wavenumber 4–6): the big meanders in the jet stream
    z500 = z_bg.copy()
    for wn in range(4, 7):
        amp   = rng.uniform(60, 150)
        phase = rng.uniform(0, 2 * np.pi)
        lat_c = rng.uniform(0.5, 0.9)   # fraction of phi range
        z500 += amp * np.cos(wn * lam + phase) * np.exp(-((phi - lat_c) ** 2) / 0.3)

    # Synoptic-scale systems (cyclones / anticyclones)
    n_systems = rng.integers(4, 9)
    for _ in range(n_systems):
        lat_c = rng.uniform(-0.8, 0.8)
        lon_c = rng.uniform(0, 2 * np.pi)
        sigma = rng.uniform(0.15, 0.35)
        amp   = rng.choice([-1, 1]) * rng.uniform(40, 120)
        dist2 = (phi - lat_c) ** 2 + ((lam - lon_c + np.pi) % (2 * np.pi) - np.pi) ** 2
        z500 += amp * np.exp(-dist2 / (2 * sigma ** 2))

    # ── t850: correlated with z500 but with its own frontal structure ─────────
    # Thermal wind: warm air → high z, cold air → low z (roughly)
    t850 = -40 + 65 * (np.sin(phi) + 1) / 2   # -40°C at pole, +25°C at equator

    # Add fronts correlated with z gradient
    gz_lat = np.gradient(z500, axis=0)
    t850 += 0.05 * gz_lat

    # Independent temperature perturbations
    for _ in range(rng.integers(3, 7)):
        lat_c = rng.uniform(-0.7, 0.7)
        lon_c = rng.uniform(0, 2 * np.pi)
        sigma = rng.uniform(0.1, 0.3)
        amp   = rng.uniform(-8, 8)
        dist2 = (phi - lat_c) ** 2 + ((lam - lon_c + np.pi) % (2 * np.pi) - np.pi) ** 2
        t850 += amp * np.exp(-dist2 / (2 * sigma ** 2))

    return z500.astype(np.float32), t850.astype(np.float32)


# ── Numerical time-stepping ───────────────────────────────────────────────────

def _laplacian_norm(field: np.ndarray) -> np.ndarray:
    """Normalised 5-point Laplacian (divided by dx²=1 in grid units).
    Periodic in longitude, reflective at latitude boundaries."""
    fp = np.pad(field, ((1, 1), (0, 0)), mode="reflect")
    return (
        np.roll(field,  1, axis=1) + np.roll(field, -1, axis=1) +
        fp[:-2, :] + fp[2:, :] - 4 * field
    )


def _advect(field: np.ndarray, u: np.ndarray, v: np.ndarray,
            dt: float, dx: float) -> np.ndarray:
    """First-order upwind advection: ∂f/∂t = -u·∂f/∂x - v·∂f/∂y.
    u, v in grid-cells/second (= m/s / dx_m).
    """
    dfx = np.where(u > 0,
                   field - np.roll(field, 1, axis=1),
                   np.roll(field, -1, axis=1) - field)
    fp  = np.pad(field, ((1, 1), (0, 0)), mode="reflect")
    dfy = np.where(v > 0, field - fp[:-2, :], fp[2:, :] - field)
    return field - dt * (u * dfx + v * dfy)


def step_model(
    z500: np.ndarray,
    t850: np.ndarray,
    phi: np.ndarray,
    lam: np.ndarray,
    dt_hours: float = 6.0,
    alpha: float = 0.05,   # diffusion coefficient in grid units (dimensionless)
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance the state by one timestep (dt_hours hours).

    Simplified barotropic advection model:
      - Jet-stream zonal wind advects both fields
      - Weak sponge-layer diffusion prevents blow-up
      - Units: wind in grid-cells/step, diffusion in grid-units²/step
    """
    nlon = z500.shape[1]

    # Jet-stream profile: peaks ~45°N/S (phi ≈ ±0.785 rad), max ~0.3 grid/step
    # (A real jet ~25 m/s; at ~300 km grid spacing that's ~0.3 grid cells/hour)
    U = 0.3 * dt_hours * (
        np.exp(-((phi - 0.785) ** 2) / 0.15) +
        np.exp(-((phi + 0.785) ** 2) / 0.15)
    )

    # Meridional wind: geostrophic-like, small fraction of zonal
    dz_dy = np.gradient(z500, axis=0)
    dz_std = np.std(dz_dy) + 1e-8
    V = -0.02 * dt_hours * dz_dy / dz_std   # normalised, small

    # Clamp winds to CFL < 0.8 for stability
    U = np.clip(U, -0.8, 0.8)
    V = np.clip(V, -0.8, 0.8)

    z500_new = _advect(z500, U,       V,       1.0, 1.0)
    t850_new = _advect(t850, U * 0.8, V * 0.6, 1.0, 1.0)

    # Diffusion: alpha must satisfy alpha ≤ 0.25 for stability
    a = min(alpha, 0.24)
    z500_new = z500_new + a * _laplacian_norm(z500_new)
    t850_new = t850_new + a * _laplacian_norm(t850_new)

    return z500_new.astype(np.float32), t850_new.astype(np.float32)


def simulate(
    z0: np.ndarray,
    t0: np.ndarray,
    n_days: float = 5.0,
    dt_hours: float = 6.0,
    n_snapshots: int = 6,
) -> Tuple[np.ndarray, np.ndarray, list, float, int]:
    """Run the toy NWP model forward for n_days.

    Returns
    -------
    z_snaps : (n_snapshots, H, W)
    t_snaps : (n_snapshots, H, W)
    snap_days : list[float]  — day of each snapshot
    wall_time : float
    n_steps : int
    """
    _, _, phi, lam = make_grid(*z0.shape)
    n_steps = int(n_days * 24 / dt_hours)
    snap_at = {int(round(i * (n_steps - 1) / (n_snapshots - 1))): i
               for i in range(n_snapshots)}

    z, t = z0.copy(), t0.copy()
    z_snaps = np.zeros((n_snapshots, *z0.shape), dtype=np.float32)
    t_snaps = np.zeros((n_snapshots, *t0.shape), dtype=np.float32)
    snap_days = [0.0] * n_snapshots

    t_wall = time.perf_counter()
    for step in range(n_steps):
        if step in snap_at:
            idx = snap_at[step]
            z_snaps[idx] = z
            t_snaps[idx] = t
            snap_days[idx] = step * dt_hours / 24

        z, t = step_model(z, t, phi, lam, dt_hours=dt_hours)

    wall = time.perf_counter() - t_wall
    return z_snaps, t_snaps, snap_days, wall, n_steps


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    z0, t0 = make_initial_state(64, 128, rng=rng)
    z_s, t_s, days, wall, steps = simulate(z0, t0, n_days=5, n_snapshots=6)
    print(f"Simulated {steps} steps ({5} days) in {wall:.2f}s")
    print(f"z500 range: {z_s[-1].min():.0f} – {z_s[-1].max():.0f} m")
    print(f"t850 range: {t_s[-1].min():.1f} – {t_s[-1].max():.1f} °C")
