# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Shallow Water Equations (SWE) solver for hurricane storm surge.

Implements linearised 2-D SWE on a collocated grid:

    ∂η/∂t + H0·(∂u/∂x + ∂v/∂y) = 0          (mass conservation)
    ∂u/∂t = -g·∂η/∂x + τx/(ρ·H0) - r·u       (x-momentum + wind stress + friction)
    ∂v/∂t = -g·∂η/∂y + τy/(ρ·H0) - r·v       (y-momentum)

where η is the sea surface elevation anomaly (above rest), H0 is the
undisturbed water depth, and r is linear bottom friction [1/s].

Flooding:  a land cell floods when η > bathy (the surge exceeds land elevation).
           Flood depth = max(0, η - bathy) on land cells.

Time scheme:  Adams-Bashforth 2 (AB2) — 2nd-order in time, centered in space.
              A Shapiro 1-2-1 smoother is applied to η every N steps for stability.

Boundary conditions:
  South (open ocean): Sommerfeld radiation  ∂η/∂t + c ∂η/∂y = 0
  North/East/West:    no-flux  (u=0 or v=0 at walls)

Usage
-----
python solver.py          # runs a demo and saves results/solver_demo.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore

# Physical constants
G = 9.81          # gravitational acceleration [m/s²]
RHO_W = 1025.0    # seawater density [kg/m³]
RHO_A = 1.225     # air density [kg/m³]
C_D = 1.5e-3      # bulk drag coefficient (dimensionless)
R_FRICTION = 5e-5  # linear bottom friction [1/s]


# ---------------------------------------------------------------------------
# Bathymetry generator
# ---------------------------------------------------------------------------

def random_coastal_bathymetry(
    H: int = 64,
    W: int = 64,
    dx: float = 1000.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a synthetic coastal bathymetry field.

    Domain: south row = open ocean, north row = coast/land.

    Returns
    -------
    np.ndarray shape (H, W), dtype float32
        Elevation [m].  Positive = above sea level (land),
        negative = below (ocean).
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.linspace(0, 1, H)[:, None]  # 0=south(ocean), 1=north(coast)

    # Linear shelf: -30 m at south → +5 m at north
    bathy = -30.0 + 35.0 * y

    # Smooth random perturbations
    noise = np.zeros((H, W))
    for k in range(1, 5):
        amp = rng.uniform(1.0, 3.0)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        xs = np.linspace(0, 2 * np.pi * k, W)
        ys = np.linspace(0, 2 * np.pi * k, H)
        XX, YY = np.meshgrid(xs, ys)
        noise += amp * np.sin(XX + phase_x) * np.cos(YY + phase_y)
    noise = gaussian_filter(noise, sigma=3.0)
    bathy = (bathy + noise * 0.4).astype(np.float32)

    # Hard constraints: south strip = deep water, north strip = land
    bathy[:4, :] = np.clip(bathy[:4, :], -40.0, -5.0)
    bathy[-6:, :] = np.clip(bathy[-6:, :], 0.5, 8.0)
    return bathy


# ---------------------------------------------------------------------------
# Hurricane wind field
# ---------------------------------------------------------------------------

def gaussian_hurricane_wind(
    H: int,
    W: int,
    dx: float,
    t_hours: float,
    track: List[Tuple[float, float]],
    radius_km: float = 80.0,
    max_wind_ms: float = 55.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rankine vortex wind field interpolated along the storm track.

    Returns
    -------
    wind_u, wind_v : np.ndarray each shape (H, W) — eastward/northward [m/s]
    """
    # Interpolate storm centre
    t_clamp = float(np.clip(t_hours, 0, len(track) - 1))
    idx = int(t_clamp)
    frac = t_clamp - idx
    if idx + 1 < len(track):
        cx = (track[idx][0] + frac * (track[idx + 1][0] - track[idx][0])) * 1e3
        cy = (track[idx][1] + frac * (track[idx + 1][1] - track[idx][1])) * 1e3
    else:
        cx, cy = track[-1][0] * 1e3, track[-1][1] * 1e3

    xs = np.arange(W) * dx
    ys = np.arange(H) * dx
    XX, YY = np.meshgrid(xs, ys)

    dx_g = XX - cx
    dy_g = YY - cy
    r_km = np.sqrt(dx_g ** 2 + dy_g ** 2) / 1e3 + 1e-6

    # Rankine vortex profile (solid-body inside eye wall, power-law outside)
    r_max = radius_km
    v_tang = np.where(
        r_km <= r_max,
        max_wind_ms * (r_km / r_max),
        max_wind_ms * (r_max / r_km) ** 0.6,
    )

    # Anti-clockwise rotation + 20° inflow angle
    theta = np.arctan2(dy_g, dx_g)
    inflow = np.deg2rad(20.0)
    wind_u = -v_tang * np.sin(theta + inflow)
    wind_v =  v_tang * np.cos(theta + inflow)
    return wind_u.astype(np.float32), wind_v.astype(np.float32)


# ---------------------------------------------------------------------------
# Shallow Water Solver
# ---------------------------------------------------------------------------

class ShallowWaterSolver:
    """Adams-Bashforth 2 solver for linearised storm surge SWE.

    State variable η (sea-surface elevation anomaly) is stored separately
    from the background depth H0 (derived from bathymetry).  Flood depth
    on land is computed as max(0, η − bathy).

    Parameters
    ----------
    H, W : int
        Grid dimensions.
    dx : float
        Uniform grid spacing [m].
    dt : float
        Time step [s].  Stability: dt < dx / (sqrt(2) * sqrt(g * H0_max)).
    g : float
        Gravitational acceleration [m/s²].
    smooth_every : int
        Apply Shapiro 1-2-1 smoother to η every this many steps.
    """

    def __init__(
        self,
        H: int = 64,
        W: int = 64,
        dx: float = 1000.0,
        dt: float = 20.0,
        g: float = G,
        smooth_every: int = 10,
    ) -> None:
        self.H = H
        self.W = W
        self.dx = dx
        self.dt = dt
        self.g = g
        self.smooth_every = smooth_every
        self._step_count = 0

        # State (sea-surface elevation anomaly, velocities)
        self.eta = np.zeros((H, W), dtype=np.float64)
        self.u   = np.zeros((H, W), dtype=np.float64)
        self.v   = np.zeros((H, W), dtype=np.float64)
        # Previous tendencies for AB2
        self._deta_prev = np.zeros((H, W), dtype=np.float64)
        self._du_prev   = np.zeros((H, W), dtype=np.float64)
        self._dv_prev   = np.zeros((H, W), dtype=np.float64)

        self.bathy  = np.zeros((H, W), dtype=np.float32)
        self.H0     = np.ones((H, W), dtype=np.float64)   # background depth
        self.land_mask = np.zeros((H, W), dtype=bool)

    def set_bathymetry(self, bathy: np.ndarray) -> None:
        """Set bathymetry and derive background depth H0.

        Parameters
        ----------
        bathy : np.ndarray shape (H, W)
            Elevation [m].  Positive = land, negative = ocean.
        """
        self.bathy = bathy.astype(np.float32)
        # Background depth: undisturbed water depth (0 on land)
        self.H0 = np.maximum(0.5, -bathy.astype(np.float64))
        self.H0[bathy >= 0] = 0.5   # minimal depth over land for numerics
        self.land_mask = bathy >= 0.0

    def reset(self) -> None:
        """Reset all state to zero (calm ocean at rest)."""
        self.eta[:] = 0.0
        self.u[:]   = 0.0
        self.v[:]   = 0.0
        self._deta_prev[:] = 0.0
        self._du_prev[:]   = 0.0
        self._dv_prev[:]   = 0.0
        self._step_count = 0

    def step(
        self,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
        n_steps: int = 1,
    ) -> None:
        """Advance n_steps timesteps under the given wind field.

        Parameters
        ----------
        wind_u, wind_v : np.ndarray shape (H, W)
            10-m wind components [m/s].
        n_steps : int
        """
        wu = wind_u.astype(np.float64)
        wv = wind_v.astype(np.float64)

        # Wind stress via bulk formula
        wspd = np.sqrt(wu ** 2 + wv ** 2)
        tau_x = RHO_A * C_D * wspd * wu
        tau_y = RHO_A * C_D * wspd * wv

        for _ in range(n_steps):
            self._ab2_step(tau_x, tau_y)
            self._apply_boundaries()
            self._step_count += 1
            if self._step_count % self.smooth_every == 0:
                self._shapiro_smooth()

    def _central_diff_x(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂x via centered differences, shape (H, W)."""
        df = np.zeros_like(f)
        df[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * self.dx)
        df[:, 0]    = (f[:, 1] - f[:, 0]) / self.dx
        df[:, -1]   = (f[:, -1] - f[:, -2]) / self.dx
        return df

    def _central_diff_y(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂y via centered differences, shape (H, W)."""
        df = np.zeros_like(f)
        df[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * self.dx)
        df[0, :]    = (f[1, :] - f[0, :]) / self.dx
        df[-1, :]   = (f[-1, :] - f[-2, :]) / self.dx
        return df

    def _tendencies(
        self,
        eta: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        tau_x: np.ndarray,
        tau_y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        deta = -self.H0 * (self._central_diff_x(u) + self._central_diff_y(v))
        du   = -self.g * self._central_diff_x(eta) + tau_x / (RHO_W * self.H0) - R_FRICTION * u
        dv   = -self.g * self._central_diff_y(eta) + tau_y / (RHO_W * self.H0) - R_FRICTION * v
        return deta, du, dv

    def _ab2_step(self, tau_x: np.ndarray, tau_y: np.ndarray) -> None:
        """Adams-Bashforth 2 time step."""
        deta, du, dv = self._tendencies(self.eta, self.u, self.v, tau_x, tau_y)

        if self._step_count == 0:
            # Euler on the first step
            self.eta += self.dt * deta
            self.u   += self.dt * du
            self.v   += self.dt * dv
        else:
            # AB2: y_{n+1} = y_n + dt * (3/2 * f_n - 1/2 * f_{n-1})
            self.eta += self.dt * (1.5 * deta - 0.5 * self._deta_prev)
            self.u   += self.dt * (1.5 * du   - 0.5 * self._du_prev)
            self.v   += self.dt * (1.5 * dv   - 0.5 * self._dv_prev)

        self._deta_prev[:] = deta
        self._du_prev[:]   = du
        self._dv_prev[:]   = dv

    def _apply_boundaries(self) -> None:
        """No-flux walls on N/E/W; radiation on S."""
        self.u[:, 0]  = 0.0
        self.u[:, -1] = 0.0
        self.v[-1, :] = 0.0
        # South: Sommerfeld radiation
        c = np.sqrt(self.g * np.maximum(self.H0[1, :], 0.5))
        self.eta[0, :] = self.eta[1, :] - (self.dt * c / self.dx) * (self.eta[1, :] - self.eta[2, :])

    def _shapiro_smooth(self) -> None:
        """Apply a mild 1-2-1 Shapiro filter to η to suppress 2Δx noise."""
        eta = self.eta
        smooth_x = 0.25 * (eta[:, :-2] + 2 * eta[:, 1:-1] + eta[:, 2:])
        eta[:, 1:-1] = smooth_x
        smooth_y = 0.25 * (eta[:-2, :] + 2 * eta[1:-1, :] + eta[2:, :])
        eta[1:-1, :] = smooth_y

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Return (3, H, W) float32: [η, u, v]."""
        return np.stack([self.eta, self.u, self.v]).astype(np.float32)

    def get_flood_depth(self) -> np.ndarray:
        """Flood depth on land cells: max(0, η − bathy). Shape (H, W)."""
        flood = np.maximum(0.0, self.eta - self.bathy.astype(np.float64))
        flood[~self.land_mask] = 0.0
        return flood.astype(np.float32)

    def get_water_level(self) -> np.ndarray:
        """Total water level η [m] (sea surface anomaly). Shape (H, W)."""
        return self.eta.astype(np.float32)


# ---------------------------------------------------------------------------
# High-level simulation runner
# ---------------------------------------------------------------------------

def run_hurricane_simulation(
    bathy: np.ndarray,
    track: List[Tuple[float, float]],
    n_hours: int = 24,
    steps_per_hour: int = 180,
    n_snapshots: int = 25,
    dx: float = 1000.0,
    dt: float = 20.0,
    max_wind_ms: float = 55.0,
    radius_km: float = 80.0,
) -> Dict:
    """Run a full hurricane storm surge simulation and return snapshots.

    Returns
    -------
    dict with keys:
        'eta'     — (n_snapshots, H, W) sea-surface elevation anomaly [m]
        'flood'   — (n_snapshots, H, W) flood depth on land [m]
        'wind_u'  — (n_snapshots, H, W)
        'wind_v'  — (n_snapshots, H, W)
        'bathy'   — (H, W)
        'track'   — list of (x_km, y_km)
        'times_h' — list of float [hours]
    """
    H, W = bathy.shape
    solver = ShallowWaterSolver(H=H, W=W, dx=dx, dt=dt)
    solver.set_bathymetry(bathy)
    solver.reset()

    total_steps = n_hours * steps_per_hour
    save_every = max(1, total_steps // (n_snapshots - 1))

    eta_snaps, flood_snaps, wu_snaps, wv_snaps, times = [], [], [], [], []

    # t = 0 snapshot
    wu0, wv0 = gaussian_hurricane_wind(H, W, dx, 0.0, track, radius_km, max_wind_ms)
    eta_snaps.append(solver.get_water_level())
    flood_snaps.append(solver.get_flood_depth())
    wu_snaps.append(wu0)
    wv_snaps.append(wv0)
    times.append(0.0)

    for step_idx in range(1, total_steps + 1):
        t_hours = step_idx / steps_per_hour
        wu, wv = gaussian_hurricane_wind(H, W, dx, t_hours, track, radius_km, max_wind_ms)
        solver.step(wu, wv, n_steps=1)

        if step_idx % save_every == 0 and len(eta_snaps) < n_snapshots:
            eta_snaps.append(solver.get_water_level())
            flood_snaps.append(solver.get_flood_depth())
            wu_snaps.append(wu)
            wv_snaps.append(wv)
            times.append(t_hours)

    # Pad to exactly n_snapshots
    while len(eta_snaps) < n_snapshots:
        eta_snaps.append(eta_snaps[-1])
        flood_snaps.append(flood_snaps[-1])
        wu_snaps.append(wu_snaps[-1])
        wv_snaps.append(wv_snaps[-1])
        times.append(times[-1])

    return {
        "eta":    np.stack(eta_snaps[:n_snapshots]),
        "flood":  np.stack(flood_snaps[:n_snapshots]),
        "wind_u": np.stack(wu_snaps[:n_snapshots]),
        "wind_v": np.stack(wv_snaps[:n_snapshots]),
        "bathy":  bathy,
        "track":  track,
        "times_h": times[:n_snapshots],
    }


# ---------------------------------------------------------------------------
# Track generator
# ---------------------------------------------------------------------------

def random_hurricane_track(
    H: int = 64,
    W: int = 64,
    dx: float = 1000.0,
    n_hours: int = 24,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[float, float]]:
    """Generate a random northward-moving hurricane track.

    Returns list of (x_km, y_km) at each integer hour 0..n_hours.
    """
    if rng is None:
        rng = np.random.default_rng()

    domain_x_km = W * dx / 1e3
    domain_y_km = H * dx / 1e3

    landfall_x = rng.uniform(0.25, 0.75) * domain_x_km
    start_y = -0.3 * domain_y_km
    fwd_speed_kmh = rng.uniform(18.0, 48.0)
    drift_kmh = rng.uniform(-6.0, 6.0)

    return [
        (float(landfall_x + drift_kmh * t), float(start_y + fwd_speed_kmh * t))
        for t in range(n_hours + 1)
    ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Running hurricane storm surge demo …")
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()

    H, W = 64, 64
    bathy = random_coastal_bathymetry(H, W, rng=rng)
    track = random_hurricane_track(H, W, n_hours=24, rng=rng)
    result = run_hurricane_simulation(bathy, track, n_hours=24, n_snapshots=25)

    elapsed = time.perf_counter() - t0
    max_flood = float(np.nanmax(result["flood"]))
    max_eta = float(np.nanmax(np.abs(result["eta"])))
    print(f"  Solver finished in {elapsed:.1f}s")
    print(f"  Max sea-surface anomaly: {max_eta:.2f} m")
    print(f"  Max flood depth on land: {max_flood:.2f} m")
    print(f"  Snapshots: {len(result['times_h'])} × {H}×{W}")

    snap_ids = [0, 4, 8, 12, 16, 24]
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))

    for col, si in enumerate(snap_ids):
        t_h = result["times_h"][si]
        ax_flood = axes[0, col]
        ax_wind  = axes[1, col]

        im = ax_flood.imshow(result["flood"][si], origin="lower", cmap="Blues",
                             vmin=0, vmax=max(max_flood, 0.1))
        ax_flood.contour(bathy, levels=[0], colors="peru", linewidths=1.0)
        ax_flood.set_title(f"Flood [m]\nt={t_h:.1f}h", fontsize=8)
        ax_flood.axis("off")
        plt.colorbar(im, ax=ax_flood, fraction=0.046)

        wspd = np.sqrt(result["wind_u"][si] ** 2 + result["wind_v"][si] ** 2)
        im2 = ax_wind.imshow(wspd, origin="lower", cmap="YlOrRd", vmin=0, vmax=65)
        step = 8
        ys2, xs2 = np.mgrid[0:H:step, 0:W:step]
        ax_wind.quiver(xs2, ys2,
                       result["wind_u"][si][::step, ::step],
                       result["wind_v"][si][::step, ::step],
                       scale=500, width=0.005, color="white")
        ax_wind.set_title(f"Wind [m/s]\nt={t_h:.1f}h", fontsize=8)
        ax_wind.axis("off")
        plt.colorbar(im2, ax=ax_wind, fraction=0.046)

    plt.suptitle("Hurricane Storm Surge — Shallow Water Solver", fontsize=12)
    plt.tight_layout()
    out = Path(__file__).parent / "results" / "solver_demo.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  Saved → {out}")
