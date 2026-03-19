# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
solver.py — 2-D steady-state chip thermal simulation.

Physics
-------
    -∇²T = Q(x,y)      on Ω = [0,1]²
    T = T_ambient       on ∂Ω  (isothermal package boundary)

Effective thermal conductivity is absorbed into the power-map units, so the
equation is dimensionless but Q is calibrated to reproduce a realistic
junction-temperature range (40–93 °C at full chip load).

Public API
----------
chip_floorplan_power_map()  — realistic quad-core power-density map
solve_heat_fd()             — scipy sparse direct solver
generate_dataset()          — build & save (Q, T) training pairs to disk
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_DIR = Path(__file__).parent / "data"

# Normalised chip geometry (origin = bottom-left corner, y↑)
# Each entry: (xmin, ymin, xmax, ymax)
_CORES = [
    (0.08, 0.08, 0.47, 0.43),   # Core 0 — bottom-left
    (0.53, 0.08, 0.92, 0.43),   # Core 1 — bottom-right
    (0.08, 0.57, 0.47, 0.92),   # Core 2 — top-left
    (0.53, 0.57, 0.92, 0.92),   # Core 3 — top-right
]
# Annotation positions for compare.py figure (normalised, (x, y) from bottom-left)
LAYOUT_LABELS = [
    (0.275, 0.255, "Core 0"),
    (0.725, 0.255, "Core 1"),
    (0.275, 0.745, "Core 2"),
    (0.725, 0.745, "Core 3"),
    (0.500, 0.500, "L3 Cache"),
    (0.090, 0.500, "MC"),
    (0.910, 0.500, "MC"),
    (0.500, 0.030, "I/O Ring"),
]


# ─── Power map ───────────────────────────────────────────────────────────────

def chip_floorplan_power_map(H: int = 128, W: int = 128, rng=None) -> np.ndarray:
    """Generate a (H, W) power-density map for a randomised chip workload.

    Power values are calibrated so that fully-loaded peak junction temperature
    ≈ 93 °C when solved with solve_heat_fd(T_ambient=40).

    Returns
    -------
    Q : ndarray (H, W), float32
    """
    if rng is None:
        rng = np.random.default_rng()

    x  = np.linspace(0, 1, W, dtype=np.float32)
    y  = np.linspace(0, 1, H, dtype=np.float32)
    XX, YY = np.meshgrid(x, y)            # YY[i, j] = y-coordinate of row i

    Q = np.full((H, W), 120.0, dtype=np.float32)   # background leakage

    # I/O ring — outer 6 % of die
    io = (XX < 0.06) | (XX > 0.94) | (YY < 0.06) | (YY > 0.94)
    Q[io] = float(rng.uniform(200, 450))

    # L3 cache — horizontal band
    l3_util = float(rng.uniform(0.30, 0.95))
    Q[(YY >= 0.46) & (YY <= 0.54) & (XX >= 0.06) & (XX <= 0.94)] = 480 * l3_util

    # Memory controllers — override the two ends of the L3 band
    mc_util = float(rng.uniform(0.30, 0.90))
    mc = ((XX >= 0.06) & (XX <= 0.12) | (XX >= 0.88) & (XX <= 0.94)) & \
         (YY >= 0.46) & (YY <= 0.54)
    Q[mc] = 1100 * mc_util

    # Four compute cores
    for xmin, ymin, xmax, ymax in _CORES:
        util = float(rng.uniform(0.20, 1.00))
        core   = (XX >= xmin) & (XX <= xmax) & (YY >= ymin) & (YY <= ymax)
        shell  = core & ~(
            (XX >= xmin + 0.04) & (XX <= xmax - 0.04) &
            (YY >= ymin + 0.04) & (YY <= ymax - 0.04)
        )
        compute = core & (
            (XX >= xmin + 0.04) & (XX <= xmax - 0.04) &
            (YY >= ymin + 0.04) & (YY <= ymax - 0.04)
        )
        Q[shell]   = 550  * util   # L2 cache ring
        Q[compute] = 2200 * util   # compute units

    return Q


# ─── Sparse Laplacian ────────────────────────────────────────────────────────

def _laplacian_matrix(H: int, W: int) -> sp.csr_matrix:
    """Return the (H-2)(W-2) × (H-2)(W-2) FD Laplacian matrix A where

        A u = f   ≡   -∇²u = f   (interior nodes, u=0 on boundary)

    Entries: diagonal = 4/h², off-diagonal = −1/h²,  h = 1/(H−1).
    Fully vectorised — no Python loops.
    """
    h      = 1.0 / (H - 1)
    ni, nj = H - 2, W - 2
    n      = ni * nj
    c      = 1.0 / h**2          # coefficient

    idx = np.arange(n).reshape(ni, nj)

    # Diagonal
    dr, dc, dv = idx.ravel(), idx.ravel(), np.full(n, 4 * c)
    # East / West
    er, ec, ev = idx[:, :-1].ravel(), idx[:, 1:].ravel(), np.full((ni)*(nj-1), -c)
    wr, wc, wv = idx[:, 1:].ravel(),  idx[:, :-1].ravel(), np.full((ni)*(nj-1), -c)
    # North / South
    nr, nc_, nv = idx[:-1, :].ravel(), idx[1:, :].ravel(), np.full((ni-1)*nj, -c)
    sr, sc, sv  = idx[1:, :].ravel(),  idx[:-1, :].ravel(), np.full((ni-1)*nj, -c)

    rows = np.concatenate([dr, er, wr, nr, sr])
    cols = np.concatenate([dc, ec, wc, nc_, sc])
    vals = np.concatenate([dv, ev, wv, nv,  sv])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


# ─── Solver ──────────────────────────────────────────────────────────────────

def solve_heat_fd(
    Q: np.ndarray,
    T_ambient: float = 40.0,
) -> Tuple[np.ndarray, float]:
    """Solve -∇²T = Q  with T = T_ambient on ∂Ω  using scipy sparse (LU).

    Parameters
    ----------
    Q         : ndarray (H, W), power-density map
    T_ambient : float, boundary/ambient temperature [°C]

    Returns
    -------
    T       : ndarray (H, W) in °C
    elapsed : float [seconds]
    """
    H, W = Q.shape
    t0  = time.perf_counter()
    A   = _laplacian_matrix(H, W)
    rhs = Q[1:-1, 1:-1].ravel().astype(np.float64)
    T_rise = spla.spsolve(A, rhs)
    elapsed = time.perf_counter() - t0

    T = np.full((H, W), T_ambient, dtype=np.float32)
    T[1:-1, 1:-1] = T_ambient + T_rise.reshape(H - 2, W - 2)
    return T, elapsed


# ─── Dataset generation ──────────────────────────────────────────────────────

def generate_dataset(
    n_train: int = 1000,
    n_test:  int = 200,
    H:       int = 128,
    W:       int = 128,
    seed:    int = 0,
    out_path: Path = None,
) -> Path:
    """Generate (Q, T) pairs with the chip thermal solver and save to .npz.

    Skipped silently if the file already exists.
    """
    if out_path is None:
        out_path = DATA_DIR / "chip_thermal_dataset.npz"
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng   = np.random.default_rng(seed)
    total = n_train + n_test

    print(f"Generating {total} samples at {H}×{W} with scipy sparse FD …")
    Q_all = np.empty((total, H, W), dtype=np.float32)
    T_all = np.empty((total, H, W), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(total):
        Q = chip_floorplan_power_map(H, W, rng)
        T, _ = solve_heat_fd(Q)
        Q_all[i], T_all[i] = Q, T
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i+1}/{total}  ({rate:.1f} samples/s, "
                  f"ETA {(total-i-1)/rate:.0f}s)")

    np.savez(
        out_path,
        Q_train=Q_all[:n_train], T_train=T_all[:n_train],
        Q_test =Q_all[n_train:], T_test =T_all[n_train:],
    )
    elapsed = time.perf_counter() - t0
    print(f"Saved → {out_path}  ({elapsed:.1f}s, {total/elapsed:.1f} samples/s)")
    return out_path


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng   = np.random.default_rng(42)
    Q     = chip_floorplan_power_map(128, 128, rng)
    T, dt = solve_heat_fd(Q)

    print(f"Q  range : [{Q.min():.0f}, {Q.max():.0f}]")
    print(f"T  range : [{T.min():.1f} °C, {T.max():.1f} °C]")
    print(f"Solve time: {dt*1000:.1f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(Q, cmap="hot", origin="lower")
    axes[0].set_title("Power Map Q(x,y)", fontsize=11); axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], label="Power Density")

    im1 = axes[1].imshow(T, cmap="inferno", origin="lower",
                         vmin=T.min(), vmax=T.max())
    axes[1].set_title(f"Temperature  [{T.min():.0f}–{T.max():.0f} °C]", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], label="T [°C]")

    plt.tight_layout()
    plt.savefig("solver_demo.png", dpi=130)
    print("Saved solver_demo.png")
