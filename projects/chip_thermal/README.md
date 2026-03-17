# Chip Thermal Prediction — 3-D Real-Time Heat Simulation

Simulate transient heat flow through a silicon die (x, y, z) from cold start,
and learn a fast FNO surrogate that predicts the full volumetric temperature
field at any time during the first 10 ms.

```
ρCp ∂T/∂t = k ∇²T + Q(x,y,z)

Domain  : 1 mm × 1 mm × 0.5 mm
Grid    : 32 × 32 × 16
α       : 8.8×10⁻⁵ m²/s  (silicon)
Time    : 0 → 10 ms
IC      : T = 25 °C  (cold start)
BC      : Dirichlet T = 25 °C on bottom + four lateral faces (heat sink)
          Neumann ∂T/∂z = 0  (adiabatic) on top face
Q       : Gaussian hotspots up to 10⁹ W/m³
```

**Without AI** (`solver.py`): explicit finite-difference, ~5–30 s per simulation (12 000+ time steps).
**With AI** (`train.py` + `compare.py`): FNO-3D surrogate, <20 ms per simulation — **500–2000× faster**.

Real use case: die architects run thousands of 3-D layout variants to optimise
thermals in tight power-budget designs. The surrogate makes that loop real-time.

---

## Quickstart

```bash
cd projects/chip_thermal

# Step 0 — install matplotlib if you want plots / animations
pip install matplotlib pillow

# Step 1 — sanity-check the solver alone (no ML needed)
python3.11 solver.py
# → saves solver_demo.png  (2-D demo) and  solver_3d_demo.png  (3-D demo)

# Step 2 — generate 3-D transient data + train FNO surrogate
python3.11 train.py --device cuda --n_sim 300 --epochs 80
# Generates 300 simulations × 8 time snapshots via solver → 2400 samples.
# Cached in data/thermal_3d_dataset.npz so it won't regenerate on next run.
# Saves checkpoints/best_fno_3d.pt  and  checkpoints/norm_stats_3d.npz

# Quick smoke-test (10 sims, 5 epochs — completes in under 2 min on CPU)
python3.11 train.py --n_sim 10 --epochs 5

# Step 3 — head-to-head benchmark
python3.11 compare.py --device cuda --n 20
# → prints timing + error table, saves results/compare_3d_1.png … _3.png

# Step 4 — animated visualisation
python3.11 visualize.py --device cuda
# → saves results/chip_thermal_3d.gif
```

### CPU-only (no GPU)

```bash
python3.11 train.py --device cpu --n_sim 50 --epochs 30
python3.11 compare.py --device cpu --n 5
python3.11 visualize.py --device cpu
```

---

## What you'll see

### Benchmark table (`compare.py`)

```
   #    Solver (s)    FNO (ms)    Speedup    Rel-L2
---------------------------------------------------------
   1         8.312       12.41      670×      0.0243
   2         9.105       12.38      735×      0.0218
   3         7.891       12.44      635×      0.0271
  ...
=========================================================
  Solver  avg: 8.77s
  FNO     avg: 12.4ms
  Speedup avg: 707×
  Rel-L2  avg: 0.0241  (max 0.0389)
```

### Slice-panel figure (`compare_3d_N.png`)

Three rows, one per cut plane, four columns:

| Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|
| Q map [W/m³] | FD Solver T [°C] | FNO T [°C] | \|Error\| |

Rows: XY mid-plane · XZ mid-plane · YZ mid-plane.

### Animated GIF (`chip_thermal_3d.gif`)

3 rows × 2 cols cycling through 8 time snapshots (0–10 ms), comparing solver
and FNO side-by-side for each cut plane.

---

## Old 2-D vs New 3-D

| Feature | 2-D Steady-State (old) | 3-D Transient (new) |
|---|---|---|
| Physics | Steady-state Poisson | Transient heat equation |
| Grid | 64 × 64 | 32 × 32 × 16 |
| Solver | Gauss-Seidel iterative | Explicit FD (12 000+ steps) |
| Solver time | 0.5–2 s/design | 5–30 s/simulation |
| Model | FNO-2D, 1-channel in | FNO-3D, 2-channel in (Q + t) |
| Inputs | Q(x,y) | Q(x,y,z) + time t |
| Output | T(x,y) at equilibrium | T(x,y,z) at any t ∈ [0, 10ms] |
| Typical speedup | 100–500× | 500–2000× |

---

## Tuning

| Flag | Default | Notes |
|---|---|---|
| `--n_sim` | 300 | More sims → lower error |
| `--epochs` | 80 | More epochs → lower error |
| `--hidden` | 32 | FNO channel width (increase for more capacity) |
| `--modes` | 8 | Fourier modes kept per dimension |
| `--n_layers` | 4 | FNO depth |
| `--n_times` | 8 | Snapshot count per simulation |
