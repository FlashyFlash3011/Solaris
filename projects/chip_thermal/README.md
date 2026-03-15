# Chip Thermal Prediction — FD Solver vs FNO Surrogate

Predict steady-state temperature distribution across a chip given its power map.

```
-k ∇²T = Q(x,y)    on [0,1]²
 T = 25°C           on boundary
```

**Without PhysicsNeMo** (`solver.py`): iterative finite-difference, ~0.5–2s per design.
**With PhysicsNeMo** (`train.py` + `compare.py`): FNO surrogate, <5ms per design — **100–500× faster**.

Real use case: thermal engineers run thousands of layout variants to find the design that keeps the chip coolest. The surrogate makes that loop real-time.

---

## Quickstart

```bash
cd projects/chip_thermal

# Step 0 — install matplotlib if you want plots
pip install matplotlib

# Step 1 — sanity-check the solver alone (no PhysicsNeMo needed)
python solver.py
# → saves solver_demo.png

# Step 2 — generate data + train FNO surrogate (~5 min on GPU)
python train.py --device cuda --n_train 800 --n_val 200 --epochs 50
# Generates 1000 (power_map, temperature) pairs via solver, then trains.
# Cached in data/thermal_dataset.npz so it won't regenerate next run.
# Saves checkpoints/best_fno.pt

# Step 3 — head-to-head comparison
python compare.py --device cuda --n 20
# → prints timing table + saves results/compare.png
```

### CPU-only (no GPU)

```bash
python train.py --device cpu --n_train 400 --n_val 100 --epochs 30
python compare.py --device cpu --n 10
```

---

## What you'll see

```
   #    Solver (s)    FNO (ms)    Speedup    Rel-L2
-------------------------------------------------------
   1         0.847        3.21      264×      0.0183
   2         0.931        3.18      293×      0.0201
   3         0.712        3.20      223×      0.0157
  ...
======================================================
  Solver   avg: 0.841s
  FNO      avg: 3.21ms
  Speedup  avg: 262×
  Rel-L2   avg: 0.0189  (max 0.0312)
```

`compare.png` shows four columns per sample:
1. Power map (where heat is generated)
2. Ground-truth temperature (from solver)
3. FNO prediction
4. Absolute error

---

## Tuning

| Flag | Default | Notes |
|---|---|---|
| `--resolution` | 64 | Grid size. 128 needs more training data. |
| `--n_train` | 800 | More data → lower error |
| `--epochs` | 50 | More epochs → lower error |
| `--hidden` | 64 | FNO channel width |
| `--modes` | 16 | Fourier modes kept |
| `--n_layers` | 4 | FNO depth |
