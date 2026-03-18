# Chip Thermal — FNO Surrogate on Real-World PDE Data

Predict the steady-state temperature / pressure field in a variable-conductivity
medium given its permeability map — the Darcy flow / heat conduction equation:

```
-∇·(a(x)∇u) = f    on [0,1]²
 u = 0              on boundary
```

Mathematically identical to steady-state heat conduction with spatially-varying
thermal conductivity `k(x) = a(x)`:  `-∇·(k(x)∇T) = Q`

---

## Dataset

**Pre-made, professionally generated data** (Zenodo record 12784353,
~356 MB download, cached after first run).

| Property | Value |
|---|---|
| Source | FEM solver (FEniCS), high-fidelity |
| Resolution | 128 × 128 |
| Samples available | 10,000+ |
| Training samples used | 1,000 |
| Test samples | 200 |
| Time to generate (FEM) | Hours on a workstation |

The FNO is trained on 1,000 of these samples.  At inference time,
the FD solver must re-assemble and solve a 16,000 × 16,000 sparse system
for each new design — taking **~50 ms per design**.  The FNO predicts in
**< 3 milliseconds** — a **~20× per-query speedup**.

For an engineer exploring 100,000 design variants:

| Method | Time for 100 k queries |
|---|---|
| Traditional FD solver (128×128) | ~83 minutes |
| FNO surrogate | ~5 minutes |

---

## Quickstart

```bash
cd projects/chip_thermal

# Step 1 — train (downloads ~356 MB Darcy dataset on first run, ~5 min on GPU)
python train.py --device cuda

# Step 2 — head-to-head comparison
python compare.py --device cuda --n 20
# → prints timing table + saves results/compare.png
```

### CPU-only

```bash
python train.py --device cpu --n_train 200 --epochs 20
python compare.py --device cpu --n 5
```

### Smoke-test the solver alone

```bash
python solver.py
# → prints FD solver vs FEM ground truth rel-L2, saves solver_demo.png
```

---

## Expected output

```
   #    FD Solver (s)    FNO (ms)    Speedup    Rel-L2
---------------------------------------------------------
   1            0.052        2.81       19×      0.0071
   2            0.049        2.78       18×      0.0083
   3            0.051        2.80       18×      0.0065
  ...
=========================================================
  FD Solver  avg: 0.051s
  FNO        avg: 2.80ms
  Speedup    avg: 18×
  Rel-L2     avg: 0.0076  (max 0.0124)
```

`results/compare.png` shows four columns:
1. Conductivity map `a(x)` — where the material varies
2. FEM ground truth `u(x)` — the reference solution
3. FNO prediction — what the surrogate produces in <3ms
4. Absolute error `|u_FNO − u_FEM|`

---

## Tuning

| Flag | Default | Notes |
|---|---|---|
| `--n_train` | 1000 | Training samples (up to 10 k in dataset) |
| `--subsample` | 1 | Spatial stride (1 = full 128×128) |
| `--epochs` | 100 | More epochs → lower rel-L2 |
| `--hidden` | 64 | FNO hidden channel width |
| `--modes` | 12 | Fourier modes kept per dimension |
| `--n_layers` | 4 | FNO depth |
| `--resolution` | 128 | Dataset resolution (128 or 421) |
