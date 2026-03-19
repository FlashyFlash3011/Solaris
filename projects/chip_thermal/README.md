# Chip Thermal — FNO Surrogate for 2-D Steady-State Heat Conduction

Predict the steady-state temperature field on a chip given its power-density map:

```
-∇²T = Q(x,y)    on [0,1]²
T = T_ambient     on boundary  (isothermal package)
```

The FNO learns this mapping from (Q → T) in ~1% rel-L2 error, then evaluates
**1 000 new chip layouts** in milliseconds instead of ~50 seconds.

---

## Dataset

Generated on-the-fly by the scipy sparse FD solver in `solver.py`.

| Property | Value |
|---|---|
| Source | scipy sparse FD (vectorised Laplacian, direct LU) |
| Resolution | 128 × 128 |
| Samples | 1 000 train + 200 test |
| Temperature range | 40–93 °C (calibrated to real junction temps) |
| Time per sample | ~50 ms on CPU |
| Total generation time | ~1 min |

Chip floorplan: 4 compute cores (with L2 cache shells), L3 cache band, 2 memory
controllers, and an I/O ring — each with randomised utilisation so the model sees
diverse thermal profiles.

---

## Quickstart

```bash
cd projects/chip_thermal

# Step 1 — generate dataset + train (~1 min to generate, ~5 min on GPU)
python train.py --device cuda

# Step 2 — batch throughput comparison (1 000 new layouts)
python compare.py --device cuda
# → prints timing table + saves results/compare.png
```

### CPU-only

```bash
python train.py --device cpu --n_train 200 --epochs 20
python compare.py --device cpu --n_batch 50
```

### Smoke-test the solver alone

```bash
python solver.py
# → prints Q/T ranges, solve time, saves solver_demo.png
```

---

## Expected output

```
============================================================
  Batch size        : 1000 chip layouts
  FD Solver  total  : 49.3s  (49.3 ms/layout)
  FNO        total  : 521ms  (0.52 ms/layout)
  Speedup           : 95×
  Rel-L2 avg        : 0.0088  (max 0.0151)
============================================================
```

`results/compare.png` shows four panels for one representative layout:
1. **Power Map Q(x,y)** — chip architecture with labelled components
2. **FD Solver T [°C]** — reference temperature (ms/layout timing)
3. **FNO Surrogate T [°C]** — prediction (ms/layout timing)
4. **|Error| [°C]** — absolute error + rel-L2 percentage

Supertitle shows total batch speedup: `1000 layouts · FD: 49s · FNO: 521ms · 95× faster`

---

## Tuning

| Flag | Default | Notes |
|---|---|---|
| `--n_train` | 1000 | Training samples |
| `--epochs` | 200 | More epochs → lower rel-L2 |
| `--hidden` | 64 | FNO hidden channel width |
| `--modes` | 16 | Fourier modes kept per dimension |
| `--n_layers` | 4 | FNO depth |
| `--resolution` | 128 | Grid resolution |
| `--n_batch` | 1000 | Compare: layouts in batch test |
| `--batch_size` | 64 | Compare: GPU inference batch size |
