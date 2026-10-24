# hurricane_flood/

End-to-end hurricane storm surge demo showcasing Solaris's combined
physics-constrained + multi-physics + uncertainty capabilities.

## What this project does

A Category 4 hurricane tracks northward and makes landfall on a synthetic
coastal domain.  A traditional shallow water equations (SWE) solver provides
ground truth.  A neural surrogate trained once predicts the same scenario in
milliseconds — and comes with a provable uncertainty guarantee.

**Guarantees built into the model (hold for any weights, also after coupling):**
- **Flood depth is never negative** — the flood operator ends in a
  non-negative output (`ConstrainedFNO(constraint="nonnegative")`).
- **No flooding over open ocean** — the output is multiplied by the land mask
  (an input channel), so ocean cells are exactly 0, as in the solver.
- **Calibrated uncertainty** — `ConformalNeuralOperator` is calibrated on
  storms the model never trained on; coverage is then measured on a further
  set of held-out storms. The guarantee: for a new storm from the same
  distribution, the true depth lies inside `[prediction ± q̂]` everywhere on
  the map with probability ≥ 90 %.

What is deliberately *not* enforced: flood volume is not conserved between
time steps (water rises and drains), and the wind is not divergence-free (the
simulated storm spirals inward), so neither is imposed as a constraint.

Data are split **by storm** (70 / 10 / 10 / 10 % train / validation /
calibration / test) so no storm's time steps appear in two splits, and
normalisation statistics come from the training storms only.

## Models used

| Component | Model | Constraint |
|-----------|-------|-----------|
| Flood depth | `FloodOperator` (`ConstrainedFNO` × land mask) | depth ≥ 0, depth = 0 on ocean |
| Wind field | `FNO` | none |
| Coupling | `CoupledOperator` | `coupling_mode="learned"` (re-applies the flood constraints) |
| Uncertainty | `ConformalNeuralOperator` | split-conformal, α = 0.1 |

## Physics

Linearised 2-D shallow water equations:

```
∂η/∂t + H₀·(∂u/∂x + ∂v/∂y) = 0
∂u/∂t = -g·∂η/∂x + τx/(ρ·H₀) - r·u
∂v/∂t = -g·∂η/∂y + τy/(ρ·H₀) - r·v
```

where η is sea-surface elevation, H₀ is background depth, τ is the bulk
wind stress, and r is linear bottom friction.  Solved with Adams-Bashforth 2
time-stepping on a 64×64 km coastal grid (1 km/cell, dt=20 s).

## Quickstart

```bash
# 1. Smoke-test the solver (saves results/solver_demo.png)
python solver.py

# 2. Train the surrogate (quick CPU test; needs >= 10 storms for the 4-way split)
python train.py --n_sims 20 --epochs 5

# 3. Full training run
python train.py --n_sims 500 --epochs 80 --device cuda

# 4. Benchmark surrogate vs solver
python compare.py --n 10

# 5. Generate the animated visualization
#    Solver only (no checkpoint needed):
python visualize.py --generate --solver-only --output results/hurricane_flood.gif

#    Full surrogate + uncertainty (requires trained checkpoint):
python visualize.py --generate --output results/hurricane_flood.gif

#    Interactive window:
python visualize.py --generate --solver-only --interactive
```

## Output files

| File | Description |
|------|-------------|
| `results/solver_demo.png` | 6-panel snapshot grid from the SWE solver |
| `results/compare.png` | 5-column benchmark figure (bathy / solver / surrogate / error / uncertainty) |
| `results/hurricane_flood.gif` | Animated 2×2 visualization |
| `checkpoints/best_coupled.pt` | Trained CoupledOperator weights |
| `checkpoints/conformal_predictor.pt` | ConformalNeuralOperator with calibrated q̂ |
| `checkpoints/norm_stats.npz` | Normalisation statistics (training storms only) |

## Visualization panels

```
┌─────────────────────┬──────────────────────┐
│  Flood depth [m]    │  Hurricane wind [m/s]│
│  Blues + coastline  │  speed map + quiver  │
├─────────────────────┼──────────────────────┤
│  Uncertainty ±q̂   │  Cumulative extent   │
│  90% coverage band  │  ever-flooded mask   │
└─────────────────────┴──────────────────────┘
```
