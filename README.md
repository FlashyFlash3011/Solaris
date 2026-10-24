# Solaris

[![Tests](https://github.com/FlashyFlash3011/Solaris/actions/workflows/tests.yml/badge.svg)](https://github.com/FlashyFlash3011/Solaris/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](pyproject.toml)

**Neural operators for physics simulation, on AMD GPUs.**

Classical PDE solvers are accurate but slow: every new design or boundary
condition means solving from scratch. Solaris trains neural networks on solved
examples so new cases can be predicted in milliseconds — and adds the things
you need to trust those predictions: physics laws enforced exactly, error bars
with a statistical guarantee, and a way to fall back on a cheap solver for the
part a network gets wrong.

It runs on AMD GPUs through PyTorch's ROCm backend (and unchanged on NVIDIA
GPUs or CPU). Its design follows [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo).

> **Status: 0.2 — research-grade library.** The core models and utilities are
> tested (unit, multi-process and end-to-end tests), but the API may still
> change. See [CHANGELOG.md](CHANGELOG.md) and [Known limitations](#known-limitations).

---

## What's in the box

| Feature | What it gives you | How it is verified |
|---|---|---|
| **Neural operators** — FNO (1-D/2-D/3-D), AFNO, UNO, WNO, DeepONet, MeshGraphNet, MLP | Learn solution operators of PDEs; resolution-flexible spectral models | unit tests per model |
| **Hard physics constraints** — `ConstrainedFNO` | Outputs that are divergence-free, conservative, satisfy a Dirichlet boundary value, or are non-negative — exactly, for any weights | tests check each law to float precision, including after multi-physics coupling |
| **Residual correction** — `NeuralResidualCorrector` | A cheap, unconverged solver does the bulk of the physics; the network learns only its error | chip-thermal benchmark below |
| **Calibrated uncertainty** — `ConformalNeuralOperator` | Error bands with a finite-sample coverage guarantee, for any trained model | statistical test over 400 draws; 90.2 % coverage on 1 000 unseen chip layouts (target 90 %) |
| **Multi-physics coupling** — `CoupledOperator` | Several operators that exchange information, starting from the uncoupled solution | tests; hurricane demo |
| **Multi-scale FNO** — `MultiScaleFNO` | Separate spectral heads for low/mid/high frequency bands that mean the same thing at every resolution | tests |
| **Multi-GPU** | Data-parallel training example (`torchrun`), tensor-parallel 2-D spectral convolution, sharded (FSDP) checkpoints | 2-process tests compare against single-process results |
| **Training utilities** | `WarmupCosineScheduler`, HIP/CUDA graph capture (`StaticCaptureTraining`), early stopping, gradient clipping, Optuna tuning, W&B logging, safe checkpoints | tests; graph-capture speed-up measured in `benchmarks/` |
| **Export** | ONNX export of FNO-family models, AFNO and MLPs | onnxruntime reproduces PyTorch outputs |

---

## Results

### Chip thermal: power map → steady-state temperature

1 000 fresh 128 × 128 chip layouts that no model saw during training; ground
truth from a sparse finite-difference solve (scipy, CPU). "Error on the rise"
is relative to the temperature rise above the 40 °C package boundary — the
part the model actually has to predict.

| Model | Error on the rise (mean / worst) | Worst error | Boundary deviation | Time per layout | vs FD solver |
|---|---|---|---|---|---|
| FNO | 0.074% / 0.76% | 0.55 °C | 0.41 °C | 1.6 ms | 16× faster |
| Constrained FNO (exact 40 °C boundary) | 0.074% / 0.74% | 0.57 °C | **0 (exact)** | 1.4 ms | 18× faster |
| Residual corrector (coarse FD + FNO) | 0.012% / 0.07% | 0.08 °C | 0.08 °C | 2.7 ms | 9× faster |

FD solver: 25.4 ms per layout (scipy sparse LU, CPU). Models: batched on an AMD Radeon RX 9070 XT (ROCm 7.2, WSL2), including host↔device copies. All models: width 64, 4 layers, 16 modes, 200 epochs, `norm=None`.

**Calibrated error bars (FNO):** calibrated on 500 fresh layouts, the conformal intervals are ±0.14 °C and contained the true temperature at every pixel for 88.9 % of 1 000 further layouts (target 90 %; a single run is expected to land within about ±1.6 points of 90.2 % — the average is verified by a 400-run statistical test).

Reproduce: `python projects/chip_thermal/train.py --model {fno,constrained,residual} --norm none`,
then `python benchmarks/chip_thermal.py --checkpoint_dir projects/chip_thermal/checkpoints`.

### Other demos

| Project | Physics | What it shows |
|---|---|---|
| [`chip_thermal`](projects/chip_thermal) | steady 2-D heat conduction | FNO vs boundary-constrained FNO vs residual corrector; conformal error bars |
| [`hurricane_flood`](projects/hurricane_flood) | storm surge (shallow-water equations) | coupled flood + wind operators; flood depth ≥ 0 and = 0 over ocean, exactly; calibrated on held-out storms |
| [`navier_stokes`](projects/navier_stokes) | 2-D vorticity | FNO time stepping |
| [`wave_equation`](projects/wave_equation) | 2-D wave propagation | time-conditioned FNO |
| [`water_heat_diffusion`](projects/water_heat_diffusion) | time-dependent heat diffusion | FNO queried at arbitrary times |
| [`weather_forecast`](projects/weather_forecast) | toy atmospheric advection | AFNO multi-day forecasts |

Every demo trains and evaluates end to end in CI (tiny settings).

---

## Install

Solaris needs PyTorch ≥ 2.4. **Install the right PyTorch build first** — Solaris
itself only requires `torch`, `numpy` and `loguru`, so pip never replaces the
PyTorch you installed.

**AMD GPU on Linux (native ROCm):** use the ROCm wheels from pytorch.org, e.g.

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

**AMD GPU under WSL2:** the pytorch.org ROCm wheels cannot see the GPU there
(WSL exposes it as `/dev/dxg`, not `/dev/kfd`). Use AMD's WSL build from
[repo.radeon.com](https://repo.radeon.com/rocm/manylinux/) on top of ROCm for
WSL (tested: ROCm 7.2, `torch 2.9.1+rocm7.2`, Python 3.11).

**NVIDIA GPU or CPU:** `pip install torch`.

Then:

```bash
git clone https://github.com/FlashyFlash3011/Solaris.git && cd Solaris
pip install -e ".[demos]"        # library + what the demo projects need
```

Extras: `demos` (scipy, matplotlib), `datapipes` (HDF5), `viz` (matplotlib),
`tuning` (Optuna), `tracking` (Weights & Biases), `export` (ONNX), `dev`
(tests, lint, type-check).

Check the GPU is visible:

```python
import solaris, torch
print(solaris.get_gpu_backend())        # "rocm", "cuda" or "cpu"
print(torch.cuda.is_available())        # True on ROCm too (torch.cuda maps to HIP)
```

**Docker (native Linux with ROCm only):** `docker build -f Dockerfile.rocm -t solaris .`
(based on `rocm/pytorch` ROCm 7.2 / PyTorch 2.9.1; does not work under WSL2).

---

## Quick start

```python
import torch
from solaris.models import FNO
from solaris.utils import WarmupCosineScheduler

model = FNO(in_channels=1, out_channels=1, hidden_channels=32, n_layers=4,
            modes=12, dim=2, norm=None).to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=100, base_lr=1e-3)

x = torch.randn(8, 1, 64, 64, device="cuda")      # e.g. a source term
y = torch.randn(8, 1, 64, 64, device="cuda")      # e.g. the solution field
for epoch in range(100):
    opt.zero_grad()
    torch.nn.functional.mse_loss(model(x), y).backward()
    opt.step()
    sched.step()

model.save("fno.pt")                               # self-describing checkpoint
same = FNO.load("fno.pt")                          # rebuilt from the saved config
```

Hard constraints and calibrated uncertainty:

```python
from solaris.models import ConstrainedFNO, ConformalNeuralOperator

# temperature pinned to the boundary value on every edge, for any weights
model = ConstrainedFNO(1, 1, constraint="dirichlet", boundary_value=0.0, norm=None)

# after training: intervals that contain the truth for >= 90 % of new samples
predictor = ConformalNeuralOperator(model)
predictor.calibrate(cal_inputs, cal_targets, alpha=0.1)
lower, upper, prediction = predictor.predict(new_inputs)
```

More: [`examples/`](examples) (standalone scripts, including data-parallel
training with `torchrun --nproc_per_node=N examples/train_fno_darcy.py --device cuda`)
and [`projects/`](projects) (complete solver + surrogate pipelines).

---

## Models

| Model | Class | Notes |
|---|---|---|
| FNO | `solaris.models.FNO` | 1-D/2-D/3-D Fourier Neural Operator |
| ConstrainedFNO | `solaris.models.ConstrainedFNO` | `constraint=` `"divergence_free"`, `"conservative"`, `"dirichlet"`, `"nonnegative"`, `"none"` |
| NeuralResidualCorrector | `solaris.models.NeuralResidualCorrector` | coarse solver + learned correction |
| MultiScaleFNO | `solaris.models.MultiScaleFNO` | per-band spectral heads + cross-scale attention |
| UNO | `solaris.models.UNO` | U-Net with Fourier blocks at each scale |
| WNO | `solaris.models.WNO` | Haar-wavelet neural operator (sharp fronts) |
| AFNO | `solaris.models.AFNO` | adaptive Fourier token mixing (weather-style) |
| DeepONet | `solaris.models.DeepONet` | branch/trunk operator network |
| MeshGraphNet | `solaris.models.MeshGraphNet` | message passing on irregular meshes |
| FullyConnected | `solaris.models.FullyConnected` | MLP / PINN baseline (incl. learnable Stan activation) |
| CoupledOperator | `solaris.models.CoupledOperator` | multi-physics composition |
| ConformalNeuralOperator | `solaris.models.ConformalNeuralOperator` | wraps any model with calibrated intervals |

FNO-family models take `norm=None` (the reference FNO, recommended) or
`norm="instance"` (the pre-0.2 default, which hides each sample's mean and
scale from the network). Leaving it out keeps the old behaviour with a
`FutureFeatureWarning`, so existing checkpoints load unchanged.

---

## Known limitations

- **Tested hardware:** AMD Radeon RX 9070 XT (RDNA4, `gfx1201`) under WSL2 with
  ROCm 7.2, plus CPU. Other ROCm GPUs should work through the same PyTorch API
  but have not been tested.
- **`torch.compile` on the GPU** crashes in AMD's WSL build of PyTorch 2.9.1
  (even for trivial functions) — a PyTorch issue; `Module.compile_model()`
  works on CPU. HIP graph capture (`StaticCaptureTraining`) works on the GPU.
- **Tensor-parallel 3-D spectral convolution** is not implemented (multi-rank
  `SpectralConv3d` raises; 2-D is implemented and tested).
- The demo datasets are synthetic and generated by the included solvers; they
  are meant to exercise the methods, not to replace validated engineering data.

---

## Development

```bash
pip install -e ".[dev,export]"
pytest tests/                      # everything (~2 min, includes end-to-end demo runs)
pytest tests/ -m "not slow"        # skip the end-to-end runs
ruff check . && ruff format --check .
mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Projects built on Solaris

- **[plasma-gsfno](https://github.com/FlashyFlash3011/plasma-gsfno)** — FNO surrogate for the
  Grad–Shafranov tokamak equilibrium (coil/vacuum flux + profiles → poloidal flux ψ(R, Z)).

## License

Apache 2.0
