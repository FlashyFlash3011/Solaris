# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (ROCm GPU)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install -e ".[dev,datapipes,viz]"

# Install (CPU / NVIDIA)
pip install torch
pip install -e ".[dev,datapipes,viz]"

# Run all tests
pytest tests/ -v --cov=solaris --cov-report=term-missing

# Run a single test file
pytest tests/test_models.py -v

# Run a single test by name
pytest tests/test_models.py::test_fno_forward -v

# Lint
ruff check solaris/ tests/
ruff format solaris/ tests/
```

## Architecture

Solaris is a physics AI framework targeting AMD ROCm GPUs (but fully CUDA/CPU portable via PyTorch's transparent backend API). It provides neural operators for PDE surrogate modelling, with hard physics constraint enforcement and calibrated uncertainty.

### Core base classes (`solaris/core/`)

Every model inherits from `solaris.core.Module` (not `nn.Module` directly):
- `_capture_init_args(**kwargs)` must be called at the end of `__init__` — stores constructor arguments so checkpoints are self-describing and models can be reconstructed without manual arg tracking.
- Device is tracked via a persistent buffer (`_device_buffer`), so `.device` always reflects the correct device without manual management.
- `save(path)` / `load(path)` produce `.pt` files containing `state_dict`, `init_args`, `class` (dotted import path), and a version field.

`ModelMetaData` is a dataclass attached as `Model._meta`; it carries the model name, nvp_tags (for registry search), and feature flags (`amp`, `onnx`, `jit`, etc.).

`ModelRegistry` is a singleton populated from `pyproject.toml` entry points (`[project.entry-points."solaris.models"]`). Register new models there.

### Models (`solaris/models/`)

| Model | File | Key constraint/feature |
|---|---|---|
| FNO | `fno.py` | 1-D/2-D/3-D Fourier Neural Operator |
| AFNO | `afno.py` | Adaptive FNO (token mixing in Fourier space) |
| ConstrainedFNO | `constrained_fno.py` | Hard `"conservative"` or `"divergence_free"` constraint via projection layer |
| NeuralResidualCorrector | `residual_corrector.py` | Coarse solver output + learned correction |
| MultiScaleFNO | `multiscale_fno.py` | Multi-frequency with cross-scale attention |
| CoupledOperator | `coupled.py` | Composes multiple operators; `coupling_mode="learned"` uses a soft gating matrix |
| ConformalNeuralOperator | `conformal.py` | Wraps any model; `calibrate()` then `predict()` returns `(lower, upper, point)` with ≥ (1−α) coverage |
| DeepONet | `deeponet.py` | Branch+trunk; `forward(u, y)` where `u` is sensor values, `y` is query coordinates |
| WNO | `wno.py` | Wavelet Neural Operator (Haar DWT, good for sharp fronts) |
| UNO | `uno.py` | U-Net Neural Operator (encoder-decoder with FNO blocks at each scale) |
| MeshGraphNet | `meshgraphnet.py` | Graph network for irregular meshes |

`ConstrainedFNO` notes:
- `constraint="divergence_free"` requires `out_channels=2` (enforces ∇·u=0 via Helmholtz decomposition in Fourier space).
- `constraint="conservative"` matches the spatial integral of output channel 0 to input channel 0 (`ConservationProjection(source_channel=0, output_channel=0)`).

`CoupledOperator.forward()` takes and returns `Dict[str, Tensor]` keyed by the operator names passed to the constructor. `ConformalNeuralOperator` wraps a model with `forward(x) -> Tensor`, so a thin adapter is needed when wrapping `CoupledOperator`.

### Neural network layers (`solaris/nn/`)

Spectral convolutions: `SpectralConv1d/2d/3d` — used inside FNO blocks.
Constraint layers: `DivergenceFreeProjection2d`, `ConservationProjection`, `SpectralBandFilter`.
Activations: `CappedGELU`, `Siren`, `Stan`.
Embeddings: `FourierEmbedding`, `SinusoidalTimestepEmbedding`.

### Utils (`solaris/utils/`)

- `save_checkpoint(path, model, optimizer, scheduler, epoch, loss, extra={})` — standard checkpoint format used across all projects.
- `load_checkpoint(path)` — returns the saved dict.
- `get_logger(name)` — Loguru-backed logger, used everywhere instead of `print`.
- `EarlyStopping(patience, min_delta, mode)` and `GradientClipper(max_norm)` in `utils/training.py`.

### Demo projects (`projects/`)

Each project is self-contained with `solver.py` (traditional FD/FV ground truth), `train.py`, `compare.py`, and `visualize.py`. Run scripts from the repo root with `python3.11 projects/<name>/script.py`; each script inserts its own directory and the repo root into `sys.path`.

| Project | Physics | Model |
|---|---|---|
| `chip_thermal/` | Steady-state 2-D heat conduction | FNO, ConstrainedFNO, NeuralResidualCorrector |
| `water_heat_diffusion/` | Time-evolving 2-D heat diffusion | Time-conditioned FNO |
| `weather_forecast/` | Multi-day atmospheric forecasting | AFNO |
| `navier_stokes/` | 2-D vorticity / incompressible NS | FNO |
| `wave_equation/` | 1-D/2-D wave propagation | FNO |
| `hurricane_flood/` | 24-h storm surge (shallow water equations) | CoupledOperator (ConstrainedFNO × 2) + ConformalNeuralOperator |

### Python version

The repo uses `python3.11` for all project scripts (torch installed under 3.11). `python3.10` is used in CI. Do not assume `python` resolves correctly — use `python3.11` or `python3.10` explicitly.
