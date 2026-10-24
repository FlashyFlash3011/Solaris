# Changelog

## 0.2.0 — 2026-10

A correctness release. A full audit found that several advertised features
did not do what they claimed; this release fixes them and adds tests for each.

### Upgrade notes

- **`norm` for FNO-family models.** `FNO`, `ConstrainedFNO`, `UNO`,
  `NeuralResidualCorrector`, `MultiScaleFNO` and `WNO` take `norm=None`
  (recommended — the reference FNO) or `norm="instance"` (the previous
  behaviour). Instance norm hides each sample's mean and scale from the network
  (an FNO could not learn `y = 2x`). Omitting `norm` keeps the old behaviour and
  emits a `FutureFeatureWarning`, so existing checkpoints load unchanged; the
  default will become `None` in a later release. New models: pass `norm=None`.
- **`ConservationProjection`** now conserves the signed integral (it used to
  balance the sum of absolute values). Default `mode="additive"`;
  `mode="multiplicative"` for strictly positive fields. Only meaningful when the
  input and output channels are in the same physical units.
- **`ConformalNeuralOperator.calibrate`** returns `inf` when the calibration
  set is too small for the requested `alpha`.
- **`WarmupCosineScheduler`** is closed-form: the first epoch trains at
  `base_lr / warmup_epochs` (it used to be 0) and `warmup_epochs=0` starts at
  `base_lr` (it used to stay at `min_lr`).
- **`MultiScaleFNO`** bands are defined by mode index; a trained model's bands
  cover different modes than in 0.1.0, so retrain MultiScaleFNO models.
- **Distributed checkpoints** store epoch/loss/scheduler in `meta.pt` next to
  the DCP shards.
- **`relative_l2_error`** now averages per-sample errors (the literature
  convention). The old batch-pooled value is `reduction="pooled"`; reported
  error numbers change accordingly.
- **Checkpoint loading is safe by default** (`weights_only=True`); pass
  `trusted=True` to load files containing arbitrary Python objects.
- **Dependencies:** only `torch`, `numpy` and `loguru` are required; see the
  extras (`datapipes`, `viz`, `demos`, `tuning`, `tracking`, `export`, `dev`).

### Fixed

- Spectral layers: crash when `modes` exceeded what the grid supports (UNO with
  its defaults); `FNO.set_modes` in 1-D/3-D; fp16 autocast crashes in 1-D/3-D,
  MultiScaleFNO and the Helmholtz projections.
- Distributed FFT gradients were wrong (≈0.55× weight gradients, even on one
  GPU); multi-rank `SpectralConv2d` picked the wrong frequencies on every rank.
  Verified against single-process results with a 2-process test.
- Divergence-free projection was not divergence-free at the Nyquist frequency.
- `SpectralBandFilter` dropped the highest-frequency mode.
- `WNO` crashed for sizes not divisible by `2**levels` after padding.
- `CoupledOperator` started half-coupled instead of uncoupled, and its coupling
  silently broke the hard constraints of the operators it combined.
- Conformal quantile could undershoot the finite-sample order statistic.
- `StaticCaptureTraining` never captured with Adam/AdamW (always fell back to
  eager); LR schedulers now keep working after capture.
- `load_checkpoint_distributed` loaded nothing (and raised).
- `export_onnx` failed on current PyTorch and for every spectral model.
- `PhysicsDataset` crashed on plain `.npy` arrays.
- Constructor-argument capture lost the arguments of subclasses of subclasses;
  `Module.load(map_location=...)` ignored the device.
- `get_logger` relabelled every logger with the most recent name.
- Demo projects: four scripts crashed; compare scripts could not load models
  trained with non-default settings; dataset caches were reused when settings
  changed; train/validation splits leaked simulations between the two.

### Added

- `ConstrainedFNO(constraint="dirichlet", boundary_value=...)` and
  `constraint="nonnegative"`; `apply_constraint()` for re-projection.
- `BoundaryValueProjection` (resolution-independent Dirichlet constraint).
- `solaris.datapipes.cached_arrays` (config-keyed dataset cache) and
  `split_by_group` (leakage-free splits by simulation).
- `DistributedManager.initialize(device_type=...)`, `.backend`, `.reset()`.
- `examples/train_fno_darcy.py` trains data-parallel under `torchrun`.
- `examples/train_conservative_diffusion.py` (replaces the Darcy conservation
  demo, whose "conserved" quantity was not conserved by the physics).
- `benchmarks/static_capture.py`, `benchmarks/chip_thermal.py`.

### Changed

- chip_thermal: the constrained model enforces the 40 °C package boundary
  (the old "conservation" constraint was not a law of this problem);
  calibration uses a random calibration set (the score-based "DP" selection
  broke the coverage guarantee); `--profile_modes` became `--mode_sweep`.
- hurricane_flood: the flood model enforces depth ≥ 0 and depth = 0 over open
  ocean; wind is no longer forced divergence-free (the simulated wind spirals
  inward); data are split by storm and coverage is reported on held-out storms.
- Dockerfile: ROCm 7.2 / PyTorch 2.9.1 base that cannot be replaced by a
  generic torch wheel.

### Removed

- chip_thermal `compare_dp.py` and `dp_policy.py`: the policy never actually
  skipped a solver call, so its "FD calls saved" figure was not real.
