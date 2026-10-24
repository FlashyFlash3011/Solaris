# solaris/

The library package.

| Module | Contents |
|---|---|
| `core/` | `Module` base class (self-describing save/load, safe checkpoint loading), `ModelMetaData`, `ModelRegistry` |
| `models/` | FNO, ConstrainedFNO, NeuralResidualCorrector, MultiScaleFNO, UNO, WNO, AFNO, DeepONet, MeshGraphNet, FullyConnected, CoupledOperator, ConformalNeuralOperator |
| `nn/` | Spectral convolutions (1-D/2-D/3-D), constraint layers (divergence-free, curl-free, conservation, boundary value, Neumann/Dirichlet), activations, embeddings, block normalisation |
| `datapipes/` | Datasets (in-memory, `.npy`, HDF5), transforms (incl. vector-aware symmetry augmentation), config-keyed dataset cache, leakage-free splits |
| `distributed/` | `DistributedManager` (RCCL/NCCL or gloo), distributed 2-D FFT, tensor-parallel primitives |
| `metrics/` | `relative_l2_error` (per-sample), `rmse`, `nrmse`, `r2_score` |
| `utils/` | Checkpoints (incl. FSDP/DCP), schedulers, early stopping, gradient clipping, graph capture, ONNX export, logging, profiling markers, seeding, Optuna tuner, W&B logger |
