# solaris/

Core package. Contains all neural network architectures, physics constraint layers, training utilities, metrics, and distributed training support.

| Module | Contents |
|---|---|
| `models/` | FNO, AFNO, MeshGraphNet, ConstrainedFNO, MultiScaleFNO, NeuralResidualCorrector, ConformalNeuralOperator, CoupledOperator |
| `nn/` | Spectral convolutions, activations, embeddings, physics constraint layers |
| `core/` | Base module class, model metadata, model registry |
| `distributed/` | ROCm-aware distributed training manager (RCCL backend) |
| `datapipes/` | Datasets, dataloaders, transforms |
| `metrics/` | relative_l2_error, rmse, nrmse, r2_score |
| `utils/` | Checkpointing, logging |
