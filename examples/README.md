# examples/

Standalone training scripts. Each script generates its own synthetic data, trains a model, and prints results. All run on CPU or AMD/NVIDIA GPU.

| Script | What it does |
|---|---|
| `train_fno_darcy.py` | FNO on Darcy flow (porous media pressure prediction) |
| `train_pinn_poisson.py` | Physics-informed MLP solving the 2D Poisson equation |
| `train_constrained_darcy.py` | ConstrainedFNO vs standard FNO — shows conservation violation stays at ~0 |
| `train_multiscale_fno.py` | MultiScaleFNO vs FNO on multi-frequency data |
| `demo_residual_correction.py` | NeuralResidualCorrector vs pure FNO — data efficiency comparison |
| `demo_conformal.py` | ConformalNeuralOperator — verifies guaranteed coverage holds |
| `demo_coupled.py` | CoupledOperator — thermal + fluid coupling demo |
