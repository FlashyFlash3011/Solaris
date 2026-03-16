# projects/

End-to-end demo projects. Each has its own data generator, traditional solver, neural surrogate, and comparison script.

| Project | Physics problem | Model used |
|---|---|---|
| `chip_thermal/` | Steady-state 2D heat conduction — predict chip temperature from power map | FNO, ConstrainedFNO, NeuralResidualCorrector |
| `water_heat_diffusion/` | Time-evolving 2D heat diffusion in a microfluidics channel | Time-conditioned FNO |
| `weather_forecast/` | Multi-day atmospheric pressure and temperature forecasting | AFNO |

Each project folder contains:
- `solver.py` — traditional finite-difference solver (ground truth)
- `train.py` — train the neural surrogate
- `compare.py` — benchmark neural vs solver (speed + accuracy)
- `benchmark.py` — full head-to-head comparison of all methods (chip_thermal only)
