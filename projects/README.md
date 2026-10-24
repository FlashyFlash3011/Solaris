# projects/

End-to-end demo projects. Each has a traditional solver that generates the
training data (and the ground truth), a training script for the neural
surrogate, and a script that compares the two.

| Project | Physics problem | Models |
|---|---|---|
| `chip_thermal/` | Steady 2-D heat conduction: chip temperature from its power map | FNO, ConstrainedFNO (fixed 40 °C boundary), NeuralResidualCorrector, conformal intervals |
| `hurricane_flood/` | 24-hour storm surge (linear shallow-water equations) | CoupledOperator (flood + wind), flood depth ≥ 0 and = 0 over ocean, conformal intervals on held-out storms |
| `navier_stokes/` | 2-D incompressible flow (vorticity form), periodic | FNO |
| `wave_equation/` | 2-D wave propagation | time-conditioned FNO |
| `water_heat_diffusion/` | Time-dependent 2-D heat diffusion | time-conditioned FNO |
| `weather_forecast/` | Toy atmospheric advection, multi-day forecasts | AFNO |

Each project folder contains:
- `solver.py` (or `data_gen.py`) — traditional solver, used as ground truth
- `train.py` — trains the surrogate; datasets are cached keyed by their
  generation settings, and train/validation splits are by whole simulation
- `compare.py` — surrogate vs solver: accuracy and speed
- chip_thermal also has `calibrate.py` (conformal intervals); the
  repository-level `benchmarks/chip_thermal.py` evaluates all its models

Install what the demos need with `pip install -e ".[demos]"`. Every project's
train → compare pipeline runs in CI with tiny settings
(`tests/test_projects_smoke.py`).
