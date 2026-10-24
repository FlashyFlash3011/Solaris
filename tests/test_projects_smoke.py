"""End-to-end smoke tests: every project and example runs with tiny settings.

Each script is executed as a subprocess on the CPU inside a temporary
directory (all outputs land there).  They check that the scripts run and
that train → compare/calibrate pipelines fit together, not accuracy.

These take a couple of minutes; skip them locally with ``-m "not slow"``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
P = REPO / "projects"
E = REPO / "examples"

pytestmark = pytest.mark.slow

# (name, [(script, args), ...]) — steps of one pipeline run in order, sharing cwd.
PIPELINES = {
    "navier_stokes": [
        (
            P / "navier_stokes/train.py",
            "--n_sims 3 --epochs 1 --H 16 --W 16 --modes 4 --hidden 8 "
            "--n_layers 1 --n_steps 4 --cache d.npz --checkpoint_dir ck",
        ),
        (P / "navier_stokes/compare.py", "--checkpoint_dir ck --n_test 1 --output ns.png"),
    ],
    "wave_equation": [
        (
            P / "wave_equation/train.py",
            "--n_sims 3 --epochs 1 --n_steps 20 --n_snapshots 4 "
            "--modes 4 --hidden 8 --n_layers 1 --cache d.npz --checkpoint_dir ck",
        ),
        (P / "wave_equation/compare.py", "--checkpoint_dir ck --n_test 1 --output wave.png"),
    ],
    "water_heat_diffusion": [
        (
            P / "water_heat_diffusion/train.py",
            "--device cpu --resolution 16 --n_sims 4 "
            "--n_times 2 --epochs 1 --modes 4 --hidden 8 --n_layers 1 --cache d.npz "
            "--checkpoint_dir ck",
        ),
        (
            P / "water_heat_diffusion/compare.py",
            "--device cpu --checkpoint_dir ck --output water.png",
        ),
    ],
    "weather_forecast": [
        (
            P / "weather_forecast/train.py",
            "--device cpu --nlat 16 --nlon 32 --n_sims 4 "
            "--n_days 2 --epochs 1 --hidden 16 --n_layers 1 --cache d.npz --checkpoint_dir ck",
        ),
        (
            P / "weather_forecast/compare.py",
            "--device cpu --checkpoint_dir ck --output weather.png",
        ),
    ],
    "hurricane_flood": [
        (
            P / "hurricane_flood/train.py",
            "--device cpu --H 16 --W 16 --n_sims 10 --n_hours 4 "
            "--n_snapshots 4 --epochs 1 --modes 4 --hidden_flood 8 --hidden_wind 8 --n_layers 1 "
            "--cache d.npz --checkpoint_dir ck",
        ),
        (
            P / "hurricane_flood/compare.py",
            "--device cpu --checkpoint_dir ck --n 1 --n_plot 1 --output hur.png",
        ),
    ],
    "chip_thermal": [
        (
            P / "chip_thermal/train.py",
            "--device cpu --resolution 32 --n_train 24 --n_val 8 "
            "--epochs 1 --modes 4 --hidden 8 --n_layers 1 --data d.npz --checkpoint_dir ck",
        ),
        (
            P / "chip_thermal/train.py",
            "--device cpu --resolution 32 --n_train 24 --n_val 8 "
            "--epochs 1 --modes 4 --hidden 8 --n_layers 1 --model constrained --data d.npz "
            "--checkpoint_dir ck",
        ),
        (
            P / "chip_thermal/train.py",
            "--device cpu --resolution 32 --n_train 24 --n_val 8 "
            "--epochs 1 --modes 4 --hidden 8 --n_layers 1 --model residual --data d.npz "
            "--checkpoint_dir ck",
        ),
        (
            P / "chip_thermal/compare.py",
            "--device cpu --checkpoint ck/best_fno.pt --n_batch 4 --output chip.png",
        ),
        (
            P / "chip_thermal/calibrate.py",
            "--device cpu --resolution 32 --model constrained "
            "--checkpoint ck/best_constrained.pt --cal_size 20 --test_size 10",
        ),
        (REPO / "benchmarks/chip_thermal.py", "--device cpu --checkpoint_dir ck --n 4"),
    ],
    "examples": [
        (
            E / "train_fno_darcy.py",
            "--epochs 1 --n_samples 16 --resolution 16 --hidden 8 "
            "--modes 4 --n_layers 1 --checkpoint_dir ck",
        ),
        (E / "train_pinn_poisson.py", "--epochs 5 --n_collocation 64 --n_boundary 16"),
        (
            E / "train_conservative_diffusion.py",
            "--res 16 --n_train 8 --n_val 4 --epochs 1 --hidden 8 --modes 4 --n_layers 1",
        ),
        (
            E / "train_multiscale_fno.py",
            "--res 16 --n_train 8 --n_val 4 --epochs 1 --hidden 8 --modes 4 --n_layers 1",
        ),
        (E / "demo_conformal.py", "--res 16 --n_train 8 --n_cal 10 --n_test 4 --epochs 1"),
        (
            E / "demo_coupled.py",
            "--res 16 --n_train 8 --n_val 4 --epochs 1 --hidden 8 --modes 4 --n_layers 1",
        ),
        (
            E / "demo_residual_correction.py",
            "--res 16 --n_train 8 --n_val 4 --epochs 1 "
            "--hidden 8 --modes 4 --n_layers 1 --coarse_iters 5",
        ),
    ],
}


@pytest.mark.parametrize("name", list(PIPELINES))
def test_pipeline_runs(name, tmp_path):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    }
    for script, args in PIPELINES[name]:
        run = subprocess.run(
            [sys.executable, str(script), *args.split()],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        out = run.stdout + run.stderr
        assert run.returncode == 0, f"{script.name} {args}\n{out[-3000:]}"
        assert "Traceback" not in out, f"{script.name}\n{out[-3000:]}"
