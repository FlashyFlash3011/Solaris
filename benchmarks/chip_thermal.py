# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark trained chip-thermal surrogates against the FD solver.

Evaluates every checkpoint found in a chip_thermal checkpoint directory
(best_fno.pt, best_constrained.pt, best_residual.pt) on freshly generated
layouts that no model has seen, and reports:

* rel-L2 of the temperature *rise* above the 40 °C boundary — the part the
  model has to predict (the headline accuracy metric);
* rel-L2 of absolute temperature (looks smaller because every pixel includes
  the 40 °C offset; reported for comparison with older numbers);
* worst absolute error [°C] and worst boundary deviation from 40 °C;
* time per layout: scipy sparse FD solve (CPU, one layout at a time) vs the
  model (batched on the chosen device, including host↔device copies).

    python benchmarks/chip_thermal.py --checkpoint_dir projects/chip_thermal/checkpoints/v2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CHIP = Path(__file__).resolve().parent.parent / "projects" / "chip_thermal"
sys.path.insert(0, str(CHIP))

from solver import T_AMBIENT, chip_floorplan_power_map, solve_heat_fd  # noqa: E402
from train import build_model, make_coarse_solver  # noqa: E402


def load(path: Path, device, stats):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kind = ckpt.get("model_type", "fno")
    q_mean, q_std, t_mean, t_std = (float(stats[k]) for k in ("Q_mean", "Q_std", "T_mean", "T_std"))
    coarse = make_coarse_solver(q_mean, q_std, t_mean, t_std) if kind == "residual" else None
    model = build_model(
        kind,
        ckpt.get("hidden_channels", 64),
        ckpt.get("n_layers", 4),
        ckpt.get("modes", 16),
        coarse_solver=coarse,
        boundary_value=(T_AMBIENT - t_mean) / t_std,
        norm=ckpt.get("norm", "instance"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return kind, model.to(device).eval(), ckpt.get("resolution", 128)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default=str(CHIP / "checkpoints"))
    p.add_argument("--n", type=int, default=1000, help="fresh layouts to evaluate")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=9999)
    p.add_argument("--json", default=None, help="also write results to this JSON file")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    stats = np.load(ckpt_dir / "norm_stats.npz")
    paths = [ckpt_dir / f"best_{k}.pt" for k in ("fno", "constrained", "residual")]
    models = [load(pth, device, stats) for pth in paths if pth.exists()]
    if not models:
        raise SystemExit(f"no best_*.pt checkpoints in {ckpt_dir}")
    res = models[0][2]

    rng = np.random.default_rng(args.seed)
    Q = np.stack([chip_floorplan_power_map(res, res, rng) for _ in range(args.n)])
    t0 = time.perf_counter()
    T = np.stack([solve_heat_fd(q)[0] for q in Q])
    fd_ms = (time.perf_counter() - t0) / args.n * 1e3

    q_mean, q_std, t_mean, t_std = (float(stats[k]) for k in ("Q_mean", "Q_std", "T_mean", "T_std"))
    Q_t = torch.as_tensor(((Q - q_mean) / q_std)[:, None], dtype=torch.float32)
    rise = (T - T_AMBIENT).reshape(args.n, -1)

    rows = []
    for kind, model, _ in models:
        with torch.no_grad():
            model(Q_t[: args.batch_size].to(device))  # warm-up
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = [
                model(Q_t[i : i + args.batch_size].to(device)).cpu()
                for i in range(0, args.n, args.batch_size)
            ]
            if device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.n * 1e3
        pred = torch.cat(out).numpy()[:, 0] * t_std + t_mean
        err = (pred - T).reshape(args.n, -1)
        edge = np.concatenate([pred[:, 0], pred[:, -1], pred[:, :, 0], pred[:, :, -1]], axis=1)
        rel_rise = np.linalg.norm(err, axis=1) / np.linalg.norm(rise, axis=1)
        rel_abs = np.linalg.norm(err, axis=1) / np.linalg.norm(T.reshape(args.n, -1), axis=1)
        rows.append(
            {
                "model": kind,
                "rel_l2_rise_mean": float(rel_rise.mean()),
                "rel_l2_rise_max": float(rel_rise.max()),
                "rel_l2_abs_mean": float(rel_abs.mean()),
                "max_abs_err_C": float(np.abs(err).max()),
                "max_boundary_dev_C": float(np.abs(edge - T_AMBIENT).max()),
                "ms_per_layout": ms,
                "speedup_vs_fd": fd_ms / ms,
            }
        )

    print(f"{args.n} fresh {res}x{res} layouts | FD solver {fd_ms:.2f} ms/layout (CPU)")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    hdr = "| model | rel-L2 (rise) mean / max | rel-L2 (abs T) | max err °C | boundary dev °C | ms/layout | speed-up |"  # noqa: E501
    print(hdr)
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {r['rel_l2_rise_mean']:.2%} / {r['rel_l2_rise_max']:.2%} | "
            f"{r['rel_l2_abs_mean']:.3%} | {r['max_abs_err_C']:.2f} | "
            f"{r['max_boundary_dev_C']:.3f} | {r['ms_per_layout']:.2f} | "
            f"{r['speedup_vs_fd']:.1f}x |"
        )
    if args.json:
        Path(args.json).write_text(json.dumps({"fd_ms": fd_ms, "models": rows}, indent=2))


if __name__ == "__main__":
    main()
