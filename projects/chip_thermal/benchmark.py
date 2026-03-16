# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Chip Thermal Benchmark — Traditional Solver vs Neural Methods.

Generates data, trains all methods, then benchmarks each on fresh test cases.

Methods compared
----------------
1. FD Solver (converged)      — ground truth, slow
2. FD Solver (coarse, 20 it)  — fast but inaccurate baseline
3. FNO Surrogate              — pure neural, no physics knowledge
4. NeuralResidualCorrector    — coarse solver + neural error correction
5. ConstrainedFNO             — FNO with hard conservation enforcement

Outputs
-------
- Terminal table: accuracy, speed, speedup vs FD solver
- benchmark_results.png: power map / FD truth / each method / error maps

Usage
-----
    cd projects/chip_thermal
    python3.11 benchmark.py                        # quick (300 train samples)
    python3.11 benchmark.py --n_train 800 --epochs 50   # publication quality
    python3.11 benchmark.py --device cuda               # GPU
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from physicsnemo.models import FNO, ConstrainedFNO, NeuralResidualCorrector
from physicsnemo.metrics import relative_l2_error
from physicsnemo.utils import get_logger
from solver import random_power_map, solve_heat


# ── Data ──────────────────────────────────────────────────────────────────────

def generate_dataset(n, res, seed=0):
    log = get_logger("data")
    log.info(f"Generating {n} samples at {res}×{res} (converged FD solver) …")
    rng = np.random.default_rng(seed)
    Q_all, T_all = [], []
    t0 = time.perf_counter()
    for i in range(n):
        Q = random_power_map(res, res, rng=rng)
        T, _, _ = solve_heat(Q, max_iter=10_000, tol=1e-4)
        Q_all.append(Q); T_all.append(T)
        if (i + 1) % max(1, n // 5) == 0:
            log.info(f"  {i+1}/{n}  ({time.perf_counter()-t0:.1f}s)")
    Q_arr = np.stack(Q_all)[:, None].astype(np.float32)
    T_arr = np.stack(T_all)[:, None].astype(np.float32)
    log.info(f"Done in {time.perf_counter()-t0:.1f}s")
    return Q_arr, T_arr


def normalise(arr):
    m, s = arr.mean(), arr.std() + 1e-8
    return (arr - m) / s, m, s


# ── Coarse solver wrapper (for NeuralResidualCorrector) ───────────────────────

def make_coarse_solver(n_iter, device):
    def solver(x_t: torch.Tensor) -> torch.Tensor:
        x_np = x_t.detach().cpu().numpy()
        out = []
        for i in range(len(x_np)):
            T, _, _ = solve_heat(x_np[i, 0], max_iter=n_iter, tol=1e-6)
            out.append(T.astype(np.float32))
        return torch.as_tensor(np.stack(out)[:, None], device=x_t.device)
    return solver


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_dl, val_dl, device, epochs, lr, name, log):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_l2, best_state = float("inf"), None

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        val_l2 = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_l2 += relative_l2_error(model(x), y).item() * len(x)
        val_l2 /= len(val_dl.dataset)

        if val_l2 < best_l2:
            best_l2 = val_l2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % max(1, epochs // 4) == 0:
            log.info(f"  [{name}] ep {ep}/{epochs}  val-rel-L2={val_l2:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    return best_l2


# ── Benchmark ─────────────────────────────────────────────────────────────────

def time_inference(fn, x, device, n_warmup=3, n_repeat=10):
    """Measure median inference time in milliseconds."""
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            fn(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run(args):
    log = get_logger("benchmark")
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    log.info(f"Device: {device}")

    # ── 1. Generate / load dataset ──
    cache = Path("data/benchmark_dataset.npz")
    total = args.n_train + args.n_val + args.n_test
    if cache.exists():
        log.info(f"Loading cached dataset from {cache}")
        d = np.load(cache)
        Q_arr, T_arr = d["Q"], d["T"]
        if len(Q_arr) < total:
            Q_arr, T_arr = None, None
            log.info("Cache too small, regenerating")
    else:
        Q_arr, T_arr = None, None

    if Q_arr is None:
        Q_arr, T_arr = generate_dataset(total, args.res)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, Q=Q_arr, T=T_arr)

    Q_norm, Q_mean, Q_std = normalise(Q_arr)
    T_norm, T_mean, T_std = normalise(T_arr)

    # Split
    n_tr, n_val = args.n_train, args.n_val
    Q_t = torch.as_tensor(Q_norm)
    T_t = torch.as_tensor(T_norm)

    train_dl = DataLoader(TensorDataset(Q_t[:n_tr], T_t[:n_tr]),
                          batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(TensorDataset(Q_t[n_tr:n_tr+n_val], T_t[n_tr:n_tr+n_val]),
                          batch_size=args.batch_size)
    # Test set (raw, not normalised — for comparison with FD solver output in °C)
    Q_test_raw = Q_arr[n_tr+n_val:]   # (n_test, 1, H, W)
    T_test_raw = T_arr[n_tr+n_val:]   # ground truth in solver units
    Q_test_t   = Q_t[n_tr+n_val:].to(device)
    T_test_t   = T_t[n_tr+n_val:].to(device)

    h, w = args.res, args.res
    hidden, modes, layers = args.hidden, args.modes, args.n_layers

    # ── 2. Build models ──
    coarse_solver = make_coarse_solver(args.coarse_iters, device)

    fno = FNO(1, 1, hidden, layers, modes, dim=2).to(device)
    corrector = NeuralResidualCorrector(
        solver=coarse_solver, in_channels=1, out_channels=1,
        solver_out_channels=1, hidden_channels=hidden,
        n_layers=layers, modes=modes,
    ).to(device)
    cfno = ConstrainedFNO(1, 1, hidden, layers, modes,
                          constraint="conservative").to(device)

    log.info(f"\nModel parameter counts:")
    log.info(f"  FNO:                  {fno.num_parameters():,}")
    log.info(f"  NeuralResidualCorr:   {corrector.num_parameters():,}")
    log.info(f"  ConstrainedFNO:       {cfno.num_parameters():,}")

    # ── 3. Train neural models ──
    log.info(f"\n{'='*55}")
    log.info("Training FNO …")
    fno_val_l2 = train_model(fno, train_dl, val_dl, device,
                              args.epochs, args.lr, "FNO", log)

    log.info("\nTraining NeuralResidualCorrector …")
    corr_val_l2 = train_model(corrector, train_dl, val_dl, device,
                               args.epochs, args.lr, "Corrector", log)

    log.info("\nTraining ConstrainedFNO …")
    cfno_val_l2 = train_model(cfno, train_dl, val_dl, device,
                               args.epochs, args.lr, "ConstFNO", log)

    # ── 4. Benchmark on test set ──
    log.info(f"\n{'='*55}")
    log.info(f"Benchmarking on {args.n_test} fresh test samples …\n")
    rng = np.random.default_rng(9999)

    # Per-sample results
    results = {
        "FD (converged)":       {"times": [], "rel_l2s": []},
        f"FD (coarse {args.coarse_iters} it)": {"times": [], "rel_l2s": []},
        "FNO surrogate":        {"times": [], "rel_l2s": []},
        "ResidualCorrector":    {"times": [], "rel_l2s": []},
        "ConstrainedFNO":       {"times": [], "rel_l2s": []},
    }

    samples_to_plot = []

    for i in range(args.n_test):
        Q_raw = Q_test_raw[i, 0]   # (H, W) numpy
        T_ref  = T_test_raw[i, 0]  # ground truth

        # FD converged
        T_conv, _, t_conv = solve_heat(Q_raw, max_iter=10_000, tol=1e-4)
        results["FD (converged)"]["times"].append(t_conv * 1000)
        results["FD (converged)"]["rel_l2s"].append(0.0)  # reference

        # FD coarse
        _, _, t_coarse = solve_heat(Q_raw, max_iter=args.coarse_iters, tol=1e-6)
        T_coarse, _, _ = solve_heat(Q_raw, max_iter=args.coarse_iters, tol=1e-6)
        rel_l2_coarse = (np.linalg.norm(T_coarse[1:-1,1:-1] - T_ref[1:-1,1:-1])
                         / (np.linalg.norm(T_ref[1:-1,1:-1]) + 1e-8))
        results[f"FD (coarse {args.coarse_iters} it)"]["times"].append(t_coarse * 1000)
        results[f"FD (coarse {args.coarse_iters} it)"]["rel_l2s"].append(rel_l2_coarse)

        # Neural methods — normalised input
        Q_norm_i = (Q_raw - Q_mean) / Q_std
        x_i = torch.as_tensor(Q_norm_i[None, None], dtype=torch.float32, device=device)

        def denorm(t): return t.cpu().numpy()[0, 0] * T_std + T_mean
        def rel_l2_np(pred_np):
            return (np.linalg.norm(pred_np[1:-1,1:-1] - T_ref[1:-1,1:-1])
                    / (np.linalg.norm(T_ref[1:-1,1:-1]) + 1e-8))

        for name, model in [("FNO surrogate", fno),
                             ("ResidualCorrector", corrector),
                             ("ConstrainedFNO", cfno)]:
            t_ms = time_inference(model, x_i, device)
            with torch.no_grad():
                T_pred_np = denorm(model(x_i))
            results[name]["times"].append(t_ms)
            results[name]["rel_l2s"].append(rel_l2_np(T_pred_np))

        if i < args.n_plot:
            with torch.no_grad():
                samples_to_plot.append({
                    "Q": Q_raw,
                    "T_ref": T_ref,
                    "T_coarse": T_coarse,
                    "T_fno": denorm(fno(x_i)),
                    "T_corr": denorm(corrector(x_i)),
                    "T_cfno": denorm(cfno(x_i)),
                })

    # ── 5. Print summary table ──
    log.info(f"\n{'Method':<30} {'Avg time':>10} {'Speedup':>10} {'Avg rel-L2':>12} {'Max rel-L2':>12}")
    log.info("-" * 78)
    fd_avg_ms = np.mean(results["FD (converged)"]["times"])
    for name, r in results.items():
        avg_t  = np.mean(r["times"])
        speedup = fd_avg_ms / avg_t
        avg_l2  = np.mean(r["rel_l2s"])
        max_l2  = np.max(r["rel_l2s"])
        unit = "ms" if avg_t < 1000 else "s "
        t_str = f"{avg_t:.1f}{unit}" if avg_t < 1000 else f"{avg_t/1000:.2f}s"
        log.info(f"  {name:<28} {t_str:>10} {speedup:>9.0f}× {avg_l2:>12.4f} {max_l2:>12.4f}")

    # Conservation violation check
    log.info("\nConservation violation (ConstrainedFNO should be ~0):")
    viols = []
    with torch.no_grad():
        for x, _ in val_dl:
            x = x.to(device)
            pred = cfno(x)
            src_int = x[:, 0].sum(dim=[-2, -1])
            out_int = pred[:, 0].sum(dim=[-2, -1])
            viols.append(((src_int - out_int).abs() / (src_int.abs() + 1e-8)).cpu())
    viol_mean = torch.cat(viols).mean().item()
    log.info(f"  ConstrainedFNO conservation violation: {viol_mean:.2e}")
    log.info(f"  (Standard FNO would show ~0.1–1.0 violation)")

    # ── 6. Plot ──
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_rows = len(samples_to_plot)
        cols = ["Power Map Q", f"FD Solver\n(ground truth)", f"FD Coarse\n({args.coarse_iters} iter)",
                "FNO Surrogate", "ResidualCorrector", "ConstrainedFNO"]
        n_cols = len(cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
        if n_rows == 1:
            axes = axes[None, :]  # ensure 2D

        for row, s in enumerate(samples_to_plot):
            T_ref = s["T_ref"]
            vmin, vmax = T_ref.min(), T_ref.max()
            images = [s["Q"], T_ref, s["T_coarse"], s["T_fno"], s["T_corr"], s["T_cfno"]]
            cmaps = ["hot", "inferno", "inferno", "inferno", "inferno", "inferno"]
            # For error columns (methods), show |error| instead
            for col_i, (img, cmap, title) in enumerate(zip(images, cmaps, cols)):
                ax = axes[row, col_i]
                if col_i >= 2:  # show absolute error vs ground truth
                    err = np.abs(img - T_ref)
                    rl2 = np.linalg.norm(err[1:-1,1:-1]) / (np.linalg.norm(T_ref[1:-1,1:-1]) + 1e-8)
                    im = ax.imshow(err, cmap="RdBu_r", origin="lower",
                                   vmin=0, vmax=np.abs(T_ref).max() * 0.15)
                    ax.set_title(f"{title}\nrel-L2={rl2:.4f}", fontsize=8)
                elif col_i == 0:
                    im = ax.imshow(img, cmap=cmap, origin="lower")
                    ax.set_title(title, fontsize=8)
                else:
                    im = ax.imshow(img, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
                    ax.set_title(title, fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis("off")

        fd_t = np.mean(results["FD (converged)"]["times"])
        fno_t = np.mean(results["FNO surrogate"]["times"])
        corr_t = np.mean(results["ResidualCorrector"]["times"])
        fig.suptitle(
            f"Chip Thermal Benchmark  |  "
            f"FD: {fd_t:.0f}ms  |  FNO: {fno_t:.1f}ms ({fd_t/fno_t:.0f}×)  |  "
            f"Corrector: {corr_t:.1f}ms ({fd_t/corr_t:.0f}×)\n"
            f"FNO rel-L2: {np.mean(results['FNO surrogate']['rel_l2s']):.4f}  |  "
            f"Corrector rel-L2: {np.mean(results['ResidualCorrector']['rel_l2s']):.4f}  |  "
            f"ConstFNO rel-L2: {np.mean(results['ConstrainedFNO']['rel_l2s']):.4f}",
            fontsize=10, fontweight="bold",
        )

        out = Path("results/benchmark_results.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        log.info(f"\nFigure saved → {out}")
        plt.close(fig)
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--res",          type=int,   default=64)
    p.add_argument("--n_train",      type=int,   default=300)
    p.add_argument("--n_val",        type=int,   default=100)
    p.add_argument("--n_test",       type=int,   default=20)
    p.add_argument("--n_plot",       type=int,   default=3)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden",       type=int,   default=32)
    p.add_argument("--modes",        type=int,   default=12)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--coarse_iters", type=int,   default=20,
                   help="Iterations for the coarse FD solver used in ResidualCorrector")
    run(p.parse_args())
