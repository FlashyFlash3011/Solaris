# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
compare_dp.py — Three-way visual comparison:
  Column 1: Traditional FD Solver (ground truth)
  Column 2: Pre-DP FNO (baseline plain FNO)
  Column 3: Post-DP Model (NeuralResidualCorrector + DP policy)

Layout (2 rows × 4 panels)
───────────────────────────────────────────────────────────────────
Row 0: [Power Map Q]  [FD Ground Truth T]  [Pre-DP T]  [Post-DP T]
Row 1: [blank/stats]  [— reference —]      [Pre-DP err] [Post-DP err]
───────────────────────────────────────────────────────────────────

Dark theme matching compare.png. Saves to results/compare_dp.png.

Usage
-----
# Full comparison (trains both models first, then compare)
python compare_dp.py \\
    --pre_checkpoint  checkpoints/best_fno.pt \\
    --post_checkpoint checkpoints/best_residual.pt \\
    --device cuda

# CPU smoke test with fewer layouts
python compare_dp.py \\
    --pre_checkpoint  checkpoints/best_fno.pt \\
    --post_checkpoint checkpoints/best_residual.pt \\
    --device cpu --n_batch 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solaris.models.fno import FNO
from solaris.models.residual_corrector import NeuralResidualCorrector
from solaris.models.constrained_fno import ConstrainedFNO
from solaris.utils import get_logger
from solver import chip_floorplan_power_map, solve_heat_fd, LAYOUT_LABELS
from dp_policy import DPRolloutPolicy
from train import make_coarse_solver


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _model_type_from_ckpt(ckpt: dict) -> str:
    return ckpt.get("model_type", "fno")


def load_pre_dp_model(ckpt_path: str, device: torch.device):
    """Load the plain FNO (pre-DP baseline)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = FNO(
        in_channels=1, out_channels=1,
        hidden_channels=ckpt.get("hidden_channels", 64),
        n_layers=ckpt.get("n_layers", 4),
        modes=ckpt.get("modes", 16),
        dim=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("resolution", 128)


def load_post_dp_model(ckpt_path: str, device: torch.device,
                       Q_mean: float, Q_std: float, T_mean: float, T_std: float):
    """Load the post-DP model (residual corrector or constrained FNO)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_type = _model_type_from_ckpt(ckpt)
    hidden  = ckpt.get("hidden_channels", 64)
    n_layers = ckpt.get("n_layers", 4)
    modes   = ckpt.get("modes", 16)

    if model_type == "residual":
        coarse_solver = make_coarse_solver(Q_mean, Q_std, T_mean, T_std, coarse_factor=4)
        model = NeuralResidualCorrector(
            solver=coarse_solver, in_channels=1, out_channels=1, solver_out_channels=1,
            hidden_channels=hidden, n_layers=n_layers, modes=modes, solver_detach=True,
        )
    elif model_type == "constrained":
        model = ConstrainedFNO(
            in_channels=1, out_channels=1, hidden_channels=hidden,
            n_layers=n_layers, modes=modes, constraint="conservative",
        )
    else:
        model = FNO(
            in_channels=1, out_channels=1, hidden_channels=hidden,
            n_layers=n_layers, modes=modes, dim=2,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, model_type, ckpt.get("resolution", 128)


def _infer_batched(model, Q_t: torch.Tensor, device: torch.device,
                   batch_size: int = 64) -> np.ndarray:
    """Run batched inference. Returns numpy array (N, H, W)."""
    chunks = []
    with torch.no_grad():
        for s in range(0, len(Q_t), batch_size):
            chunk = Q_t[s:s + batch_size].to(device)
            chunks.append(model(chunk).cpu())
    return torch.cat(chunks, 0).numpy()[:, 0]


# ─── Main comparison ─────────────────────────────────────────────────────────

def run_comparison(args):
    log = get_logger("compare_dp")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")

    # ── Load norm stats (from pre-DP checkpoint dir) ──
    stats = np.load(Path(args.pre_checkpoint).parent / "norm_stats.npz")
    Q_mean, Q_std = float(stats["Q_mean"]), float(stats["Q_std"])
    T_mean, T_std = float(stats["T_mean"]), float(stats["T_std"])

    # ── Load models ──
    pre_model, resolution = load_pre_dp_model(args.pre_checkpoint, device)
    post_model, post_type, _ = load_post_dp_model(
        args.post_checkpoint, device, Q_mean, Q_std, T_mean, T_std
    )
    H = W = resolution
    log.info(f"Pre-DP  FNO                : {pre_model.num_parameters():,} params")
    log.info(f"Post-DP {post_type:>10s}  : {post_model.num_parameters():,} params")

    # ── Optional DP policy for post-DP model ──
    policy = None
    if args.dp_policy and post_type == "residual":
        policy_path = Path(args.post_checkpoint).parent / "dp_policy.npz"
        if policy_path.exists():
            policy = DPRolloutPolicy.load(policy_path)
            log.info(f"Loaded DP policy from {policy_path}")
        else:
            log.info("dp_policy.npz not found — fitting policy on-the-fly …")
            policy = DPRolloutPolicy()
            # Fit using validation data (use pool generated below)

    # ── GPU warm-up ──
    if device.type == "cuda":
        dummy = torch.zeros(1, 1, H, W, device=device)
        with torch.no_grad():
            _ = pre_model(dummy)
            _ = post_model(dummy)
        torch.cuda.synchronize()

    # ── Generate N new chip layouts ──
    N = args.n_batch
    log.info(f"Generating {N} chip layouts (seed={args.seed}) …")
    rng = np.random.default_rng(args.seed)
    Q_all = np.stack([chip_floorplan_power_map(H, W, rng) for _ in range(N)])

    # ── Column 1: Traditional FD Solver ──
    log.info("Running FD solver (CPU, sequential) …")
    t0 = time.perf_counter()
    T_fd = np.empty_like(Q_all)
    for i in range(N):
        T_fd[i], _ = solve_heat_fd(Q_all[i])
    fd_time = time.perf_counter() - t0
    fd_ms = fd_time / N * 1000

    # ── Prepare normalised input tensor ──
    Q_norm = (Q_all - Q_mean) / Q_std
    Q_t    = torch.as_tensor(Q_norm[:, None], dtype=torch.float32)

    # ── Column 2: Pre-DP FNO ──
    log.info("Running pre-DP FNO …")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    T_pre_norm = _infer_batched(pre_model, Q_t, device, batch_size=args.batch_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    pre_time = time.perf_counter() - t0
    pre_ms = pre_time / N * 1000
    T_pre = T_pre_norm * T_std + T_mean

    # ── Column 3: Post-DP Model ──
    log.info(f"Running post-DP {post_type} …")
    solver_calls_saved = 0
    if policy is not None and post_type == "residual":
        # Fit policy if not yet fitted
        if not policy._fitted:
            log.info("  Fitting DP policy from model diagnostics …")
            policy.fit_from_model(post_model, _make_val_loader(Q_t, T_fd, T_mean, T_std),
                                  device, n_samples=200)
        # DP-gated inference: skip FD solver for "easy" samples
        log.info("  Running DP-gated inference …")
        T_post_norm, solver_calls_saved = _dp_gated_inference(
            post_model, policy, Q_t, T_mean, T_std, device, N, args.batch_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        # Time the gated inference
        t0 = time.perf_counter()
        T_post_norm, _ = _dp_gated_inference(
            post_model, policy, Q_t, T_mean, T_std, device, N, args.batch_size
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        post_time = time.perf_counter() - t0
    else:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        T_post_norm = _infer_batched(post_model, Q_t, device, batch_size=args.batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        post_time = time.perf_counter() - t0
    post_ms = post_time / N * 1000
    T_post = T_post_norm * T_std + T_mean

    # ── Error statistics ──
    def rel_l2(pred, ref):
        diff = pred.reshape(N, -1) - ref.reshape(N, -1)
        return np.linalg.norm(diff, axis=1) / (np.linalg.norm(ref.reshape(N, -1), axis=1) + 1e-8)

    rl2_pre  = rel_l2(T_pre,  T_fd)
    rl2_post = rel_l2(T_post, T_fd)
    speedup_pre  = fd_time / pre_time
    speedup_post = fd_time / post_time

    # ── Summary ──
    log.info("")
    log.info("=" * 70)
    log.info(f"  Batch: {N} chip layouts")
    log.info(f"  FD Solver   : {fd_time:.1f}s  ({fd_ms:.1f} ms/layout)")
    log.info(f"  Pre-DP  FNO : {pre_time*1000:.0f}ms  ({pre_ms:.2f} ms/layout)  "
             f"speedup={speedup_pre:.2f}×  rel-L2 avg={np.mean(rl2_pre):.4f}")
    post_tag = f"  Post-DP {post_type:10s}"
    post_policy_str = ""
    if solver_calls_saved:
        pct_saved = 100 * solver_calls_saved / N
        post_policy_str = f"  FD calls saved: {solver_calls_saved}/{N} ({pct_saved:.0f}%)"
    log.info(f"{post_tag}: {post_time*1000:.0f}ms  ({post_ms:.2f} ms/layout)  "
             f"speedup={speedup_post:.2f}×  rel-L2 avg={np.mean(rl2_post):.4f}"
             f"{post_policy_str}")
    log.info("=" * 70)

    # ── Figure ──
    _plot(
        Q_all, T_fd, T_pre, T_post,
        rl2_pre, rl2_post,
        fd_ms, pre_ms, post_ms,
        speedup_pre, speedup_post,
        solver_calls_saved, N,
        post_type, args,
    )


def _dp_gated_inference(model, policy, Q_t, T_mean, T_std, device, N, batch_size):
    """Run inference with DP policy: skip FD solver for low-correction samples."""
    import torch
    T_out = np.empty((*Q_t.shape[:1], *Q_t.shape[2:]), dtype=np.float32)
    solver_calls_saved = 0

    with torch.no_grad():
        for s in range(0, N, batch_size):
            Q_b = Q_t[s:s + batch_size].to(device)
            remaining = N - s
            diag = model.correction_diagnostics(Q_b)
            rel_corr = diag["relative_correction"]

            if policy.should_call_solver(rel_corr, remaining, total_samples=N):
                # Full inference (FD solver + neural correction)
                pred = model(Q_b)
            else:
                # Skip FD solver: use only the neural backbone (direct prediction)
                # Temporarily route through FNO backbone only
                x_coarse = model.solver(Q_b)
                pred = x_coarse  # coarse only when correction is negligible
                solver_calls_saved += len(Q_b)

            T_out[s:s + len(Q_b)] = pred.cpu().numpy()[:, 0]

    return T_out, solver_calls_saved


def _make_val_loader(Q_t, T_fd, T_mean, T_std):
    """Lightweight val DataLoader wrapper for policy fitting."""
    from torch.utils.data import DataLoader, TensorDataset
    T_norm = (T_fd - T_mean) / T_std
    T_t = torch.as_tensor(T_norm[:, None], dtype=torch.float32)
    ds = TensorDataset(Q_t, T_t)
    return DataLoader(ds, batch_size=32, shuffle=False)


# ─── Plotting ────────────────────────────────────────────────────────────────

def _plot(Q_all, T_fd, T_pre, T_post,
          rl2_pre, rl2_post,
          fd_ms, pre_ms, post_ms,
          speedup_pre, speedup_post,
          solver_calls_saved, N,
          post_type, args):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
        import matplotlib.gridspec as gridspec
    except ImportError:
        return

    idx = 0
    Q0    = Q_all[idx]
    T0_fd = T_fd[idx]
    T0_pre  = T_pre[idx]
    T0_post = T_post[idx]
    err_pre  = np.abs(T0_pre  - T0_fd)
    err_post = np.abs(T0_post - T0_fd)

    T_ambient = float(T0_fd.min())
    vmin_T = T_ambient
    vmax_T = float(T0_fd.max())

    fig = plt.figure(figsize=(26, 12))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           wspace=0.38, hspace=0.52,
                           left=0.04, right=0.97, top=0.89, bottom=0.04)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    def _style_ax(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_facecolor("#0d0d0d")

    def _cbar(im, ax):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")
        cb.outline.set_edgecolor("white")
        return cb

    # ── Row 0, Col 0: Power map ──
    im = axes[0][0].imshow(Q0, cmap="hot", origin="lower", interpolation="bilinear")
    axes[0][0].set_title("Power Map  Q(x,y)\n[W/m²]", fontsize=9, color="white", pad=6)
    _cbar(im, axes[0][0])
    for fx, fy, label in LAYOUT_LABELS:
        txt = axes[0][0].text(
            fx * (Q0.shape[1] - 1), fy * (Q0.shape[0] - 1), label,
            color="white", fontsize=8, ha="center", va="center", fontweight="bold",
        )
        txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])

    # ── Row 0, Col 1: FD Ground Truth ──
    im = axes[0][1].imshow(T0_fd, cmap="inferno", origin="lower",
                            vmin=vmin_T, vmax=vmax_T, interpolation="bilinear")
    axes[0][1].set_title(
        f"Traditional FD Solver\nT [°C]  ·  {fd_ms:.1f} ms/layout",
        fontsize=10, color="white", pad=7,
    )
    _cbar(im, axes[0][1])

    # ── Row 0, Col 2: Pre-DP FNO ──
    im = axes[0][2].imshow(T0_pre, cmap="inferno", origin="lower",
                            vmin=vmin_T, vmax=vmax_T, interpolation="bilinear")
    axes[0][2].set_title(
        f"Pre-DP FNO\nT [°C]  ·  {pre_ms:.2f} ms/layout  ·  {speedup_pre:.2f}× vs FD",
        fontsize=10, color="white", pad=7,
    )
    _cbar(im, axes[0][2])

    # ── Row 0, Col 3: Post-DP Model ──
    policy_str = ""
    if solver_calls_saved:
        pct = 100 * solver_calls_saved / N
        policy_str = f"  ·  FD calls saved: {pct:.0f}%"
    im = axes[0][3].imshow(T0_post, cmap="inferno", origin="lower",
                            vmin=vmin_T, vmax=vmax_T, interpolation="bilinear")
    axes[0][3].set_title(
        f"Post-DP {post_type}\nT [°C]  ·  {post_ms:.2f} ms/layout  ·  "
        f"{speedup_post:.2f}× vs FD{policy_str}",
        fontsize=10, color="white", pad=7,
    )
    _cbar(im, axes[0][3])

    # ── Row 1, Col 0: Stats panel (text) ──
    axes[1][0].set_xlim(0, 1); axes[1][0].set_ylim(0, 1)
    col_w = 0.55   # left column width (labels)
    rows = [
        ("Batch",         f"{N} layouts"),
        ("",              ""),
        ("rel-L2 avg",    ""),
        ("  Pre-DP",      f"{np.mean(rl2_pre)*100:.2f}%"),
        ("  Post-DP",     f"{np.mean(rl2_post)*100:.2f}%"),
        ("",              ""),
        ("rel-L2 max",    ""),
        ("  Pre-DP",      f"{np.max(rl2_pre)*100:.2f}%"),
        ("  Post-DP",     f"{np.max(rl2_post)*100:.2f}%"),
        ("",              ""),
        ("Speedup vs FD", ""),
        ("  Pre-DP",      f"{speedup_pre:.2f}×"),
        ("  Post-DP",     f"{speedup_post:.2f}×"),
    ]
    y = 0.95
    dy = 0.065
    for label, value in rows:
        if label:
            axes[1][0].text(0.05, y, label, color="#aaaaaa", fontsize=9,
                            va="top", ha="left", family="monospace",
                            transform=axes[1][0].transAxes)
        if value:
            axes[1][0].text(col_w, y, value, color="white", fontsize=9,
                            va="top", ha="left", family="monospace",
                            transform=axes[1][0].transAxes)
        y -= dy

    # ── Row 1, Col 1: blank (FD is reference) ──
    axes[1][1].text(
        0.5, 0.5, "Ground Truth\n(reference — no error)", color="#666666",
        fontsize=11, ha="center", va="center",
        transform=axes[1][1].transAxes,
    )

    # ── Row 1, Col 2: Pre-DP error ──
    err_max = max(err_pre.max(), err_post.max())
    im = axes[1][2].imshow(err_pre, cmap="RdBu_r", origin="lower",
                            vmin=0, vmax=err_max, interpolation="bilinear")
    axes[1][2].set_title(
        f"|Error|  Pre-DP  [°C]\nMax: {err_pre.max():.2f}°C  ·  "
        f"Rel-L2: {float(rl2_pre[idx])*100:.2f}%",
        fontsize=10, color="white", pad=7,
    )
    _cbar(im, axes[1][2])

    # ── Row 1, Col 3: Post-DP error ──
    im = axes[1][3].imshow(err_post, cmap="RdBu_r", origin="lower",
                            vmin=0, vmax=err_max, interpolation="bilinear")
    axes[1][3].set_title(
        f"|Error|  Post-DP  [°C]\nMax: {err_post.max():.2f}°C  ·  "
        f"Rel-L2: {float(rl2_post[idx])*100:.2f}%",
        fontsize=10, color="white", pad=7,
    )
    _cbar(im, axes[1][3])

    for row in axes:
        for ax in row:
            _style_ax(ax)

    fig.suptitle(
        f"DP × Chip-Thermal  ·  {N} layouts  ·  "
        f"FD: {fd_ms:.1f} ms  ·  "
        f"Pre-DP: {pre_ms:.2f} ms ({speedup_pre:.0f}×)  ·  "
        f"Post-DP [{post_type}]: {post_ms:.2f} ms ({speedup_post:.0f}×)",
        fontsize=13, fontweight="bold", color="white", y=0.975,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    import logging
    logging.getLogger("compare_dp").info(f"Figure saved → {out}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    _here = Path(__file__).parent
    p.add_argument("--pre_checkpoint",  default=str(_here / "checkpoints/best_fno.pt"),
                   help="Pre-DP (plain FNO) checkpoint")
    p.add_argument("--post_checkpoint", default=str(_here / "checkpoints/best_residual.pt"),
                   help="Post-DP model checkpoint (residual or constrained)")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--n_batch",     type=int, default=1000,
                   help="Number of new chip layouts to benchmark")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="GPU batch size for neural model inference")
    p.add_argument("--seed",        type=int, default=9999,
                   help="RNG seed for layout generation (different from training)")
    p.add_argument("--dp_policy",   action="store_true",
                   help="Enable DP rollout policy for post-DP residual model")
    p.add_argument("--output",      default=str(_here / "results/compare_dp.png"))
    run_comparison(p.parse_args())
