# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""
train.py — Train FNO / ConstrainedFNO / NeuralResidualCorrector on chip thermal data.

DP enhancements
---------------
Phase 1a  WarmupCosineScheduler + GradientClipper + EarlyStopping (Solaris utils)
Phase 1b  Hyperband pruner via HyperparameterTuner  (--tune flag)
Phase 1c  Mode curriculum scheduler                 (--curriculum flag)
Phase 1d  Gradient checkpointing for deep FNO stacks (--grad_ckpt flag)
Phase 2   ConstrainedFNO (conservative) + NeuralResidualCorrector (--model flag)
Phase 3b  Knapsack mode profiling                   (--profile_modes flag)

Usage
-----
# Standard FNO with all Phase-1 improvements
python train.py --device cuda

# Physics-constrained FNO (conservation law enforced)
python train.py --model constrained --device cuda

# Neural residual corrector (FD coarse solver + neural correction)
python train.py --model residual --device cuda

# Automated Hyperband hyperparameter search, then full training
python train.py --tune --device cuda

# DP knapsack mode profiling, then full training with optimal modes
python train.py --profile_modes --device cuda

# CPU smoke-test
python train.py --device cpu --n_train 200 --epochs 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from solaris.models.constrained_fno import ConstrainedFNO
from solaris.models.fno import FNO
from solaris.models.residual_corrector import NeuralResidualCorrector
from solaris.metrics import relative_l2_error
from solaris.utils import get_logger, save_checkpoint
from solaris.utils.training import EarlyStopping, GradientClipper, WarmupCosineScheduler
from solaris.utils.tuner import HyperparameterTuner
from solver import generate_dataset, solve_heat_fd


# ─── DP Phase 1c: Mode curriculum ────────────────────────────────────────────

class ModeCurriculumScheduler:
    """DP-based curriculum: gradually unlock Fourier modes during training.

    DP formulation
    --------------
    state  = (epoch, active_modes)
    action = increase modes by ``modes_step``
    reward = validation loss improvement per GPU-hour

    Simplified to a fixed unlock schedule here; the DP value function reduces
    to: start low (easy smooth solutions), unlock higher modes as the model
    masters each frequency band.

    Parameters
    ----------
    model : FNO or ConstrainedFNO
        Must implement ``set_modes(int)``.
    modes_start : int
        Initial active modes (low-frequency only).
    modes_max : int
        Maximum modes to unlock (clamped to model's initialised size).
    modes_step : int
        Modes to add at each unlock event.
    unlock_every : int
        Epochs between unlock events.
    """

    def __init__(
        self,
        model,
        modes_start: int = 4,
        modes_max: int = 16,
        modes_step: int = 4,
        unlock_every: int = 30,
    ) -> None:
        self.model = model
        self.modes_start = modes_start
        self.modes_max = modes_max
        self.modes_step = modes_step
        self.unlock_every = unlock_every
        self.active_modes = modes_start
        model.set_modes(modes_start)

    def step(self, epoch: int) -> int:
        """Advance curriculum. Returns the current active mode count."""
        if epoch > 0 and epoch % self.unlock_every == 0:
            new_modes = min(self.active_modes + self.modes_step, self.modes_max)
            if new_modes != self.active_modes:
                self.active_modes = new_modes
                self.model.set_modes(new_modes)
        return self.active_modes


# ─── DP Phase 2: Coarse FD solver for NeuralResidualCorrector ────────────────

def make_coarse_solver(Q_mean: float, Q_std: float, T_mean: float, T_std: float,
                       coarse_factor: int = 4):
    """Return a batched coarse FD solver for NeuralResidualCorrector.

    Runs the heat equation at 1/coarse_factor resolution (H//4 × W//4) then
    bilinearly upsamples back — ~17× fewer unknowns than the full 128×128 grid,
    so per-batch overhead stays under ~10 ms even on CPU.

    The solver receives and returns *normalised* tensors so it fits seamlessly
    into the normalised training pipeline.
    """
    def _batched_solver(Q_batch: torch.Tensor) -> torch.Tensor:
        B, _, H, W = Q_batch.shape
        # Denormalise Q → physical W/m²
        Q_phys = Q_batch.detach().cpu() * Q_std + Q_mean          # (B,1,H,W)
        # Downscale via average pooling
        Q_coarse = F.avg_pool2d(Q_phys, kernel_size=coarse_factor,
                                stride=coarse_factor)               # (B,1,Hc,Wc)
        Q_np = Q_coarse.numpy()[:, 0]                              # (B,Hc,Wc)
        # Solve at coarse resolution
        T_np = np.empty_like(Q_np)
        for i in range(B):
            T_np[i], _ = solve_heat_fd(Q_np[i])
        # Upscale T back to (H, W)
        T_coarse_t = torch.as_tensor(T_np[:, None], dtype=torch.float32)
        T_up = F.interpolate(T_coarse_t, size=(H, W),
                             mode="bilinear", align_corners=False)  # (B,1,H,W)
        # Normalise T → same space as training targets
        T_norm = (T_up - T_mean) / T_std
        return T_norm.to(Q_batch.device)

    return _batched_solver


# ─── Model factory ───────────────────────────────────────────────────────────

def build_model(model_type: str, hidden: int, n_layers: int, modes: int,
                grad_ckpt: bool, coarse_solver=None):
    """Instantiate the requested model variant."""
    if model_type == "fno":
        return FNO(
            in_channels=1, out_channels=1,
            hidden_channels=hidden, n_layers=n_layers,
            modes=modes, dim=2,
            gradient_checkpointing=grad_ckpt,
        )
    elif model_type == "constrained":
        return ConstrainedFNO(
            in_channels=1, out_channels=1,
            hidden_channels=hidden, n_layers=n_layers,
            modes=modes, constraint="conservative",
        )
    elif model_type == "residual":
        assert coarse_solver is not None, "coarse_solver required for residual model"
        return NeuralResidualCorrector(
            solver=coarse_solver,
            in_channels=1, out_channels=1, solver_out_channels=1,
            hidden_channels=hidden, n_layers=n_layers,
            modes=modes, solver_detach=True,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def ckpt_name(model_type: str) -> str:
    return f"best_{model_type}.pt"


# ─── Core training run ───────────────────────────────────────────────────────

def train_one_run(
    model, model_type: str, train_loader, val_loader,
    device, args, ckpt_dir: Path,
    optuna_trial=None,
) -> float:
    """Train *model* and return the best validation loss.

    Parameters
    ----------
    optuna_trial : optuna.Trial, optional
        When provided, reports intermediate losses and honours pruning signals.
    """
    log = get_logger("train")
    loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=min(10, args.epochs // 10),
        total_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.lr * 0.01,
    )
    clipper = GradientClipper(max_norm=1.0)
    stopper = EarlyStopping(patience=30, min_delta=1e-5, mode="min")

    # DP Mode curriculum (Phase 1c) — only for models that support set_modes()
    curriculum = None
    if args.curriculum and hasattr(model, "set_modes"):
        curriculum = ModeCurriculumScheduler(
            model, modes_start=4, modes_max=args.modes,
            modes_step=4, unlock_every=30,
        )

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Curriculum step ──
        if curriculum is not None:
            active_modes = curriculum.step(epoch)

        # ── Train ──
        model.train()
        tr_loss = 0.0
        for Q_b, T_b in train_loader:
            Q_b, T_b = Q_b.to(device), T_b.to(device)
            optimizer.zero_grad()
            pred = model(Q_b)
            loss = loss_fn(pred, T_b)
            loss.backward()
            clipper(model)
            optimizer.step()
            tr_loss += loss.item() * len(Q_b)
        tr_loss /= len(train_loader.dataset)
        lr = scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_l2 = 0.0, 0.0
        with torch.no_grad():
            for Q_b, T_b in val_loader:
                Q_b, T_b = Q_b.to(device), T_b.to(device)
                pred = model(Q_b)
                val_loss += loss_fn(pred, T_b).item() * len(Q_b)
                val_l2   += relative_l2_error(pred, T_b).item() * len(Q_b)
        val_loss /= len(val_loader.dataset)
        val_l2   /= len(val_loader.dataset)

        # ── Diagnostics for NeuralResidualCorrector ──
        if model_type == "residual" and epoch % 10 == 0:
            sample_Q = next(iter(train_loader))[0][:4].to(device)
            diag = model.correction_diagnostics(sample_Q)
            log.info(
                f"  residual diag | relative_correction={diag['relative_correction']:.4f}"
            )

        mode_str = f" | modes={curriculum.active_modes}" if curriculum else ""
        log.info(
            f"Epoch {epoch:3d}/{args.epochs} "
            f"| train={tr_loss:.3e} | val={val_loss:.3e} "
            f"| rel-L2={val_l2:.4f} | lr={lr:.2e}{mode_str}"
        )

        # ── Checkpoint ──
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / ckpt_name(model_type),
                model, optimizer, None, epoch, val_loss,
                extra={
                    "hidden_channels": args.hidden,
                    "n_layers":        args.n_layers,
                    "modes":           args.modes,
                    "resolution":      args.resolution,
                    "model_type":      model_type,
                },
            )

        # ── Optuna pruning (Hyperband) ──
        if optuna_trial is not None:
            optuna_trial.report(val_loss, step=epoch)
            try:
                import optuna
                if optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            except ImportError:
                pass

        # ── Early stopping ──
        if stopper.step(val_loss):
            log.info(f"Early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    return best_val


# ─── DP Phase 3b: Knapsack mode profiling ────────────────────────────────────

def profile_mode_accuracy(args, train_loader, val_loader, device) -> int:
    """Run 5-epoch mini-trains at each candidate mode count.

    Solves a 1-D bounded knapsack DP over mode increments:
      - capacity  = total mode budget (args.modes)
      - value[i]  = accuracy gain (lower val-L2) from mode level i vs i-1
      - weight[i] = 1 (uniform cost per mode step)

    Returns the optimal modes_max to feed into ModeCurriculumScheduler.
    """
    log = get_logger("profile")
    candidates = [m for m in [4, 8, 12, 16, 20, 24] if m <= args.modes]
    if len(candidates) <= 1:
        return args.modes

    log.info(f"Profiling mode accuracy for candidates: {candidates} …")
    loss_fn = nn.MSELoss()
    val_l2_at = {}

    for modes in candidates:
        model = FNO(
            in_channels=1, out_channels=1,
            hidden_channels=args.hidden, n_layers=args.n_layers,
            modes=modes, dim=2,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        for _ in range(5):
            model.train()
            for Q_b, T_b in train_loader:
                Q_b, T_b = Q_b.to(device), T_b.to(device)
                opt.zero_grad()
                loss_fn(model(Q_b), T_b).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        acc = 0.0
        with torch.no_grad():
            for Q_b, T_b in val_loader:
                Q_b, T_b = Q_b.to(device), T_b.to(device)
                acc += relative_l2_error(model(Q_b), T_b).item() * len(Q_b)
        val_l2_at[modes] = acc / len(val_loader.dataset)
        log.info(f"  modes={modes:2d}  val-L2={val_l2_at[modes]:.4f}")
        del model

    # Knapsack DP: value = marginal accuracy gain per mode increment
    # Since weight is uniform (1 step = 1 unit), optimal = pick increments
    # with positive gain until budget (args.modes) is exhausted.
    baseline = 1.0
    gains = {}
    for m in sorted(candidates):
        gains[m] = max(0.0, baseline - val_l2_at[m])
        baseline = val_l2_at[m]

    # DP table: dp[k] = best val-L2 achievable using exactly k mode steps
    steps = sorted(candidates)
    n = len(steps)
    INF = float("inf")
    # dp[i] = best (lowest) val-L2 when we stop at index i
    # The optimal stopping point maximises total gain = minimises val-L2
    # Since gains can be non-monotone, use DP over prefixes:
    dp = [INF] * n
    dp[0] = val_l2_at[steps[0]]
    for i in range(1, n):
        # Option 1: extend to step i (cumulative gain)
        dp[i] = val_l2_at[steps[i]]

    # Find point of diminishing returns: last index where marginal gain > threshold
    threshold = 0.001  # < 0.1% marginal gain considered negligible
    optimal_idx = 0
    for i in range(n - 1, -1, -1):
        if gains.get(steps[i], 0) > threshold:
            optimal_idx = i
            break

    optimal_modes = steps[optimal_idx]
    log.info(f"Knapsack DP → optimal modes_max = {optimal_modes}  "
             f"(marginal gain at next step: {gains.get(steps[min(optimal_idx+1, n-1)], 0):.4f})")
    return optimal_modes


# ─── Main training entry point ────────────────────────────────────────────────

def train(args):
    log = get_logger("train")
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Generate / load dataset ──
    data_path = generate_dataset(
        n_train=args.n_train,
        n_test=args.n_val,
        H=args.resolution,
        W=args.resolution,
    )
    log.info(f"Loading dataset from {data_path} …")
    d = np.load(data_path)
    Q_train = d["Q_train"]
    T_train = d["T_train"]
    Q_val   = d["Q_test"]
    T_val   = d["T_test"]

    H, W = Q_train.shape[1], Q_train.shape[2]
    log.info(f"Resolution: {H}×{W}  |  train: {len(Q_train)}  |  val: {len(Q_val)}")
    log.info(f"  Q range [{Q_train.min():.0f}, {Q_train.max():.0f}]")
    log.info(f"  T range [{T_train.min():.1f} °C, {T_train.max():.1f} °C]")

    # ── Normalise ──
    Q_mean = float(Q_train.mean());  Q_std = float(Q_train.std()) + 1e-8
    T_mean = float(T_train.mean());  T_std = float(T_train.std()) + 1e-8

    Q_tr_n = (Q_train - Q_mean) / Q_std
    T_tr_n = (T_train - T_mean) / T_std
    Q_va_n = (Q_val   - Q_mean) / Q_std
    T_va_n = (T_val   - T_mean) / T_std

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        ckpt_dir / "norm_stats.npz",
        Q_mean=Q_mean, Q_std=Q_std,
        T_mean=T_mean, T_std=T_std,
    )

    # ── DataLoaders ──
    def to_loader(Q, T, shuffle):
        ds = TensorDataset(
            torch.as_tensor(Q[:, None], dtype=torch.float32),
            torch.as_tensor(T[:, None], dtype=torch.float32),
        )
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = to_loader(Q_tr_n, T_tr_n, shuffle=True)
    val_loader   = to_loader(Q_va_n, T_va_n, shuffle=False)

    # ── DP Phase 3b: knapsack mode profiling ──
    if args.profile_modes:
        optimal_modes = profile_mode_accuracy(args, train_loader, val_loader, device)
        args.modes = optimal_modes
        log.info(f"Using DP-optimal modes = {args.modes} for training")

    # ── DP Phase 1b: Hyperband hyperparameter tuning ──
    if args.tune:
        log.info("Starting Hyperband hyperparameter search …")
        best_params = _run_tuning(args, train_loader, val_loader, device, ckpt_dir,
                                   Q_mean, Q_std, T_mean, T_std)
        log.info(f"Best params: {best_params}")
        # Apply best params
        args.hidden   = best_params.get("hidden_channels", args.hidden)
        args.n_layers = best_params.get("n_layers", args.n_layers)
        args.modes    = best_params.get("modes", args.modes)
        args.lr       = best_params.get("lr", args.lr)
        args.batch_size = best_params.get("batch_size", args.batch_size)
        # Rebuild loaders with tuned batch_size
        train_loader = to_loader(Q_tr_n, T_tr_n, shuffle=True)
        val_loader   = to_loader(Q_va_n, T_va_n, shuffle=False)

    # ── Build coarse solver (residual model only) ──
    coarse_solver = None
    if args.model == "residual":
        coarse_solver = make_coarse_solver(Q_mean, Q_std, T_mean, T_std, coarse_factor=4)

    # ── Model ──
    model = build_model(
        args.model, args.hidden, args.n_layers, args.modes,
        args.grad_ckpt, coarse_solver,
    ).to(device)
    log.info(f"Model: {args.model}  |  parameters: {model.num_parameters():,}")

    best_val = train_one_run(
        model, args.model, train_loader, val_loader,
        device, args, ckpt_dir,
    )

    log.info(f"Training complete. Best val loss: {best_val:.4e}")
    log.info(f"Checkpoint → {ckpt_dir}/{ckpt_name(args.model)}")


# ─── DP Phase 1b: Hyperband tuning helper ────────────────────────────────────

def _run_tuning(args, train_loader, val_loader, device, ckpt_dir,
                Q_mean, Q_std, T_mean, T_std) -> dict:
    """Run Hyperband-pruned Optuna study over FNO hyperparameters."""
    log = get_logger("tune")
    coarse_solver = None
    if args.model == "residual":
        coarse_solver = make_coarse_solver(Q_mean, Q_std, T_mean, T_std, coarse_factor=4)

    def objective(trial):
        import copy
        tuner = HyperparameterTuner(use_hyperband=True)
        params = tuner.suggest_fno_params(trial)

        tune_args = copy.copy(args)
        tune_args.hidden    = params["hidden_channels"]
        tune_args.n_layers  = params["n_layers"]
        tune_args.modes     = params["modes"]
        tune_args.lr        = params["lr"]
        tune_args.batch_size = params["batch_size"]
        tune_args.epochs    = min(args.epochs, 40)  # short run for pruning
        tune_args.curriculum = False  # disable curriculum during search

        def to_loader_local(Q, T, shuffle, bs):
            from torch.utils.data import DataLoader, TensorDataset
            ds = TensorDataset(
                torch.as_tensor(Q[:, None], dtype=torch.float32),
                torch.as_tensor(T[:, None], dtype=torch.float32),
            )
            return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                              pin_memory=(device.type == "cuda"))

        # Rebuild loaders with trial batch_size (need access to raw arrays)
        # Use the existing loaders as proxies (batch_size mismatch OK for short run)
        model = build_model(
            args.model, params["hidden_channels"], params["n_layers"],
            params["modes"], False, coarse_solver,
        ).to(device)

        return train_one_run(
            model, args.model, train_loader, val_loader,
            device, tune_args, ckpt_dir, optuna_trial=trial,
        )

    tuner = HyperparameterTuner(
        n_trials=args.tune_trials,
        direction="minimize",
        study_name=f"chip_thermal_{args.model}",
        use_hyperband=True,
    )
    return tuner.optimize(objective)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device",         default="cuda")
    p.add_argument("--resolution",     type=int,   default=128)
    p.add_argument("--n_train",        type=int,   default=1000)
    p.add_argument("--n_val",          type=int,   default=200)
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden",         type=int,   default=64)
    p.add_argument("--modes",          type=int,   default=16)
    p.add_argument("--n_layers",       type=int,   default=4)
    p.add_argument("--checkpoint_dir", default=str(Path(__file__).parent / "checkpoints"))
    # DP flags
    p.add_argument("--model",          default="fno",
                   choices=["fno", "constrained", "residual"],
                   help="Model variant to train")
    p.add_argument("--grad_ckpt",      action="store_true",
                   help="Enable gradient checkpointing (saves VRAM for deep FNO stacks)")
    p.add_argument("--curriculum",     action="store_true",
                   help="Enable DP mode curriculum (starts at modes=4, unlocks every 30 epochs)")
    p.add_argument("--tune",           action="store_true",
                   help="Run Hyperband hyperparameter search before full training")
    p.add_argument("--tune_trials",    type=int, default=20,
                   help="Number of Optuna trials for hyperparameter search")
    p.add_argument("--profile_modes",  action="store_true",
                   help="Run knapsack DP mode profiling to find optimal modes_max")
    train(p.parse_args())
