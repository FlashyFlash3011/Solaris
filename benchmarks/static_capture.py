# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure the training-step speedup of StaticCaptureTraining.

Times eager steps vs graph-replayed steps for a 2-D FNO at a few sizes.
Graph capture removes per-kernel CPU launch overhead, so the gain is largest
for small models / batches and shrinks as the GPU becomes the bottleneck.

    python benchmarks/static_capture.py
    python benchmarks/static_capture.py --sizes 32 64 --batch 8 --steps 200
"""

import argparse
import time

import torch
import torch.nn.functional as F

from solaris.models import FNO
from solaris.utils.training import StaticCaptureTraining


def time_steps(step, x, y, n: int) -> float:
    for _ in range(10):  # settle (and, for the capturer, finish warm-up + capture)
        step(x, y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step(x, y)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"{'grid':>6} {'eager ms':>10} {'graph ms':>10} {'speedup':>8}")
    for n in args.sizes:
        x = torch.randn(args.batch, 1, n, n, device="cuda")
        y = torch.randn(args.batch, 1, n, n, device="cuda")

        def model_opt():
            torch.manual_seed(0)
            m = FNO(1, 1, args.hidden, args.layers, modes=min(12, n // 2), norm=None).cuda()
            return m, torch.optim.AdamW(m.parameters(), lr=1e-3)

        m, opt = model_opt()

        def eager(x, y, m=m, opt=opt):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(m(x), y).backward()
            opt.step()

        t_eager = time_steps(eager, x, y, args.steps)
        m2, opt2 = model_opt()
        capturer = StaticCaptureTraining(m2, opt2, F.mse_loss, warmup_steps=3)
        t_graph = time_steps(capturer, x, y, args.steps)
        note = "" if capturer.captured else "  (capture failed: eager fallback)"
        print(f"{n:>6} {t_eager:>10.2f} {t_graph:>10.2f} {t_eager / t_graph:>7.2f}x{note}")


if __name__ == "__main__":
    main()
