# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Solaris: Physics AI framework for AMD GPUs."""

__version__ = "0.1.0"

# Detect ROCm availability
import torch

_ROCM_AVAILABLE = torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None
_CUDA_AVAILABLE = torch.cuda.is_available() and not _ROCM_AVAILABLE

def is_rocm_available() -> bool:
    """Return True if PyTorch was built with ROCm/HIP support and a GPU is available."""
    return _ROCM_AVAILABLE

def is_cuda_available() -> bool:
    """Return True if CUDA (NVIDIA) GPU is available."""
    return _CUDA_AVAILABLE

def get_gpu_backend() -> str:
    """Return the active GPU backend string: 'rocm', 'cuda', or 'cpu'."""
    if _ROCM_AVAILABLE:
        return "rocm"
    if _CUDA_AVAILABLE:
        return "cuda"
    return "cpu"
