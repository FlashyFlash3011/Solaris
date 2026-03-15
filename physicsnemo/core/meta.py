# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelMetaData:
    """Metadata container for PhysicsNeMo models.

    Attributes
    ----------
    name : str
        Human-readable model name.
    nvp_tags : List[str]
        Tags for model discovery and filtering.
    var_dim : int
        Variable/channel dimension index in tensors.
    func_torch : bool
        Whether the model supports functorch transforms.
    func_cuda : bool
        Whether the model has CUDA/ROCm-specific kernels.
    """

    name: str = "PhysicsNeMo Model"
    nvp_tags: List[str] = field(default_factory=list)
    var_dim: int = 1
    func_torch: bool = True
    func_cuda: bool = False
    jit: bool = False
    onnx: bool = False
    onnx_gpu: bool = False
    onnx_cpu: bool = False
    amp: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = False
    torch_fx: bool = False
    io: Optional[str] = None
