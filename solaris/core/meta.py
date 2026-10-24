# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import InitVar, dataclass, field

from solaris.exceptions import LegacyFeatureWarning


@dataclass
class ModelMetaData:
    """Metadata attached to every Solaris model (``Model._meta``).

    Attributes
    ----------
    name : str
        Human-readable model name.
    tags : list[str]
        Tags for model discovery and filtering.
    var_dim : int
        Variable/channel dimension index in tensors.
    func_torch : bool
        Whether the model supports functorch transforms.
    func_cuda : bool
        Whether the model has CUDA/ROCm-specific kernels.
    jit, onnx, onnx_gpu, onnx_cpu, amp, amp_cpu, amp_gpu, torch_fx : bool
        Capability flags.
    io : str, optional
        Free-form description of the expected inputs/outputs.
    nvp_tags : list[str], optional
        Deprecated alias for ``tags`` (warns).
    """

    name: str = "Solaris Model"
    tags: list[str] = field(default_factory=list)
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
    io: str | None = None
    nvp_tags: InitVar[list[str] | None] = None

    def __post_init__(self, nvp_tags: list[str] | None) -> None:
        if nvp_tags is not None:
            warnings.warn(
                "ModelMetaData(nvp_tags=...) is deprecated; use tags=...",
                LegacyFeatureWarning,
                stacklevel=3,
            )
            self.tags = list(nvp_tags)


def _nvp_tags(self: ModelMetaData) -> list[str]:
    warnings.warn(
        "ModelMetaData.nvp_tags is deprecated; use .tags", LegacyFeatureWarning, stacklevel=2
    )
    return self.tags


# Read-only alias for code written against the old field name.  Assigned after
# the dataclass is built, so the generated __init__ still takes nvp_tags=None.
ModelMetaData.nvp_tags = property(_nvp_tags)  # type: ignore[attr-defined]
