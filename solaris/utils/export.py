# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX model export with optional onnxruntime validation."""

import warnings
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    path: Union[str, Path],
    opset_version: int = 17,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> None:
    """Export a model to ONNX and validate with onnxruntime.

    Parameters
    ----------
    model : nn.Module
    dummy_input : Tensor or tuple of Tensors
        Representative inputs matching the model's ``forward`` signature.
    path : str or Path
        Output ``.onnx`` file path.
    opset_version : int
        ONNX opset version (default 17).
    rtol : float
        Relative tolerance for onnxruntime validation.
    atol : float
        Absolute tolerance for onnxruntime validation.
    """
    try:
        import onnx  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "onnx is required for ONNX export. "
            "Install it with: pip install 'solaris-rocm[export]'"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    inputs = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            opset_version=opset_version,
            input_names=[f"input_{i}" for i in range(len(inputs))],
            output_names=["output"],
            dynamic_axes={
                f"input_{i}": {0: "batch"} for i in range(len(inputs))
            } | {"output": {0: "batch"}},
        )

    # Validate the exported model
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    # Optional onnxruntime validation
    try:
        import onnxruntime as ort  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {
            f"input_{i}": inp.cpu().numpy()
            for i, inp in enumerate(inputs)
        }
        ort_out = session.run(None, feed)[0]

        with torch.no_grad():
            torch_out = model(*inputs).cpu().numpy()

        if not np.allclose(torch_out, ort_out, rtol=rtol, atol=atol):
            warnings.warn(
                "ONNX validation: onnxruntime outputs differ from PyTorch outputs "
                f"beyond tolerances (rtol={rtol}, atol={atol}).",
                stacklevel=2,
            )
    except ImportError:
        warnings.warn(
            "onnxruntime not installed — skipping numerical validation of the exported model. "
            "Install with: pip install 'solaris-rocm[export]'",
            stacklevel=2,
        )
