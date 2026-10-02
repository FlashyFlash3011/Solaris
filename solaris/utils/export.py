# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""ONNX model export with optional onnxruntime validation."""

import copy
import warnings
from pathlib import Path

import torch
import torch.nn as nn


def _with_real_weights(model: nn.Module) -> nn.Module:
    """Copy of ``model`` whose complex parameters are stored as real ``(..., 2)`` tensors.

    ONNX has no complex tensor type.  Solaris spectral layers accept their
    weights either as complex tensors or as real (real, imag) views, so the
    exported copy computes exactly the same function; ``model`` is untouched.
    """
    exportable = copy.deepcopy(model)
    for module in exportable.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if param.is_complex():
                real = nn.Parameter(torch.view_as_real(param.detach()).clone())
                setattr(module, name, real)
    return exportable


def export_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor | tuple[torch.Tensor, ...],
    path: str | Path,
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
        import onnxscript  # noqa: F401, PLC0415  (needed by torch's ONNX exporter)
    except ImportError as e:
        raise ImportError(
            "onnx and onnxscript are required for ONNX export. "
            "Install them with: pip install 'solaris-rocm[export]'"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    inputs = dummy_input if isinstance(dummy_input, tuple) else (dummy_input,)
    exportable = _with_real_weights(model).eval()

    with torch.no_grad():
        torch.onnx.export(
            exportable,
            inputs,
            str(path),
            opset_version=opset_version,
            input_names=[f"input_{i}" for i in range(len(inputs))],
            output_names=["output"],
            # Batch dimension dynamic where possible (Dim.AUTO keeps size-1
            # example batches from failing the export).
            dynamic_shapes=tuple({0: torch.export.Dim.AUTO} for _ in inputs),
            dynamo=True,
        )

    # Validate the exported model
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

    # Optional onnxruntime validation
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feed = {f"input_{i}": inp.cpu().numpy() for i, inp in enumerate(inputs)}
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
