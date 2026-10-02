"""ONNX export round-trips (skipped unless the [export] extra is installed)."""

import importlib.util
import warnings

import pytest
import torch

from solaris.models import AFNO, FNO, FullyConnected
from solaris.utils import export_onnx

_HAVE_EXPORT = all(
    importlib.util.find_spec(m) is not None for m in ("onnx", "onnxruntime", "onnxscript")
)
pytestmark = pytest.mark.skipif(not _HAVE_EXPORT, reason="needs solaris-rocm[export]")

CASES = {
    "mlp": (lambda: FullyConnected(3, 2, 16, 2), (4, 3)),
    "fno1d": (lambda: FNO(1, 1, 8, 1, 4, dim=1, norm=None), (2, 1, 32)),
    "fno2d": (lambda: FNO(2, 1, 8, 2, 4, dim=2, norm=None), (2, 2, 16, 16)),
    "fno3d": (lambda: FNO(1, 1, 4, 1, 3, dim=3, norm=None), (1, 1, 8, 8, 8)),
    "afno": (lambda: AFNO(1, 1, (16, 16), 4, 32, 1, num_blocks=4), (2, 1, 16, 16)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_export_matches_pytorch(name, tmp_path):
    factory, shape = CASES[name]
    torch.manual_seed(0)
    model = factory()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_onnx(model, torch.randn(*shape), tmp_path / f"{name}.onnx")
    mismatches = [str(w.message) for w in caught if "outputs differ" in str(w.message)]
    assert not mismatches, mismatches
    assert (tmp_path / f"{name}.onnx").stat().st_size > 0
