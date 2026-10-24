# Contributing to Solaris

This guide covers everything you need to contribute code: setting up the development environment, running tests, linting, and the exact steps for adding new models or constraint layers.

---

## 1. Development Setup

### Clone and install

```bash
git clone https://github.com/FlashyFlash3011/Solaris.git
cd Solaris

# 1. PyTorch first — pick the build for your hardware (see README "Install"):
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2  # AMD, native Linux
pip install torch                                                       # NVIDIA
#    AMD under WSL2: AMD's WSL wheel from repo.radeon.com (pytorch.org ROCm wheels can't see the GPU)

# 2. Solaris in editable mode with the development tools
pip install -e ".[dev,export]"
```

The `dev` extra brings in pytest, ruff, mypy, pre-commit and what the demo
projects need (scipy, matplotlib, h5py); `export` adds the ONNX packages so
the export tests run. Solaris never pulls in a different PyTorch build.

### Install pre-commit hooks

```bash
pre-commit install
```

This runs ruff (lint + format) automatically on every `git commit`.

---

## 2. Running Tests

```bash
pytest tests/                       # everything, with coverage (~2 min)
pytest tests/ -m "not slow"         # skip the end-to-end project/example runs
pytest tests/test_models.py -v      # one file
pytest tests/test_models.py::test_fno_forward -v
```

What the suite contains:
- unit tests for every model, layer and utility;
- **multi-process tests** (2 CPU ranks, gloo) for the distributed FFT,
  tensor-parallel primitives, data-parallel training and FSDP checkpoints;
- **GPU tests** (fp16 autocast, graph capture), skipped without a GPU;
- **ONNX export tests**, skipped without the `export` extra;
- **end-to-end smoke tests** (`slow`) that run every project and example.

A bug fix comes with a test that fails without the fix.

---

## 3. Linting

Solaris uses [ruff](https://docs.astral.sh/ruff/) for both linting and formatting (line length 100, target Python 3.10).

```bash
# Check for lint errors
ruff check solaris/ tests/

# Auto-fix lint errors
ruff check --fix solaris/ tests/

# Format code
ruff format solaris/ tests/
```

Type-check with `mypy` (configured in `pyproject.toml`; the package is
currently clean). CI runs pre-commit, mypy and the full test suite on Python
3.10, 3.11 and 3.12 for every push.

---

## 4. Adding a New Model

Follow this checklist in order. Missing any step will cause the model to be absent from the registry or break imports.

### a. Create `solaris/models/yourmodel.py`

Inherit from `solaris.core.Module` (not `nn.Module` directly) and attach a
class-level `ModelMetaData` as `_meta`. Constructor arguments are recorded
automatically so checkpoints are self-describing (`Module.load()` /
`Module.from_checkpoint()` rebuild the model); call
`self._capture_init_args(...)` at the end of `__init__` only to control what
is stored (e.g. to replace an argument that cannot be saved).

```python
# solaris/models/yourmodel.py
import torch
import torch.nn as nn

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module


class YourModel(Module):
    """One-line description of what this model does."""

    _meta = ModelMetaData(
        name="YourModel",
        tags=["pde", "your-tag"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__(meta=self._meta)

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

        # Must be last — captures all constructor args for checkpoint round-trips.
        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

Key rules:
- Stored arguments must be plain data (numbers, strings, `None`, lists, tuples,
  dicts): checkpoints are loaded with PyTorch's safe `weights_only=True` mode,
  which refuses arbitrary objects.
- Fourier-family models take a `norm` argument (`None` or `"instance"`); use
  `solaris.nn.norm.resolve_norm` / `make_norm` like the existing models.
- `self.device` is available for free via the persistent `_device_buf` buffer; you do not need to track the device manually.

### b. Add the import and `__all__` entry

Edit `solaris/models/__init__.py`:

```python
from solaris.models.yourmodel import YourModel   # add this line

__all__ = [
    # ... existing entries ...
    "YourModel",                                  # add this entry
]
```

### c. Register the entry point in `pyproject.toml`

Add a line under `[project.entry-points."solaris.models"]`:

```toml
[project.entry-points."solaris.models"]
# ... existing entries ...
your_model = "solaris.models.yourmodel:YourModel"
```

The key (left of `=`) is the registry lookup name. The value is a dotted import path to the class.

### d. Write tests

Add tests to `tests/test_models.py`. At minimum, cover:
1. Forward pass — output shape is correct.
2. Backward pass — gradients flow to all parameters.

```python
def test_yourmodel_forward():
    model = YourModel(in_channels=4, out_channels=2, hidden_channels=32)
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_yourmodel_backward():
    model = YourModel(in_channels=4, out_channels=2, hidden_channels=32)
    x = torch.randn(2, 4, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
```

Also consider a round-trip checkpoint test using `model.save()` / `YourModel.load()`.

### e. Re-install the package

Entry points are registered at install time. After editing `pyproject.toml` you must reinstall:

```bash
pip install -e .
```

Without this step the registry will not find your model.

---

## 5. Adding a New Constraint Layer

Constraint layers live in `solaris/nn/constraints.py`. They are plain `nn.Module` subclasses — no need to inherit from `solaris.core.Module` since they are building blocks, not top-level models.

### Checklist

**a. Implement the layer in `solaris/nn/constraints.py`:**

```python
class MyConstraint(nn.Module):
    """One-line description.

    This is a *hard* constraint: the output mathematically satisfies the
    physical law regardless of upstream network weights.

    Parameters
    ----------
    eps : float
        Numerical floor to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce the constraint here.
        # Return a tensor with the same shape as the input.
        ...
```

**b. Export from `solaris/nn/__init__.py`:**

```python
from solaris.nn.constraints import MyConstraint   # add
__all__ = [..., "MyConstraint"]                    # add
```

**c. Write tests in `tests/test_constraints.py` (or `tests/test_nn.py`):**

The test should verify the physical law holds to float precision, not just
that the layer runs. If the constraint is used inside a model, give the model
an `apply_constraint(input, output)` method so `CoupledOperator` can
re-impose it after coupling.

```python
def test_my_constraint_enforced():
    layer = MyConstraint()
    x = torch.randn(2, 2, 16, 16)
    out = layer(x)
    # Assert the physical property holds, e.g. divergence == 0.
    assert out.shape == x.shape
```

---

## 6. Commit Style

- Keep commits focused: one logical change per commit.
- Use a short imperative subject line (under 72 characters).
- Reference the affected component in the subject when useful.

Good examples:
```
feat: add WaveletNeuralOperator with Haar DWT layers
fix: correct ConservationProjection channel index off-by-one
tests: add forward/backward coverage for UNO
refactor: extract FNOBlock into shared nn module
```

Avoid:
- Committing unrelated changes together.
- Subject lines that just say "fix" or "update" with no context.
- Leaving debug prints or commented-out code in committed files.

---

## 7. Python Versions

Solaris supports Python 3.10–3.12 (`requires-python = ">=3.10,<3.13"`); CI
tests all three. Development and the GPU setup use Python 3.11.
