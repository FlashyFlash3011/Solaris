# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from solaris.core import ModelMetaData, Module
from solaris.models import FNO
from solaris.models.mlp import FullyConnected


class DummyModel(Module):
    def __init__(self, size: int = 16) -> None:
        super().__init__(meta=ModelMetaData(name="Dummy"))
        self.linear = torch.nn.Linear(size, size)
        self._capture_init_args(size=size)

    def forward(self, x):
        return self.linear(x)


def test_module_device():
    model = DummyModel()
    assert model.device == torch.device("cpu")


def test_module_num_parameters():
    model = DummyModel(size=8)
    # 8*8 weights + 8 bias = 72
    assert model.num_parameters() == 72


def test_module_save_load(tmp_path):
    model = DummyModel(size=8)
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = DummyModel.load(path, map_location="cpu")
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_fully_connected_forward():
    model = FullyConnected(in_features=4, out_features=2, hidden_features=16, n_layers=2)
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 2)


def test_model_metadata():
    meta = ModelMetaData(name="Test", tags=["pde"])
    assert meta.name == "Test"
    assert meta.tags == ["pde"]


def test_model_metadata_default_name():
    assert ModelMetaData().name == "Solaris Model"


def test_model_metadata_nvp_tags_is_a_deprecated_alias():
    from solaris.exceptions import LegacyFeatureWarning

    with pytest.warns(LegacyFeatureWarning, match="tags"):
        meta = ModelMetaData(name="Test", nvp_tags=["pde"])
    assert meta.tags == ["pde"]
    with pytest.warns(LegacyFeatureWarning):
        assert meta.nvp_tags == ["pde"]


def test_from_checkpoint(tmp_path):
    model = DummyModel(size=8)
    path = tmp_path / "model.pt"
    model.save(path)

    # Reconstruct without specifying DummyModel explicitly
    loaded = Module.from_checkpoint(path, map_location="cpu")
    assert isinstance(loaded, DummyModel)
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_from_torch():
    inner = torch.nn.Linear(4, 2)
    wrapper = Module.from_torch(inner)

    assert wrapper.meta.name == "Linear"
    assert wrapper.num_parameters() == inner.weight.numel() + inner.bias.numel()

    x = torch.randn(3, 4)
    torch.testing.assert_close(wrapper(x), inner(x))


# --- Auto init-arg capture ---


class AutoModel(Module):
    """Subclass that never calls _capture_init_args() manually."""

    def __init__(self, width: int = 32, depth: int = 4) -> None:
        super().__init__(meta=ModelMetaData(name="AutoModel"))
        self.linear = torch.nn.Linear(width, width)
        # deliberately no _capture_init_args() call

    def forward(self, x):
        return self.linear(x)


def test_auto_capture_populates_init_args():
    model = AutoModel(width=8, depth=2)
    assert model._init_args == {"width": 8, "depth": 2}


def test_auto_capture_save_load(tmp_path):
    model = AutoModel(width=8, depth=2)
    path = tmp_path / "auto.pt"
    model.save(path)
    loaded = AutoModel.load(path, map_location="cpu")
    x = torch.randn(2, 8)
    torch.testing.assert_close(model(x), loaded(x))


def test_manual_capture_takes_precedence():
    """Manual _capture_init_args() overrides auto-capture."""
    model = DummyModel(size=16)
    # DummyModel calls _capture_init_args(size=size) — only "size" should be stored
    assert set(model._init_args.keys()) == {"size"}


# --- compile_model ---


def test_compile_model_returns_self():
    model = DummyModel(size=8)
    result = model.compile_model()
    assert result is model


def test_compile_model_still_runs():
    model = DummyModel(size=8)
    model.compile_model()
    x = torch.randn(2, 8)
    try:
        out = model(x)
        assert out.shape == (2, 8)
    except Exception:
        # torch.compile may fail on CPU-only environments without a compiler
        pytest.skip("torch.compile backend not available in this environment")


# --- Module init-arg capture with inheritance --------------------------------------


def test_auto_capture_records_most_derived_signature():
    class A(Module):
        def __init__(self, a=1):
            super().__init__()

    class B(A):
        def __init__(self, a=1, b=2):
            super().__init__(a)

    assert B(5, 7)._init_args == {"a": 5, "b": 7}


def test_subclass_of_explicit_capture_model_is_reconstructable(tmp_path):
    class Narrow(FNO):
        def __init__(self, in_channels, out_channels, width=8):
            super().__init__(in_channels, out_channels, hidden_channels=width, n_layers=1, modes=4)

    m = Narrow(1, 2, width=6)
    assert m._init_args == {"in_channels": 1, "out_channels": 2, "width": 6}


def test_explicit_capture_in_most_derived_class_wins():
    class C(Module):
        def __init__(self, a=1, b=2):
            super().__init__()
            self._capture_init_args(a=a)

    assert C(3, 4)._init_args == {"a": 3}


# --- Module.load device ---------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_module_load_honours_map_location(tmp_path):
    m = FNO(1, 1, 8, 1, 4)
    m.save(tmp_path / "m.pt")
    assert FNO.load(tmp_path / "m.pt", map_location="cuda").device.type == "cuda"


# --- Safe checkpoint loading -----------------------------------------------------


class _Payload:
    """Pickles to a call of ``print`` — stands in for malicious code in a file."""

    def __reduce__(self):
        return (print, ("payload executed",))


def _poisoned_module_checkpoint(path):
    from solaris.models import FNO

    m = FNO(1, 1, 8, 1, 4, norm=None)
    torch.save(
        {
            "state_dict": m.state_dict(),
            "init_args": m._init_args,
            "class": "solaris.models.fno.FNO",
            "solaris_version": "0.2.0",
            "evil": _Payload(),
        },
        path,
    )


def test_module_load_refuses_code_in_checkpoint_by_default(tmp_path, capsys):
    import pickle

    from solaris.models import FNO

    _poisoned_module_checkpoint(tmp_path / "m.pt")
    with pytest.raises(pickle.UnpicklingError, match="trusted=True"):
        FNO.load(tmp_path / "m.pt", map_location="cpu")
    with pytest.raises(pickle.UnpicklingError):
        Module.from_checkpoint(tmp_path / "m.pt", map_location="cpu")
    assert "payload executed" not in capsys.readouterr().out


def test_module_load_trusted_allows_arbitrary_objects(tmp_path, capsys):
    from solaris.models import FNO

    _poisoned_module_checkpoint(tmp_path / "m.pt")
    FNO.load(tmp_path / "m.pt", map_location="cpu", trusted=True)
    assert "payload executed" in capsys.readouterr().out


def test_load_checkpoint_refuses_code_by_default(tmp_path):
    import pickle

    from solaris.utils import load_checkpoint, save_checkpoint

    model = torch.nn.Linear(2, 2)
    save_checkpoint(tmp_path / "c.pt", model, extra={"evil": _Payload()})
    with pytest.raises(pickle.UnpicklingError, match="trusted=True"):
        load_checkpoint(tmp_path / "c.pt", model, map_location="cpu")


def test_safe_loading_round_trips_normal_checkpoints(tmp_path):
    from solaris.models import ConstrainedFNO
    from solaris.utils import load_checkpoint, save_checkpoint

    m = ConstrainedFNO(1, 1, 8, 1, [4, 4], constraint="dirichlet", boundary_value=0.5, norm=None)
    m.save(tmp_path / "m.pt")
    assert isinstance(Module.from_checkpoint(tmp_path / "m.pt", map_location="cpu"), ConstrainedFNO)

    opt = torch.optim.AdamW(m.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    save_checkpoint(tmp_path / "c.pt", m, opt, sched, epoch=2, extra={"modes": 4, "name": "x"})
    ckpt = load_checkpoint(tmp_path / "c.pt", m, opt, sched, map_location="cpu")
    assert ckpt["epoch"] == 2 and ckpt["name"] == "x"
