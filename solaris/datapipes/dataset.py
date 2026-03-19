# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""Dataset and DataLoader wrappers for physics simulation data."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PhysicsDataset(Dataset):
    """Generic dataset for physics field data stored as NumPy arrays.

    Expects a directory of ``.npy`` files, each containing a field snapshot
    of shape ``(channels, *spatial_dims)``.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``.npy`` field files.
    transforms : list of callables, optional
        Transformations applied to each sample dict in sequence.
    in_keys : list[str]
        Keys in the loaded dict used as inputs.
    out_keys : list[str]
        Keys in the loaded dict used as targets.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        transforms: Optional[List[Callable]] = None,
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {self.data_dir}")
        self.transforms = transforms or []
        self.in_keys = in_keys or ["input"]
        self.out_keys = out_keys or ["target"]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr = np.load(self.files[idx], allow_pickle=True).item()
        # Support both dict-of-arrays and raw array formats
        if isinstance(arr, dict):
            sample = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in arr.items()}
        else:
            arr = np.load(self.files[idx])
            half = arr.shape[0] // 2
            sample = {
                "input": torch.as_tensor(arr[:half], dtype=torch.float32),
                "target": torch.as_tensor(arr[half:], dtype=torch.float32),
            }
        for t in self.transforms:
            sample = t(sample)
        return sample


class TensorDataset(Dataset):
    """In-memory dataset from pre-loaded tensors.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(N, ...)``.
    targets : Tensor
        Target tensor of shape ``(N, ...)``.
    transforms : list of callables, optional
        Applied per-sample.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        assert len(inputs) == len(targets), "inputs and targets must have same first dimension"
        self.inputs = inputs
        self.targets = targets
        self.transforms = transforms or []

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"input": self.inputs[idx], "target": self.targets[idx]}
        for t in self.transforms:
            sample = t(sample)
        return sample


class HDF5Dataset(Dataset):
    """Dataset backed by an HDF5 file containing input/target arrays.

    Parameters
    ----------
    path : str or Path
        Path to the ``.h5`` file.
    input_key : str
        Dataset key for the input array (shape ``(N, C, ...)``) .
    target_key : str
        Dataset key for the target array.
    transforms : list of callables, optional
        Applied per-sample as ``transform({"input": ..., "target": ...})``.
    """

    def __init__(
        self,
        path: Union[str, Path],
        input_key: str = "input",
        target_key: str = "target",
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()
        try:
            import h5py
        except ImportError as e:
            raise ImportError("h5py is required for HDF5Dataset: pip install h5py") from e
        self.path = str(path)
        self.input_key = input_key
        self.target_key = target_key
        self.transforms = transforms or []
        with h5py.File(self.path, "r") as f:
            self._len = f[input_key].shape[0]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import h5py
        with h5py.File(self.path, "r") as f:
            sample = {
                "input":  torch.as_tensor(f[self.input_key][idx],  dtype=torch.float32),
                "target": torch.as_tensor(f[self.target_key][idx], dtype=torch.float32),
            }
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """Convenience wrapper that creates a DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        **kwargs,
    )
