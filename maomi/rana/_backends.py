"""Array operations used by the local acquisition engine.

Backend selection is local to each operator/plan; there is no global registry.
Physics, interpolation, kernel conventions and tile scheduling stay in their
own modules. Optional device libraries are loaded on first device operation.
"""
from functools import cached_property
from typing import Any, Protocol
import numpy as np
from scipy import fft


class ArrayBackend(Protocol):
    device: str

    def array(self, value: Any) -> Any: ...
    def zeros(self, shape: tuple) -> Any: ...
    def stack(self, values: list, axis: int = 0) -> Any: ...
    def rfftn(self, value: Any, lengths: tuple, axes: tuple) -> Any: ...
    def irfftn(self, value: Any, lengths: tuple, axes: tuple) -> Any: ...
    def sum(self, value: Any, axis: int) -> Any: ...
    def group_sum(self, value: Any, starts: np.ndarray, width: int) -> Any: ...
    def nbytes(self, value: Any) -> int: ...
    def to_numpy(self, value: Any) -> np.ndarray: ...


class NumpyBackend:
    device = 'cpu'

    def array(self, value):
        return np.asarray(value, dtype=float)

    def zeros(self, shape):
        return np.zeros(shape, dtype=float)

    def stack(self, values, axis=0):
        return np.stack(values, axis=axis)

    def rfftn(self, value, lengths, axes):
        return fft.rfftn(value, s=lengths, axes=axes)

    def irfftn(self, value, lengths, axes):
        return fft.irfftn(value, s=lengths, axes=axes)

    def sum(self, value, axis):
        return value.sum(axis=axis)

    def group_sum(self, value, starts, width):
        return np.add.reduceat(value, starts, axis=2)

    def nbytes(self, value):
        return value.nbytes

    def to_numpy(self, value):
        return value


class TorchBackend:
    def __init__(self, device):
        self.device = device

    @cached_property
    def torch(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA-enabled PyTorch is required')
        return torch

    def array(self, value):
        torch = self.torch
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.array(value, copy=True, order='C'),
                               device=self.device, dtype=torch.float64)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device, dtype=self.torch.float64)

    def stack(self, values, axis=0):
        return self.torch.stack(values, dim=axis)

    def rfftn(self, value, lengths, axes):
        return self.torch.fft.rfftn(value, s=lengths, dim=axes)

    def irfftn(self, value, lengths, axes):
        return self.torch.fft.irfftn(value, s=lengths, dim=axes)

    def sum(self, value, axis):
        return value.sum(dim=axis)

    def group_sum(self, value, starts, width):
        return self.stack([value[..., s:s+width].sum(-1) for s in starts], axis=-1)

    def nbytes(self, value):
        return value.numel()*value.element_size()

    def to_numpy(self, value):
        return value.cpu().numpy()


def get_backend(device: str) -> ArrayBackend:
    if device == 'cpu':
        return NumpyBackend()
    if device.startswith('cuda'):
        return TorchBackend(device)
    raise ValueError('device must be cpu or cuda[:index]')
