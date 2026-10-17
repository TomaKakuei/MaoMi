from pathlib import Path
import numpy as np
import torch
import cupy as cp
from maomi.numerics import _cubic

_source = Path(__file__).with_name("optical_kernels.cu").read_text()
_gather = cp.RawKernel(_source, "gather_screens", options=("--fmad=false",))
_resample = cp.RawKernel(_source, "resample_power", options=("--fmad=false",))


def _call(kernel, total, args):
    stream = torch.cuda.current_stream()
    with cp.cuda.ExternalStream(stream.cuda_stream):
        converted = [cp.from_dlpack(a) if isinstance(a, torch.Tensor) else a for a in args]
        kernel(((int(total) + 255) // 256,), (256,), tuple(converted))


def gather(src, edge, offsets, n):
    b = len(offsets)
    z, h, w = src.shape
    out = torch.empty((b, z, n, n), device=src.device, dtype=src.dtype)
    _call(_gather, out.numel(), [src, edge, offsets, out, *[np.int32(x) for x in (b, z, h, w, n)]])
    return out


def resample_matrix(n, ss):
    """Compose exact reference resize and translate into a linear operator."""
    scale = 1 / ss
    m = int(np.ceil(n * scale))
    shift = ss / 2 - 0.5
    u = (np.arange(m) + 0.5) / scale - 0.5
    width = 4 / scale
    idx = np.floor(u - width / 2).astype(int)[:, None] + np.arange(int(np.ceil(width)) + 2)[None]
    weights = _cubic((u[:, None] - idx) * scale) * scale
    weights /= weights.sum(1, keepdims=True)
    idx %= 2 * n
    idx = np.where(idx < n, idx, 2 * n - 1 - idx)
    matrix = np.zeros((m, n))
    for i in range(m):
        for j, w in zip(idx[i], weights[i]):
            a = int(np.floor(j - shift))
            f = j - shift - a
            if 0 <= a < n:
                matrix[i, a] += w * (1 - f)
            if 0 <= a + 1 < n:
                matrix[i, a + 1] += w * f
    return matrix


class ResamplePlan:
    def __init__(self, h, w, ss, device="cuda"):
        mx, my = resample_matrix(h, ss), resample_matrix(w, ss)
        self.mx = torch.tensor(mx, dtype=torch.float32, device=device)
        self.my = torch.tensor(my, dtype=torch.float32, device=device)
        plans = []
        for matrix in (mx, my):
            k = max(np.count_nonzero(row) for row in matrix)
            inds = np.zeros((len(matrix), k), np.int32)
            weights = np.zeros((len(matrix), k), np.float32)
            for i, row in enumerate(matrix):
                j = np.nonzero(row)[0]
                inds[i, : len(j)] = j
                weights[i, : len(j)] = row[j]
            plans.append((torch.tensor(inds, device=device), torch.tensor(weights, device=device)))
        (self.ix, self.wx), (self.iy, self.wy) = plans

    def __call__(self, field, scale, native=True, order=2, axial=None):
        # field B,Z,H,W complex64; result B,X,Y,Z float32.
        field = field.contiguous()
        b, z, h, w = field.shape
        x, y = self.mx.shape[0], self.my.shape[0]
        if order not in (1, 2, 3):
            raise ValueError("Photon order must be 1, 2 or 3")
        axial = (
            torch.ones(z, device=field.device)
            if axial is None
            else torch.as_tensor(axial, dtype=torch.float32, device=field.device)
        )
        if tuple(axial.shape) != (z,):
            raise ValueError("One axial weight per plane required")
        if not native:
            intensity = field.abs() ** (2 * order)
            out = torch.matmul(torch.matmul(self.mx, intensity), self.my.T) * scale
            return (out * axial[None, :, None, None]).permute(0, 2, 3, 1).contiguous()
        out = torch.empty((b, x, y, z), device=field.device)
        _call(
            _resample,
            out.numel(),
            [
                field,
                self.ix,
                self.wx,
                self.iy,
                self.wy,
                axial,
                out,
                *[
                    np.int32(v)
                    for v in (b, z, h, w, x, y, self.ix.shape[1], self.iy.shape[1], order)
                ],
                np.float32(scale),
            ],
        )
        return out
