"""Explicit static operators. Arrays are X,Y,Z; no PSF normalization is hidden."""

import numpy as np
import torch
from scipy.fft import next_fast_len


def zstack(volume, psf, *, device="cpu", mode="matlab_convn", mask=None, collection_mask=None, rana=None):
    """FFT implementation of MATLAB convn(...,'same') or the July Python scan.

    matlab_convn: zero extension in all axes, convolution in all axes, MATLAB
    even-kernel crop. legacy_circular: periodic XY, zero Z, correlation in Z;
    reproduces simulate_frozen_cells_aligned.py without stage-specific scaling.
    Optional masks modulate the source volume before convolution, explicitly.
    Device transfers and FFT preparation occur inside this call.
    """
    v = torch.as_tensor(volume, device=device)
    if not v.is_floating_point():
        v = v.float()
    p = torch.as_tensor(psf, dtype=v.dtype, device=device)
    if v.ndim != 3 or p.ndim != 3:
        raise ValueError("Expected X,Y,Z arrays")
    for m in (mask, collection_mask):
        if m is not None:
            m = torch.as_tensor(m, dtype=v.dtype, device=device)
            if tuple(m.shape) != tuple(v.shape[:2]):
                raise ValueError("Mask must match volume XY")
            v = v * m[..., None]
    if rana is not None and rana is not False:
        if mode != 'matlab_convn':
            raise ValueError('RANA uses zero-extended local optics; legacy_circular is a different boundary model')
        result = rana.static(v.detach().cpu().numpy())
        return torch.as_tensor(result,device=device,dtype=v.dtype)
    if mode == "matlab_convn":
        n = tuple(next_fast_len(a + b - 1) for a, b in zip(v.shape, p.shape))
        full = torch.fft.irfftn(torch.fft.rfftn(v, s=n) * torch.fft.rfftn(p, s=n), s=n)
        # MATLAB uses floor(kernel_size/2); scipy 'same' differs for even sizes.
        slices = tuple(slice(b // 2, b // 2 + a) for a, b in zip(v.shape, p.shape))
        return full[slices].contiguous()
    if mode == "legacy_circular":
        h, w, z = v.shape
        a, b, k = p.shape
        if a > h or b > w:
            raise ValueError("Legacy operator requires PSF XY <= image XY")
        n = (h, w, next_fast_len(z + k - 1))
        kernel = torch.zeros((h, w, k), device=device, dtype=v.dtype)
        kernel[:a, :b] = p.flip(-1)
        # Parentheses intentionally reproduce Python's -H_psf//2 for odd sizes.
        kernel = torch.roll(kernel, (-a // 2, -b // 2), (0, 1))
        full = torch.fft.irfftn(torch.fft.rfftn(v, s=n) * torch.fft.rfftn(kernel, s=n), s=n)
        start = k - 1 - k // 2
        return full[:, :, start : start + z].contiguous()
    raise ValueError(f"Unknown operator: {mode}")


def legacy_slice_loop(volume, psf, *, device="cpu", cache_slices=False):
    """Independent reproduction of old nested loop, for controlled timings."""
    v = torch.as_tensor(volume, device=device)
    p = torch.as_tensor(psf, device=device, dtype=v.dtype)
    h, w, z = v.shape
    a, b, k = p.shape
    pad = torch.zeros((h, w, k), device=device, dtype=v.dtype)
    pad[:a, :b] = p
    pf = torch.fft.fft2(torch.roll(pad, (-a // 2, -b // 2), (0, 1)), dim=(0, 1))
    vf = torch.fft.fft2(v, dim=(0, 1)) if cache_slices else None
    out = torch.empty_like(v)
    for oz in range(z):
        acc = torch.zeros((h, w), device=device, dtype=pf.dtype)
        for j in range(k):
            iz = oz - k // 2 + j
            if 0 <= iz < z:
                f = vf[:, :, iz] if cache_slices else torch.fft.fft2(v[:, :, iz])
                acc += f * pf[:, :, j]
        out[:, :, oz] = torch.fft.ifft2(acc).real
    return out
