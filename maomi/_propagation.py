"""Shared geometry for the CPU and CUDA propagation implementations."""

from types import SimpleNamespace
import numpy as np
import torch
from . import _optical_math as math


def center_slice(n, m):
    if m > n or (n - m) % 2:
        raise ValueError(f"Incompatible centered grids {n},{m}")
    return slice((n - m) // 2, (n + m) // 2)


def geometry(config, raw_fields):
    v, p = config.parameters()
    r, ss = config.vres, config.ss
    n = int(round((v.vasc_sz[0] - v.vol_sz[0]) * r * ss))
    n2 = (
        int(round(min(max(math.gaussian_beam_size(p, config.psf_um[2] / 2, 3)) * r * 2 * ss, n)))
        if config.propcrop
        else n
    )
    if n % 2 and not n2 % 2:
        n2 -= 1
    fl = np.float32(config.focal_length_mm / 1000)
    d1 = np.float32(max(math.gaussian_beam_size(p, float(fl * 1e6))) / 1e6 / n)
    d2 = np.float32(1e-6 / r / ss)
    za = r * (config.depth_um + config.volume_um[2] / 2 - config.psf_um[2] / 2)
    pc = config.prop_um * r
    a, b, c = [field.shape[-1] for field in raw_fields]
    positions = [
        (np.r_[0, np.arange(a) + (za % pc) / pc] if za % pc else np.arange(a + 1))
        * config.prop_um
        * 1e-6,
        np.arange(b + 1) / r * 1e-6,
        np.r_[np.arange(c) * config.prop_um, config.tail_um] * 1e-6,
    ]
    cut = max(0, int(np.ceil((za - config.tail_um * r) / pc)))
    if cut >= len(positions[0]) - 1:
        raise ValueError("Top tail has no extent")
    focal_shape = np.round(np.array(config.psf_um[:2]) * r * ss).astype(int)
    return SimpleNamespace(
        n=n,
        n2=n2,
        fl=fl,
        d1=d1,
        d2=d2,
        wvl=np.float32(config.wavelength_um * 1e-6),
        nr=np.float32(config.n),
        z=positions,
        cut=cut,
        focal_shape=focal_shape,
        crop=center_slice(n, n2),
        focal_crop=tuple(center_slice(n2, k) for k in focal_shape),
    )


def edge_apodization(size, device, dtype):
    axis = torch.arange(-size / 2, size / 2, device=device, dtype=dtype) / (0.47 * size)
    return torch.exp(-axis[:, None] ** 16) * torch.exp(-axis[None, :] ** 16)
