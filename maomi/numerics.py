"""MATLAB-compatible primitives used in the optical subset."""

import numpy as np
import torch
import torch.nn.functional as F


def _cubic(x):
    a = np.abs(x)
    return ((1.5 * a - 2.5) * a * a + 1) * (a <= 1) + (((-0.5 * a + 2.5) * a - 4) * a + 2) * (
        (a > 1) & (a < 2)
    )


def resize_xy(a, scale):
    """Keys cubic a=-.5, half-pixel centers, symmetric extension, antialiasing.

    XY first, arbitrary remaining axes; matches imresize's default bicubic
    for a scalar scale. Does not crop before interpolation.
    """
    out = a
    for axis in (0, 1):
        n = out.shape[axis]
        m = int(np.ceil(n * scale))
        u = (np.arange(m) + 0.5) / scale - 0.5
        width = 4 / min(scale, 1)
        left = np.floor(u - width / 2).astype(int)
        idx = left[:, None] + np.arange(int(np.ceil(width)) + 2)[None, :]
        weights = _cubic((u[:, None] - idx) * min(scale, 1)) * min(scale, 1)
        weights /= weights.sum(1, keepdims=True)
        idx %= 2 * n
        idx = np.where(idx < n, idx, 2 * n - 1 - idx)
        ids = torch.as_tensor(idx, device=out.device)
        wt = torch.as_tensor(weights, device=out.device, dtype=out.real.dtype)
        moved = out.movedim(axis, 0)
        gathered = moved[ids]
        shape = wt.shape + (1,) * (moved.ndim - 1)
        out = (gathered * wt.reshape(shape)).sum(1).movedim(0, axis)
    return out


def resize_xy_region(a, scale, starts, shape, *, device=None):
    """Exact region of the globally resized field, using global boundary indices.

    Reads only the interpolation halo from a (which may stay on CPU). Unlike
    resizing a cropped image, the requested region has no artificial boundary.
    """
    plans = []
    for axis in (0, 1):
        n = a.shape[axis]
        u = (np.arange(starts[axis], starts[axis] + shape[axis]) + 0.5) / scale - 0.5
        width = 4 / min(scale, 1)
        idx = (
            np.floor(u - width / 2).astype(int)[:, None]
            + np.arange(int(np.ceil(width)) + 2)[None, :]
        )
        wt = _cubic((u[:, None] - idx) * min(scale, 1)) * min(scale, 1)
        wt /= wt.sum(1, keepdims=True)
        idx %= 2 * n
        idx = np.where(idx < n, idx, 2 * n - 1 - idx)
        plans.append((idx, wt, int(idx.min()), int(idx.max()) + 1))
    out = a[plans[0][2] : plans[0][3], plans[1][2] : plans[1][3]].to(device or a.device)
    for axis, (idx, wt, lo, _) in enumerate(plans):
        ids = torch.as_tensor(idx - lo, device=out.device)
        weights = torch.as_tensor(wt, device=out.device, dtype=out.real.dtype)
        moved = out.movedim(axis, 0)
        accum = None
        for j in range(idx.shape[1]):
            if not np.any(wt[:, j]):
                continue
            term = moved.index_select(0, ids[:, j]) * weights[:, j].reshape(
                (-1,) + (1,) * (moved.ndim - 1)
            )
            accum = term if accum is None else accum + term
        out = accum.movedim(0, axis)
    return out


def translate_xy(a, shift):
    """imtranslate default linear interpolation, same view, zero fill."""
    h, w = a.shape[:2]
    flat = a.reshape(h, w, -1).permute(2, 0, 1)[:, None]
    theta = torch.tensor(
        [[1, 0, -2 * shift / w], [0, 1, -2 * shift / h]], device=a.device, dtype=a.dtype
    )
    theta = theta[None].expand(flat.shape[0], -1, -1)
    grid = F.affine_grid(theta, flat.shape, align_corners=False)
    return (
        F.grid_sample(flat, grid, align_corners=False).squeeze(1).permute(1, 2, 0).reshape(a.shape)
    )


def v4_interpolate(points, values, queries, chunk=16384, *, backend="auto"):
    """MATLAB v4 interpolation in float64, with a shared solve for value columns.

    Large evaluations use a fused CUDA kernel when CuPy/CUDA are available;
    CPU evaluation uses compiled distances and bounded intermediate arrays.
    ``backend='cpu'`` or ``'cuda'`` explicitly selects an evaluator. The dense
    coefficient solve and Green function are unchanged; no approximation or
    reduced sampling is introduced. ``chunk`` limits query batches.
    """
    from scipy.spatial.distance import cdist

    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be auto, cpu or cuda")
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError("chunk must be a positive integer")
    p, v, q = np.asarray(points, float), np.asarray(values, float), np.asarray(queries, float)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p):
        raise ValueError("points must be a nonempty (N, 2) array")
    if v.ndim not in (1, 2) or len(v) != len(p) or (v.ndim == 2 and v.shape[1] == 0):
        raise ValueError("values must have shape (N,) or (N, C) with C > 0")
    if q.ndim < 1 or q.shape[-1] != 2:
        raise ValueError("queries must have shape (..., 2)")
    shape = q.shape[:-1] + v.shape[1:]
    if len(p) == 1:
        return np.broadcast_to(v[0], shape).copy()
    flat = q.reshape(-1, 2)
    if not len(flat):
        return np.empty(shape, dtype=np.float64)

    def green(x, y):
        d = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            g = d * d * (np.log(d) - 1)
        g[d == 0] = 0
        return g

    coef = np.linalg.solve(green(p, p), v)
    use_cuda = backend == "cuda"
    if backend == "auto" and len(p) * len(flat) >= 16_777_216:
        from importlib.util import find_spec

        use_cuda = torch.cuda.is_available() and find_spec("cupy") is not None
    if use_cuda:
        from .backends.cuda.interpolation import evaluate_v4

        return evaluate_v4(p, coef, flat, chunk=chunk).reshape(shape)

    result = np.empty((len(flat),) + v.shape[1:], dtype=np.float64)
    # Two distance/Green buffers, at most 64 MiB each by default.
    chunk = min(chunk, max(1, 8_388_608 // len(p)))
    for start in range(0, len(flat), chunk):
        d2 = cdist(flat[start : start + chunk], p, "sqeuclidean")
        # r^2(log(r)-1) == r^2(0.5*log(r^2)-1), including g(0)=0.
        with np.errstate(divide="ignore", invalid="ignore"):
            basis = np.log(d2)
            basis *= 0.5
            basis -= 1
            basis *= d2
        basis[d2 == 0] = 0
        result[start : start + chunk] = basis @ coef
    return result.reshape(shape)


def fresnel(u, wavelength, dx, z, phi, nidx, *, crop=None, save_from=0, return_all=False):
    """Scaled Fresnel multi-screen propagation. Hosts scalar control, caches Q.

    Spatial axes are the last two of u; phi has (...,X,Y,screen). Trajectory
    can save only a crop/late planes; propagation itself keeps its full field.
    """
    dx, z = np.asarray(dx, float).ravel(), np.asarray(z, float).ravel()
    if len(z) < 2 or len(dx) != len(z) or np.any(np.diff(z) <= 0):
        raise ValueError("Need increasing z and one dx per plane")
    if phi.shape[-1] not in (len(z), len(z) - 1):
        raise ValueError("Wrong number of phase screens")
    h, w = u.shape[-2:]
    dtype = u.real.dtype
    x = torch.arange(-h / 2, h / 2, device=u.device, dtype=dtype)[:, None]
    y = torch.arange(-w / 2, w / 2, device=u.device, dtype=dtype)[None, :]
    lam = float(wavelength) / float(nidx)
    k = 2 * np.pi / lam
    dz = np.diff(z)
    sc = dx[1:] / dx[:-1]
    u = (
        u
        * torch.exp(1j * k * ((x * dx[0]) ** 2 + (y * dx[0]) ** 2) * (1 - sc[0]) / (2 * dz[0]))
        * phi[..., 0]
    )
    saved = []

    def keep(i, field):
        if return_all and i >= save_from:
            saved.append(field[..., crop[0], crop[1]].clone() if crop else field.clone())

    keep(0, u)
    cache = {}
    dims = (-2, -1)
    for i in range(len(dz)):
        key = (float(dz[i]), float(dx[i]), float(sc[i]))
        if key not in cache:
            cache[key] = torch.exp(
                -1j
                * np.pi
                * lam
                * dz[i]
                * ((x / (h * dx[i])) ** 2 + (y / (w * dx[i])) ** 2)
                / sc[i]
            )
        f = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(u / sc[i], dim=dims)), dim=dims)
        u = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.ifftshift(cache[key] * f, dim=dims)), dim=dims
        )
        if i + 1 < phi.shape[-1]:
            u = u * phi[..., i + 1]
        keep(i + 1, u)
    out = u * torch.exp(
        1j * k / 2 * (sc[-1] - 1) / (sc[-1] * dz[-1]) * ((x * dx[-1]) ** 2 + (y * dx[-1]) ** 2)
    )
    return (out, torch.stack(saved, dim=-1)) if return_all else out
