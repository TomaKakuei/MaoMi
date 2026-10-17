# Derived numerical behavior: local NAOMi checkout, Song and Charles.
# Surface membership uses the independent solid-angle implementation.
"""NAOMi tissue geometry and fluorescence, with explicit MATLAB index contracts.

Sparse indices are one-based column-major, compatible with movie.scan_volume.
"""

from types import SimpleNamespace
import math
import numpy as np
from scipy import ndimage, fft
from .anatomy import generate_neural_body, spiral_sample_sphere
from ._tissue_random import RandomStream


def params(p):
    return SimpleNamespace(**p) if isinstance(p, dict) else p


def mround(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


from .geometry import inside_mesh


def sample_neurons(neur_params, vol_params, vessels, *, random_stream=None):
    p = params(neur_params)
    v = params(vol_params)
    r = RandomStream() if random_stream is None else random_stream
    radius = v.min_dist / 2
    lim = int(np.ceil(radius))
    xyz = np.mgrid[-lim : lim + 1, -lim : lim + 1, -lim : lim + 1]
    dilated = ndimage.binary_dilation(vessels, structure=(xyz * xyz).sum(axis=0) <= radius**2)
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    depth = int(v.vol_depth * v.vres)
    good = ~dilated[: shape[0], : shape[1], depth : depth + shape[2]]
    bad = good.copy()
    grids = np.meshgrid(
        *[np.linspace(0, v.vol_sz[i], shape[i], dtype=np.float32) for i in range(3)], indexing="ij"
    )
    sphere, tri = spiral_sample_sphere(p.n_samps)
    p = SimpleNamespace(**(vars(p) | dict(S_samp=sphere, Tri=tri)))
    locations = []
    cells = []
    nuclei = []
    angles = []
    while good.sum() > 1 and len(cells) < v.N_neur:
        cell, nucleus, _, ang = generate_neural_body(p, rng=r)
        ids = np.flatnonzero(good.ravel(order="F"))
        chosen = ids[r.rank(len(ids)) - 1]
        new = np.array([g.ravel(order="F")[chosen] for g in grids])
        if v.N_neur == 1:
            new = mround(np.asarray(v.vol_sz) / 2)
        distance = np.sqrt(sum((g - new[j]) ** 2 for j, g in enumerate(grids)))
        good[distance <= 1.1 * v.min_dist] = False
        bad[distance <= v.min_dist] = False
        good = ~(good | ~bad)
        locations.append(new)
        cells.append(cell)
        nuclei.append(nucleus)
        angles.append(ang)
    locations = np.asarray(locations, np.float32)
    cells = (np.stack(cells, axis=2) + locations.T[None, :, :]).astype(np.float32)
    nuclei = (np.stack(nuclei, axis=2) + locations.T[None, :, :]).astype(np.float32)
    return locations, cells, nuclei, tri, np.asarray(angles)


def voxelize_neurons(neur_params, vol_params, locations, cells, nuclei, vessels):
    p = params(neur_params)
    v = params(vol_params)
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    soma = np.zeros(shape, np.uint16)
    volume = np.zeros(shape, np.float32)
    gp_nuc = []
    gp_soma = []
    depth = int(v.vol_depth * v.vres)
    taken = np.array(vessels[:, :, depth : depth + shape[2]], bool)
    _, tri = spiral_sample_sphere(p.n_samps)
    if cells.ndim == 2:
        cells = cells[..., None]
        nuclei = nuclei[..., None]
    for j in range(int(v.N_neur)):
        radius = np.ceil(np.sqrt(((cells[:, :, j] - locations[j]) ** 2).sum(axis=1)).max())
        extent = int(np.ceil(radius * v.vres))
        center = mround(v.vres * locations[j]).astype(int)
        indices = [
            np.arange(max(1, center[i] - extent), min(center[i] + extent, shape[i]) + 1)
            for i in range(3)
        ]
        grid = np.meshgrid(
            *[((a - center[i]) / v.vres).astype(np.float32) for i, a in enumerate(indices)],
            indexing="ij"
        )
        test = np.sqrt(sum(g * g for g in grid)) <= radius
        points = np.column_stack(
            [
                g[test] + np.float32(center[i] / v.vres + 1 / (2 * v.vres))
                for i, g in enumerate(grid)
            ]
        )
        a = np.zeros(test.shape, bool)
        b = a.copy()
        a[test] = inside_mesh(cells[:, :, j], tri, points).astype(bool)
        b[test] = inside_mesh(nuclei[:, :, j], tri, points).astype(bool)
        ix = np.ix_(*[z - 1 for z in indices])
        cytosol = a & ~b & ~taken[ix]
        taken[ix] |= cytosol
        sub = soma[ix]
        sub[cytosol] = j + 1
        soma[ix] = sub
        coords = np.meshgrid(*[z - 1 for z in indices], indexing="ij")
        linear = coords[0] + shape[0] * coords[1] + shape[0] * shape[1] * coords[2]
        si = np.sort(linear[cytosol]).astype(np.int32) + 1
        ni = np.sort(linear[b]).astype(np.int32) + 1
        gp_soma.append(si)
        gp_nuc.append([ni, p.nuc_fluorsc])
        flat = volume.ravel(order="F").copy()
        flat[ni - 1] = p.nuc_fluorsc
        volume = flat.reshape(shape, order="F")
    return soma, volume, gp_nuc, gp_soma


def gaussian_field(shape, length, amplitude, *, random_stream=None):
    r = RandomStream() if random_stream is None else random_stream
    shape = tuple(np.asarray(shape, int))
    length = np.broadcast_to(length, (3,))
    axes = [np.linspace(-np.pi / 2, np.pi / 2, n, dtype=np.float32) ** 2 for n in shape]
    kernel = (
        np.exp(-axes[0][:, None, None] * np.float32(length[0] ** 2))
        * np.exp(-axes[1][None, :, None] * np.float32(length[1] ** 2))
        * np.exp(-axes[2][None, None, :] * np.float32(length[2] ** 2))
    )
    keep = kernel > 1e-10
    scale = np.float32(np.sqrt(np.prod(length)))
    if keep.sum() < np.prod(shape) / 2:
        count = int(keep.sum())
        a = r.draw("randn", (count,)).astype(np.float32)
        b = r.draw("randn", (count,)).astype(np.float32)
        flat = np.zeros(np.prod(shape), np.complex64)
        mask = keep.ravel(order="F")
        flat[mask] = scale * (a + 1j * b) * kernel.ravel(order="F")[mask]
        spectral = flat.reshape(shape, order="F")
    else:
        a = r.draw("randn", shape).astype(np.float32)
        b = r.draw("randn", shape).astype(np.float32)
        spectral = scale * (a + 1j * b) * kernel
    field = (
        np.float32(np.sqrt(np.prod(shape))) * fft.ifftshift(fft.ifftn(fft.ifftshift(spectral))).real
    )
    return np.float32(amplitude * (2**4.5 / np.pi**1.5)) * field


def set_fluorescence(
    vol_params,
    neur_params,
    dend_params,
    labels,
    soma,
    apical,
    locations,
    volume,
    *,
    random_stream=None
):
    v = params(vol_params)
    n = params(neur_params)
    d = params(dend_params)
    r = RandomStream() if random_stream is None else random_stream
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    out = volume.ravel(order="F").copy()
    result = []
    for j in range(int(v.N_neur + v.N_den)):
        idx = np.flatnonzero(labels.ravel(order="F") == j + 1)
        sm = soma.ravel(order="F")[idx] == j + 1
        ad = apical.ravel(order="F")[idx] == j + 1
        vals = np.ones(len(idx), np.float32)
        if j < v.N_neur:
            gp = gaussian_field(
                mround(n.avg_rad * 6 * v.vres * np.ones(3)).astype(int),
                n.fluor_dist[0] * v.vres,
                n.fluor_dist[1],
                random_stream=r,
            )
            coord = np.column_stack(np.unravel_index(idx[sm], shape, order="F")) + 1
            positions = (
                np.clip(
                    coord - np.floor(v.vres * locations[j]) + np.floor(gp.shape[0] / 2), 1, gp.shape
                ).astype(int)
                - 1
            )
            values = gp[tuple(positions.T)]
            diff = values - values.mean()
            den = np.abs(diff).max() if len(diff) else 0
            values = 0.5 * diff / den + 1 if den else np.ones_like(values)
            coord = np.column_stack(np.unravel_index(idx, shape, order="F")).astype(np.float32) + 1
            distance = np.sqrt(((coord - v.vres * locations[j]) ** 2).sum(axis=1))
            w = d.weightScale
            vals = ((w[1] * np.exp(-distance / (v.vres * w[0])) + 1 - w[1]) * (1 - w[2])).astype(
                np.float32
            )
            vals[sm] = values
            vals[ad] = 1
        else:
            sm = np.zeros(len(idx), bool)
        result.append([idx.astype(np.int32) + 1, vals, sm])
        out[idx] = vals
    return result, out.reshape(shape, order="F")
