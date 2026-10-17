# Derived numerical behavior: local NAOMi checkout, Song and Charles.
"""Tissue growth stages using one-based MATLAB sparse component indices."""

import copy
import numpy as np
from .tissue import params, mround
from ._tissue_random import RandomStream
from ._tissue_paths import random_walk


def _cell_rows(a, width):
    if isinstance(a, np.ndarray) and a.shape == (width,):
        return [a]
    return list(a)


def _baseline(shape, components, nuclei):
    flat = np.zeros(np.prod(shape), np.float32)
    for j, row in enumerate(components):
        flat[np.asarray(row[0], int).ravel() - 1] = np.asarray(row[1]).ravel()
        if j < len(nuclei):
            flat[np.asarray(nuclei[j][0], int).ravel() - 1] = nuclei[j][1]
    return flat


def dilate_paths(paths, numbers, obstruction, *, random_stream=None):
    r = RandomStream() if random_stream is None else random_stream
    shape = paths.shape
    paths = np.array(paths, np.float32, copy=True)
    paths[np.asarray(obstruction, bool)] = np.nan
    p = paths.ravel(order="F").copy()
    labels = np.asarray(numbers).ravel(order="F").copy()
    count = len(p)
    grid = np.mgrid[-20:21, -20:21, -20:21]
    squared = (grid * grid).sum(axis=0).ravel(order="F")
    order = np.argsort(squared, kind="stable")
    ends = np.flatnonzero(np.diff(squared[order])) + 1
    neighbors = np.array([-shape[0] * shape[1], shape[0] * shape[1], -shape[0], shape[0], -1, 1])
    for i in range(min(399, len(ends) - 1)):
        roots = np.flatnonzero(p > 1)
        if not len(roots):
            break
        xyz = (
            np.column_stack(np.unravel_index(order[ends[i] : ends[i + 1]], (41, 41, 41), order="F"))
            - 20
        )
        shifts = xyz[:, 2] * shape[0] * shape[1] - xyz[:, 1] * shape[0] - xyz[:, 0]
        for root in roots:
            candidates = root + shifts
            candidates = candidates[(candidates >= 0) & (candidates < count - 1)]
            candidates = candidates[p[candidates] == 0]
            connected = []
            for candidate in candidates:
                nearby = candidate + neighbors
                nearby = nearby[(nearby >= 0) & (nearby < count - 1)]
                if np.any(labels[nearby] == labels[root]):
                    connected.append(candidate)
            while p[root] > 1 and connected:
                chosen = connected.pop(r.integer(len(connected)) - 1)
                p[root] -= 1
                p[chosen] = 1
                labels[chosen] = labels[root]
    return p.reshape(shape, order="F"), labels.reshape(shape, order="F")


def generate_background(
    vol_params,
    bg_params,
    dend_params,
    volume,
    labels,
    components,
    nuclei,
    locations,
    *,
    random_stream=None
):
    v = copy.copy(params(vol_params))
    b = params(bg_params)
    d = params(dend_params)
    r = RandomStream() if random_stream is None else random_stream
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    size = np.array(shape)
    components = copy.deepcopy(_cell_rows(components, 3))
    nuclei = _cell_rows(nuclei, 2)
    free = labels.ravel(order="F") == 0
    for row in nuclei:
        free[np.asarray(row[0], int).ravel() - 1] = False
    free = free.reshape(shape, order="F")
    out = _baseline(shape, components, nuclei)
    dt = np.array(d.dtParams, copy=True)
    dt[1:3] *= v.vres
    thickness = d.thicknessScale * v.vres**2
    M = r.draw("rand", shape).astype(np.float32)
    M[~free] = np.finfo(np.float32).max
    M[[0, -1], :, :] = np.finfo(np.float32).max
    M[:, [0, -1], :] = np.finfo(np.float32).max
    M[:, :, [0, -1]] = np.finfo(np.float32).max
    idxvol = np.zeros(np.prod(shape), np.uint16)
    numvol = np.zeros(np.prod(shape), np.float32)
    count = 0
    locations = list(np.atleast_2d(locations))
    extent = np.array([dt[1], dt[1], dt[2]])
    trials = int(np.floor((np.prod(size + 2 * extent) / np.prod(size) - 1) * v.N_neur))
    for _ in range(trials):
        root = np.floor(r.rand(3) * (size + 2 * extent) - extent)
        while np.all(root > 0) and np.all(root <= size):
            root = np.floor(r.rand(3) * (size + 2 * extent) - extent)
        locations.append((root / v.vres).astype(np.float32))
        dendritic = []
        for _ in range(int(dt[0])):
            theta = r.rand() * 2 * np.pi
            radius = np.sqrt(r.rand()) * dt[1]
            end = np.floor(
                [
                    radius * np.cos(theta) + root[0],
                    radius * np.sin(theta) + root[1],
                    2 * dt[2] * (r.rand() - 0.5) + root[2],
                ]
            )
            if np.all(end > 0) and np.all(end <= size):
                with np.errstate(invalid="ignore", divide="ignore"):
                    shift = np.r_[
                        (root < 1) * (1 - root) / (end - root),
                        (root > size) * (size - root) / (end - root),
                    ]
                # MATLAB max omits NaN by default.
                axis = int(np.nanargmax(shift))
                distance = shift[axis]
                for _ in range(30):
                    start = mround(distance * (end - root) + root)
                    for k in range(3):
                        if k != axis % 3:
                            start[k] += r.integer(3)
                    start = np.clip(start, 1, size).astype(int)
                    path = (
                        random_walk(
                            M,
                            start - 1,
                            end.astype(int) - 1,
                            b.distsc,
                            b.maxlength,
                            b.fillweight,
                            b.maxel,
                            b.minlength,
                        )
                        + 1
                    )
                    if len(path):
                        path = np.vstack((start, path))
                        width = (
                            max(0, float(r.draw("normrnd", mean=1, sd=getattr(d, "dendVar", 0.25))))
                            ** 2
                        )
                        if len(path) > 2:
                            weights = width * (
                                1
                                - (1 - 1 / np.sqrt(2))
                                * np.r_[
                                    0,
                                    np.abs(np.diff(np.abs(np.diff(path, axis=0)), axis=0)).sum(
                                        axis=1
                                    )
                                    / 2,
                                    0,
                                ]
                            ).astype(np.float32)
                        else:
                            weights = width * np.ones(len(path))
                        ids = (
                            path[:, 0]
                            - 1
                            + (path[:, 1] - 1) * shape[0]
                            + (path[:, 2] - 1) * shape[0] * shape[1]
                        )
                        dendritic.extend(ids)
                        numvol[ids] = weights
                        break
        if dendritic:
            count += 1
            ids = np.asarray(dendritic, int)
            idxvol[ids] = count
            numvol[ids] = numvol[ids] * thickness * dt[3]
    _, grown = dilate_paths(
        numvol.reshape(shape, order="F"), idxvol.reshape(shape, order="F"), ~free, random_stream=r
    )
    v.N_den2 = count
    ncomp = int(v.N_neur + v.N_den)
    grown[grown > 0] += ncomp
    labels = (labels + grown).astype(np.uint16)
    w = d.weightScale
    for i in range(ncomp, ncomp + count):
        ids = np.flatnonzero(labels.ravel(order="F") == i + 1)
        values = (w[1] * np.exp(-(dt[1] / v.vres) / w[0]) + 1 - w[1]) * (
            1 - w[2] * r.rand(len(ids))
        )
        components.append([ids.astype(np.int32) + 1, values, np.zeros(len(ids), bool)])
        out[ids] = values
    return labels, out.reshape(shape, order="F"), v, components, np.asarray(locations, np.float32)


def generate_axons(
    vol_params, axon_params, volume, labels, components, nuclei, *, random_stream=None
):
    v = copy.copy(params(vol_params))
    a = params(axon_params)
    r = RandomStream() if random_stream is None else random_stream
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    size = np.array(shape)
    components = _cell_rows(components, 3)
    nuclei = _cell_rows(nuclei, 2)
    free = labels.ravel(order="F") == 0
    for row in nuclei:
        free[np.asarray(row[0], int).ravel() - 1] = False
    fill = int(mround(a.maxfill * a.maxvoxel * free.sum()))
    free = free.reshape(shape, order="F")
    out = _baseline(shape, components, nuclei)
    pad = int(a.padsize)
    padded = size + 2 * pad
    M = r.draw("rand", padded).astype(np.float32)
    M[np.pad(~free, pad)] = np.finfo(np.float32).max
    background = []
    attempts = 0
    while fill > 0 and len(background) < v.N_bg and attempts < 10000:
        path = np.empty((0, 3), int)
        attempts = 0
        while len(path) < a.minlength and attempts < 10000:
            attempts += 1
            root = np.ceil((padded - 2) * r.rand(3) + 1).astype(int)
            while M[tuple(root - 1)] > a.fillweight * a.maxvoxel:
                root = np.ceil((padded - 2) * r.rand(3) + 1).astype(int)
            end = np.clip(
                np.ceil(root + 2 * a.maxdist * v.vres * (r.rand(3) - 0.5)), 1, padded
            ).astype(int)
            path = (
                random_walk(
                    M,
                    root - 1,
                    end - 1,
                    a.distsc,
                    a.maxlength,
                    a.fillweight,
                    a.maxvoxel,
                    a.minlength,
                )
                + 1
            )
        if not len(path):
            continue
        branches = int(max(0, mround(a.numbranches + a.varbranches * r.randn())))
        for _ in range(branches):
            branch = np.empty((0, 3), int)
            for _ in range(100):
                root = path[int(np.ceil(r.rand() * len(path))) - 1].copy()
                while np.any(root == 1) or np.any(root == padded):
                    root = path[int(np.ceil(r.rand() * len(path))) - 1].copy()
                end = np.clip(
                    root + mround(2 * a.maxdist * v.vres * (r.rand(3) - 0.5)), 1, padded
                ).astype(int)
                branch = (
                    random_walk(
                        M,
                        root - 1,
                        end - 1,
                        a.distsc,
                        a.maxlength,
                        a.fillweight,
                        a.maxvoxel,
                        a.minlength,
                    )
                    + 1
                )
                if len(branch) >= a.minlength:
                    break
            path = np.vstack((path, branch))
        path = path - pad
        path = path[np.all((path > 0) & (path <= size), axis=1)]
        if len(path):
            ids = path[:, 0] - 1 + (path[:, 1] - 1) * size[0] + (path[:, 2] - 1) * size[0] * size[1]
            values = np.full(
                len(ids), (1 / a.maxel) * max(0, 1 + a.varfill * r.randn()), np.float32
            )
            background.append([ids.astype(np.int32) + 1, values])
            fill -= len(ids)
            out[ids] = out[ids] + values
    count = min(len(background) + 1, int(v.N_bg))
    if count > len(background):
        background.append([np.array([], np.int32), np.array([], np.float32)])
    v.N_bg = count
    return out.reshape(shape, order="F"), background, a, v


def sort_axons(vol_params, axon_params, background, locations, *, random_stream=None):
    v = params(vol_params)
    a = params(axon_params)
    r = RandomStream() if random_stream is None else random_stream
    count = int(a.N_proc)
    shape = tuple((np.asarray(v.vol_sz) * v.vres).astype(int))
    background = _cell_rows(background, 2)
    out = [[np.array([], np.int32), np.array([], np.float32)] for _ in range(count)]
    selected = set()
    ncomp = int(v.N_neur + v.N_den)
    if count > ncomp:
        centers = np.zeros((len(background), 3))
        for j, (ids, values) in enumerate(background):
            if np.size(ids):
                centers[j] = (
                    np.column_stack(
                        np.unravel_index(np.asarray(ids, int) - 1, shape, order="F")
                    ).mean(axis=0)
                    + 1
                )
        distances = np.sqrt(
            ((np.asarray(locations)[:ncomp, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        )
        for j in range(ncomp):
            k = int(np.argmin(distances[j]))
            distances[:, k] = np.inf
            out[j] = list(background[k])
            selected.add(k)
    for j, row in enumerate(background):
        if j in selected:
            continue
        k = (
            ncomp + int(np.ceil((count - ncomp) * r.rand())) - 1
            if count > ncomp
            else int(np.ceil(count * r.rand())) - 1
        )
        for i in (0, 1):
            out[k][i] = np.r_[out[k][i], np.asarray(row[i]).ravel()]
    return out
