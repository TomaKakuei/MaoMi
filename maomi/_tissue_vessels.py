# Derived numerical behavior: local NAOMi checkout, Song and Charles.
"""Vascular graph construction and rasterization with upstream conventions.
Positions are one-based; node/connection lists use ordinary Python indices.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from types import SimpleNamespace
import copy
import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline
from .tissue import params, mround
from ._tissue_random import RandomStream


@dataclass
class Node:
    root: object = None
    conn: list = field(default_factory=list)
    pos: object = None
    kind: object = None
    misc: object = None


@dataclass
class Connection:
    start: int = 0
    end: int = 0
    weight: float = np.nan
    locations: object = None
    kind: object = None


@lru_cache(maxsize=64)
def disk(radius):
    """Four-periodic-line disk approximation; exact Euclidean mask for r<3."""
    r = int(radius)
    if r < 3:
        x, y = np.mgrid[-r : r + 1, -r : r + 1]
        return x * x + y * y <= r * r
    extent = 2 * r * np.tan(np.pi / 16)
    points = {(0, 0)}
    for x, y in [(1, 0), (1, 1), (0, 1), (-1, 1)]:
        repeat = int(np.floor(extent / np.hypot(x, y)))
        points = {(a + k * x, b + k * y) for a, b in points for k in range(-repeat, repeat + 1)}
    extra = max(0, r - max(a for a, b in points) - 1)
    if extra:
        points = {
            (a + i, b + j)
            for a, b in points
            for i in range(-extra, extra + 1)
            for j in range(-extra, extra + 1)
        }
    coords = np.array(list(points))
    lo = coords.min(axis=0)
    shape = coords.max(axis=0) - lo + 1
    mask = np.zeros(shape, bool)
    mask[tuple((coords - lo).T)] = True
    return mask


def _dijkstra(matrix, root=0):
    n = len(matrix)
    visit = np.ones(n, bool)
    pending = np.zeros(n)
    pending[root] = 1
    distance = np.full(n, np.inf)
    distance[root] = 0
    previous = np.full(n, -1, int)
    current = root
    while np.any(pending):
        visit[current] = False
        ids = np.flatnonzero(pending)
        current = ids[np.argmin(pending[ids])]
        pending[current] = 0
        for nxt in range(n):
            if visit[nxt]:
                candidate = distance[current] + matrix[current, nxt]
                if candidate < distance[nxt]:
                    pending[nxt] = max(np.finfo(float).eps, candidate)
                    distance[nxt] = candidate
                    previous[nxt] = current
    return distance, previous


def _distance(points):
    p = np.asarray(points)
    return np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(axis=2))


def pseudo_sample(size, count, width, weight, pdf, r):
    size = np.asarray(size, int)
    nd = len(size)
    rad = int(np.ceil(2 * width))
    grid = np.mgrid[tuple(slice(-rad, rad + 1) for _ in range(nd))]
    kernel = (-weight * np.exp(-(grid * grid).sum(axis=0) / width**2)).astype(np.float32)
    pdf = np.array(pdf, np.float32, copy=True)
    positions = np.zeros((count, nd))
    i = 0
    tries = 0
    while i < count and (nd == 3 or tries < 1000):
        tries += 1
        point = np.ceil(r.rand(nd) * size).astype(int)
        if pdf[tuple(point - 1)] - r.rand() > 0:
            ks = []
            ds = []
            for j in range(nd):
                ka = int(mround(max(0, 2 * width + 1 - point[j]) + 1))
                kb = int(mround(min(0, size[j] - point[j] - 2 * width) + 4 * width + 1))
                da = int(mround(max(1, point[j] - 2 * width)))
                db = int(mround(min(size[j], point[j] + 2 * width)))
                if nd == 2:
                    if kb - ka > db - da:
                        ka += (kb - ka) - (db - da)
                    elif kb - ka < db - da:
                        da += (db - da) - (kb - ka)
                ks.append(slice(ka - 1, kb))
                ds.append(slice(da - 1, db))
            pdf[tuple(ds)] += kernel[tuple(ks)]
            positions[i] = point
            i += 1
            tries = 0
    return positions, pdf


def _branch(nodes, surface, p, index, angle, r):
    border = True
    overlap = False
    iterations = 0
    previous = nodes[index].pos[:2].copy()
    parent = index
    last = np.empty((0, 2), int)
    chance = 0
    occupied = ndimage.binary_dilation(surface, structure=disk(int(p.vesrad)))
    size = np.array(surface.shape)
    while iterations < p.maxit and border:
        if r.rand() < chance:
            chance = 0
            other = angle - abs((0.5 + r.rand()) * p.dirvar)
            angle += abs((0.5 + r.rand()) * p.dirvar)
            _branch(nodes, surface, p, parent, other, r)
            overlap = True
        else:
            chance += p.branchp
        length = max(p.varsc * r.randn() + p.lensc, p.mindist)
        position = (
            np.array([np.cos(angle), np.sin(angle)]) * length + previous + p.varpos * r.randn(2)
        )
        if np.any(position < 1) or np.any(position > size):
            position = np.clip(position, 1, size)
            border = False
        path = mround(np.linspace(position, previous, int(np.ceil(length)))).astype(int)
        path = np.vstack((path, path + [0, 1], path + [1, 0]))
        path = np.minimum(path, size)
        if not occupied[tuple((path - 1).T)].any() or overlap:
            nodes.append(Node(parent, [parent], position, "surf"))
            previous = position
            parent = len(nodes) - 1
            iterations += 1
            if len(last):
                surface[tuple((last - 1).T)] = True
            last = path
        else:
            border = False
            last = np.empty((0, 2), int)
        overlap = False
    if len(last):
        surface[tuple((last - 1).T)] = True


def _major(nv, p, v, r):
    nodes = []
    size = nv.size
    for _ in range(nv.nsource):
        flags = r.rand(2) >= np.array([size[1] / (size[0] + size[1]), 0.5])
        if flags[0]:
            position = [np.ceil(size[0] * r.rand()), 1 if flags[1] else size[1], v.depth_surf]
        else:
            position = [1 if flags[1] else size[0], np.ceil(size[1] * r.rand()), v.depth_surf]
        nodes.append(Node(-1, [], np.array(position), "edge", flags))
    surface = np.zeros(tuple(size[:2]), bool)
    for i in range(nv.nsource):
        f = nodes[i].misc
        angle = (
            0.5 if f[0] and f[1] else 1.5 if f[0] else 0.0 if f[1] else 1.0
        ) * np.pi + r.randn() * p.dirvar
        _branch(nodes, surface, p, i, angle, r)
    initial = len(nodes)
    nv.nlinks = initial - nv.nsource
    for node in nodes[nv.nsource :]:
        node.pos = mround(np.r_[node.pos, v.depth_surf])
    surface = ndimage.binary_dilation(surface, structure=disk(int(p.vesrad * 2)))
    sampled, _ = pseudo_sample(size[:2], nv.nsurf, v.mindists[0], v.sepweight, 1 - surface, r)
    points = np.vstack(
        (
            np.array([n.pos for n in nodes]),
            np.column_stack((sampled, np.full(nv.nsurf, v.depth_surf))),
        )
    )
    distances = _distance(points)
    distances[:initial, :initial] = np.inf
    distances[: nv.nsource, : nv.nsource] = 0
    for i, node in enumerate(nodes):
        if node.root is not None and node.root >= 0:
            distances[node.root, i] = distances[i, node.root] = np.linalg.norm(
                node.pos - nodes[node.root].pos
            )
    distances[initial:, :initial] = np.inf
    with np.errstate(invalid="ignore"):
        weighted = distances**v.distWeightScale * (
            1 + v.randWeightScale * r.randn(*distances.shape)
        )
    _, parents = _dijkstra(weighted)
    parents[: nv.nsource] = np.arange(nv.nsource)
    for i in range(nv.nsurf + nv.nsource):
        if parents[i] == 0:
            parents[i] = int(np.argmin(distances[i, : nv.nsource]))
    for i in range(initial, initial + nv.nsurf):
        parent = int(parents[i])
        kind = "edge" if points[i, 0] in (1, size[0]) or points[i, 1] in (1, size[1]) else "surf"
        nodes.append(Node(parent, [parent], points[i], kind))
    for i, node in enumerate(nodes):
        if node.root is not None and node.root >= 0:
            nodes[node.root].conn = sorted(set(nodes[node.root].conn + [i]))
    separation = disk(int(mround(v.mindists[0] * 2)))
    occupied = np.zeros(tuple(size[:2]), bool)

    def block(position):
        point = np.zeros_like(occupied)
        point[tuple(np.asarray(position[:2], int) - 1)] = True
        return ndimage.binary_dilation(point, structure=separation)

    for i, node in enumerate(nodes):
        if node.kind == "surf" and len(node.conn) == 1:
            if not occupied[tuple(node.pos[:2].astype(int) - 1)]:
                node.kind = "sfvt"
                occupied |= block(node.pos)
            else:
                for j in node.conn:
                    nodes[j].conn = [k for k in nodes[j].conn if k != i]
                    if nodes[j].root == i:
                        nodes[j].root = None
                nodes[i] = Node()
    surface_ids = [i for i, n in enumerate(nodes) if n.kind == "surf"]
    while sum(n.kind == "sfvt" for n in nodes) < nv.nvert:
        available = [
            i for i in surface_ids if not occupied[tuple(nodes[i].pos[:2].astype(int) - 1)]
        ]
        if not available:
            break
        index = available[int(np.ceil(r.rand() * len(available))) - 1]
        nodes[index].kind = "sfvt"
        occupied |= block(nodes[index].pos)
    vertical = [i for i, n in enumerate(nodes) if n.kind == "sfvt"]
    nv.nvert = len(vertical)
    for index in vertical:
        while nodes[index].pos[2] < size[2]:
            lateral = r.randn(2)
            vertical_step = max(p.varsc * r.randn() + p.lensc, p.mindist)
            position = np.clip(
                nodes[index].pos
                + np.ceil(np.r_[lateral, 1] * np.r_[p.varpos, p.varpos, vertical_step]),
                1,
                size,
            )
            nodes.append(Node(index, [index], position, "vert"))
            nodes[index].conn = sorted(set(nodes[index].conn + [len(nodes) - 1]))
            index = len(nodes) - 1
    nv.nnodes = len(nodes)
    for node in nodes:
        if len(node.conn) == 1:
            node.misc = v.vesSize[2] + r.draw(
                "gamrnd", shape_parameter=3, scale=(v.vesSize[1] - v.vesSize[2]) / 3
            )
    weights = {}
    for i, node in enumerate(nodes):
        if len(node.conn) == 1:
            j = i
            while nodes[j].root is not None and nodes[j].root >= 0:
                parent = nodes[j].root
                weights[j, parent] = np.sqrt(weights.get((j, parent), 0) ** 2 + node.misc**2)
                j = parent
    connections = [
        Connection(i, j, w)
        for (i, j), w in sorted(weights.items(), key=lambda x: (x[0][1], x[0][0]))
    ]
    nv.nconn = len(connections)
    return nodes, connections


def _raster(nodes, connections, size, r, indices=None, volume=None):
    size = np.asarray(size, int)
    volume = np.zeros(tuple(size), bool) if volume is None else volume
    for i in range(len(connections)) if indices is None else indices:
        c = connections[i]
        left = sorted(set(nodes[c.start].conn) ^ {c.end})
        right = sorted(set(nodes[c.end].conn) ^ {c.start})
        before = left[r.integer(len(left)) - 1] if left else None
        after = right[r.integer(len(right)) - 1] if right else None
        points = [nodes[c.start].pos, nodes[c.end].pos]
        if before is not None:
            points.insert(0, nodes[before].pos)
        if after is not None:
            points.append(nodes[after].pos)
        points = np.asarray(points).T
        steps = np.linalg.norm(np.diff(points, axis=1), axis=0)
        steps[steps == 0] = 1e-6
        t = np.r_[0, np.cumsum(steps)]
        bc = (
            (
                (1, (points[:, 1] - points[:, 0]) / steps[0]),
                (1, (points[:, 2] - points[:, 1]) / steps[1]),
            )
            if points.shape[1] == 3
            else "not-a-knot"
        )
        curve = CubicSpline(t, points, axis=1, bc_type=bc)
        start = 1 if before is not None else 0
        count = int(2 * np.linalg.norm(nodes[c.start].pos - nodes[c.end].pos))
        positions = np.unique(
            np.clip(np.ceil(curve(np.linspace(t[start], t[start + 1], count)).T), 1, size).astype(
                int
            ),
            axis=0,
        )
        c.locations = positions
        radius = int(np.ceil(c.weight))
        lo = np.maximum(positions.min(axis=0) - radius, 1)
        hi = np.minimum(positions.max(axis=0) + radius, size)
        crop = np.zeros(tuple(hi - lo + 1), bool)
        crop[tuple((positions - lo).T)] = True
        grid = np.mgrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
        kernel = (grid * grid).sum(axis=0) <= c.weight**2
        crop = ndimage.binary_dilation(crop, structure=kernel)
        sl = tuple(slice(lo[j] - 1, hi[j]) for j in range(3))
        volume[sl] |= crop
    return volume


def _capillaries(nodes, connections, volume, nv, v, r):
    radius = int(np.ceil(v.mindists[0] / nv.vres))
    grid = np.mgrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-2 * (grid * grid).sum(axis=0) / radius**2)
    size_um = np.asarray(nv.szum, int)
    suppression = np.zeros(tuple(size_um), np.float32)
    for c in connections:
        positions = np.ceil(c.locations / nv.vres).astype(int)[:: max(1, int(np.floor(radius / 3)))]
        for position in positions:
            lo = np.maximum(1, position - radius)
            hi = np.minimum(size_um, position + radius)
            kl = lo - (position - radius)
            kh = 2 * radius + 1 - (position + radius - hi)
            dst = tuple(slice(lo[k] - 1, hi[k]) for k in range(3))
            src = tuple(slice(kl[k], kh[k]) for k in range(3))
            suppression[dst] = np.maximum(suppression[dst], kernel[src])
    points, _ = pseudo_sample(
        size_um, nv.ncapp, v.mindists[2] / nv.vres, v.sepweight, 1 - suppression, r
    )
    points = points * nv.vres + 1 - r.draw("randi", points.shape, high=max(1, nv.vres))
    per_vertical = np.atleast_1d(
        r.draw("randi", (nv.nvert,), high=int(np.ceil(nv.szum[2] / v.vesFreq[2])))
    ).astype(int)
    attached = int(per_vertical.sum())
    nv.nvert_sum = attached
    if len(v.vesSize) < 4:
        v.vesSize = np.r_[v.vesSize, 0]
    vertical_ids = [i for i, n in enumerate(nodes) if n.kind == "sfvt"]
    for i, start in enumerate(vertical_ids):
        chain = [start]
        while True:
            children = [
                j for j in nodes[chain[-1]].conn if nodes[j].kind == "vert" and j not in chain
            ]
            if not children:
                break
            chain.extend(children)
        chain = chain[1:]
        for _ in range(per_vertical[i]):
            root = chain[r.integer(len(chain)) - 1]
            chosen = int(np.nanargmin(((points - nodes[root].pos) ** 2).sum(axis=1)))
            node_id = len(nodes)
            nodes.append(Node(root, [root], points[chosen].copy(), "capp"))
            nodes[root].conn = sorted(set(nodes[root].conn + [node_id]))
            connections.append(
                Connection(node_id, root, max(1, r.normal(v.vesSize[2], v.vesSize[3])), kind="vtcp")
            )
            points[chosen] = np.nan
    for position in points:
        if not np.isnan(position).any():
            nodes.append(Node(pos=position.copy(), kind="capp"))
    anchored = [i for i, n in enumerate(nodes) if n.kind == "capp" and n.root is not None]
    free = [i for i, n in enumerate(nodes) if n.kind == "capp" and n.root is None]
    ids = anchored + free
    points = np.array([nodes[i].pos for i in ids])
    count = len(ids)
    if count != nv.ncapp:
        raise ValueError("Capillary node count differs")
    distances = _distance(points)
    np.fill_diagonal(distances, np.inf)
    distances[:attached, :attached] = np.inf
    adj = np.zeros((count, count), bool)
    nearest = np.argmin(distances, axis=0)
    ar = np.arange(count)
    adj[ar, nearest] = True
    adj[nearest, ar] = True
    distances[ar, nearest] = np.inf
    distances[nearest, ar] = np.inf
    distances[distances > v.maxcappdist] = np.inf
    for i in range(count):
        if adj[i].sum() >= 3:
            distances[i, :] = np.inf
            distances[:, i] = np.inf
        distances[i, adj[i]] = np.inf
        distances[adj[i], i] = np.inf
    for i in range(attached, count):
        for j in range(i + 1, count):
            if np.isfinite(distances[i, j]):
                line = np.ceil(np.linspace(points[i], points[j], int(2 * distances[i, j]))).astype(
                    int
                )
                if volume[tuple((line - 1).T)].any():
                    distances[i, j] = distances[j, i] = np.inf
    running = True
    while running:
        degrees = adj.sum(axis=0)
        if np.min(degrees[attached:]) > 1:
            running = False
        else:
            eligible = np.flatnonzero(degrees == 1)
            if np.min(distances[:, eligible]) == np.inf:
                running = False
            i = eligible[int(np.ceil(len(eligible) * r.rand())) - 1]
            if np.min(distances[i]) < np.inf:
                inverse = 1 / distances[i] ** v.distsc
                cdf = np.r_[0, np.cumsum(inverse) / inverse.sum()]
                j = int(np.flatnonzero(np.diff(cdf > r.rand()))[0])
                adj[i, j] = adj[j, i] = True
                distances[adj[i], j] = np.inf
                distances[adj[j], i] = np.inf
                distances[j, adj[i]] = np.inf
                distances[i, adj[j]] = np.inf
                distances[i, j] = distances[j, i] = np.inf
                if adj[j].sum() >= 3:
                    distances[j, :] = np.inf
                    distances[:, j] = np.inf
    cmap = np.full((len(nodes), len(nodes)), -1, int)
    row, col = np.unravel_index(
        np.flatnonzero(np.triu(adj).ravel(order="F")), (count, count), order="F"
    )
    for i, j in zip(row, col):
        start, end = ids[i], ids[j]
        nodes[start].conn = sorted(set(nodes[start].conn + [end]))
        nodes[end].conn = sorted(set(nodes[end].conn + [start]))
        cmap[start, end] = len(connections)
        connections.append(Connection(start, end, kind="capp"))
    queue = []
    for c in connections:
        if c.kind == "vtcp":
            for j in nodes[c.start].conn:
                if cmap[c.start, j] >= 0:
                    queue.append(cmap[c.start, j])
    for i, c in enumerate(connections):
        if c.kind == "vtcp":
            cmap[c.end, c.start] = i
    valid = cmap >= 0
    cmap = np.where(valid, cmap, cmap.T)

    def infer(weights):
        if np.isnan(weights).any():
            return np.nan, False
        if len(weights) == 1:
            return weights[0], True
        u = r.rand()
        if not len(weights):
            return None, False
        lower = max(weights) ** 2 - min(weights) ** 2
        upper = max(weights) ** 2 + min(weights) ** 2
        return np.sqrt(u * (upper - lower) + lower), False

    while queue:
        index = queue.pop(0)
        c = connections[index]
        if np.isnan(c.weight):
            a = cmap[c.start, nodes[c.start].conn]
            a = a[(a != index) & (a >= 0)]
            b = cmap[c.end, nodes[c.end].conn]
            b = b[(b != index) & (b >= 0)]
            wa = np.array([connections[k].weight for k in a])
            wb = np.array([connections[k].weight for k in b])
            x, xf = infer(wa)
            y, yf = infer(wb)
            if x is None or y is None:
                weight = None
            elif np.isnan(x):
                weight = max(1, r.normal(v.vesSize[2], v.vesSize[3])) if np.isnan(y) else y
            elif np.isnan(y):
                weight = x
            else:
                weight = x if xf else y if yf else (x + y) / 2
            c.weight = np.nan if weight is None else weight
            queue.extend(b[np.isnan(wb)])
            queue.extend(a[np.isnan(wa)])
    for c in connections:
        if np.isnan(c.weight):
            c.weight = max(1, r.normal(v.vesSize[2], v.vesSize[3]))
    nv.nnodes = len(nodes)
    nv.nconn = len(connections)
    return nodes, connections


def simulate_vessels(vol_params, vasc_params, *, random_stream=None):
    vol = params(vol_params)
    v = copy.copy(params(vasc_params))
    r = RandomStream() if random_stream is None else random_stream
    vres = float(vol.vres)
    v.vesFreq = np.asarray(v.vesFreq, dtype=float)
    v.depth_surf *= vres
    v.mindists = v.vesFreq * vres / 2
    v.maxcappdist = 2 * v.vesFreq[2] * vres
    v.vesSize = np.asarray(v.vesSize) * vres
    p = copy.copy(params(v.node_params))
    for name in ["lensc", "varsc", "mindist", "varpos"]:
        setattr(p, name, getattr(p, name) * vres)
    p.vesrad = int(np.ceil(p.vesrad * vres))
    extent = np.asarray(
        getattr(vol, "vasc_sz", np.r_[vol.vol_sz[:2], float(vol.vol_sz[2]) + vol.vol_depth]),
        dtype=float,
    )
    nv = SimpleNamespace(vol_sz=extent, size=(extent * vres).astype(int), szum=extent, vres=vres)
    nv.nsource = int(
        max(
            0,
            mround(2 * (extent[0] + extent[1]) / v.sourceFreq * abs(1 + v.vesNumScale * r.randn())),
        )
    )
    nv.nvert = int(
        max(
            0,
            mround(extent[0] * extent[1] / v.vesFreq[1] ** 2 * abs(1 + v.vesNumScale * r.randn())),
        )
    )
    nv.nsurf = int(
        max(
            0,
            mround(extent[0] * extent[1] / v.vesFreq[0] ** 2 * abs(1 + v.vesNumScale * r.randn())),
        )
    )
    nv.ncapp = int(
        max(0, mround(np.prod(extent) / v.vesFreq[2] ** 3 * abs(1 + v.vesNumScale * r.randn())))
    )
    nodes, connections = _major(nv, p, v, r)
    for c in connections:
        if nodes[c.start].kind in ["edge", "surf", "sfvt"]:
            nodes[c.start].pos[2] = min(
                nodes[c.start].pos[2] + np.ceil(c.weight / len(nodes[c.start].conn)), nv.size[2]
            )
        if nodes[c.end].kind in ["edge", "surf", "sfvt"]:
            nodes[c.end].pos[2] = min(
                nodes[c.end].pos[2] + np.ceil(c.weight / len(nodes[c.start].conn)), nv.size[2]
            )
    volume = _raster(nodes, connections, nv.size, r)
    nodes, connections = _capillaries(nodes, connections, volume, nv, v, r)
    _raster(
        nodes,
        connections,
        nv.size,
        r,
        indices=[i for i, c in enumerate(connections) if c.locations is None],
        volume=volume,
    )
    crop_shape = (np.r_[vol.vol_sz[:2], vol.vol_sz[2] + vol.vol_depth] * vres).astype(int)
    offset = np.ceil((nv.size - crop_shape) / 2).astype(int)
    crop = volume[tuple(slice(offset[k], offset[k] + crop_shape[k]) for k in range(3))].copy()
    return crop, vasc_params, volume
