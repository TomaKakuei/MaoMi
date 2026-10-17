# Derived numerical behavior: local NAOMi checkout, Song and Charles.
"""Compiled path kernels preserving the local upstream MEX arithmetic.

Internal path coordinates are zero-based. Upstream quirks (destination edge
costs, excluded voxel zero, random-walk final-step truncation) are intentional
compatibility behavior, not assertions of physical or algorithmic correctness.
"""

import heapq
import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _walk(M, root, ends, distsc, maxlength, fillweight, maxel, minlength):
    shape = M.shape
    curr = root.copy()
    path = np.zeros((maxlength, 3), np.int32)
    values = np.zeros(maxlength, np.float32)
    mat = np.zeros(6, np.float32)
    maxfill = np.float32(maxel * fillweight)
    bglength = 0
    i = 0
    while i < maxlength:
        distance = np.float32(
            np.sqrt(
                float(
                    (ends[0] - curr[0]) ** 2 + (ends[1] - curr[1]) ** 2 + (ends[2] - curr[2]) ** 2
                )
            )
            / distsc
        )
        dv = np.empty(3, np.float32)
        for k in range(3):
            dv[k] = np.float32((curr[k] - ends[k]) / distance)
        jmin = 6
        minimum = np.float32(3.4028234663852886e38)
        if curr[0] != shape[0] - 1:
            mat[0] = M[curr[0] + 1, curr[1], curr[2]] + dv[0]
            if mat[1] < minimum:
                jmin = 0
                minimum = mat[0]
        if curr[1] != shape[1] - 1:
            mat[1] = M[curr[0], curr[1] + 1, curr[2]] + dv[1]
            if mat[1] < minimum:
                jmin = 1
                minimum = mat[1]
        if curr[2] != shape[2] - 1:
            mat[2] = M[curr[0], curr[1], curr[2] + 1] + dv[2]
            if mat[2] < minimum:
                jmin = 2
                minimum = mat[2]
        if curr[0] != 0:
            mat[3] = M[curr[0] - 1, curr[1], curr[2]] - dv[0]
            if mat[3] < minimum:
                jmin = 3
                minimum = mat[3]
        if curr[1] != 0:
            mat[4] = M[curr[0], curr[1] - 1, curr[2]] - dv[1]
            if mat[4] < minimum:
                jmin = 4
                minimum = mat[4]
        if curr[2] != 0:
            mat[5] = M[curr[0], curr[1], curr[2] - 1] - dv[2]
            if mat[5] < minimum:
                jmin = 5
                minimum = mat[5]
        if minimum < maxfill:
            if jmin < 3:
                curr[jmin] += 1
            else:
                curr[jmin - 3] -= 1
            path[i] = curr
            values[i] = minimum
            M[curr[0], curr[1], curr[2]] = np.float32(3.4028234663852886e38)
            if (
                curr[0] == 0
                or curr[1] == 0
                or curr[2] == 0
                or curr[0] == shape[0] - 1
                or curr[1] == shape[1] - 1
                or curr[2] == shape[2] - 1
            ):
                break
        else:
            bglength = i - 1
            break
        if curr[0] == ends[0] and curr[1] == ends[1] and curr[2] == ends[2]:
            break
        i += 1
    if bglength <= 0:
        bglength = i
    for k in range(bglength):
        p = path[k]
        if bglength >= minlength:
            if values[k] < maxfill:
                M[p[0], p[1], p[2]] = values[k] + fillweight
        else:
            M[p[0], p[1], p[2]] = values[k]
    return path[:bglength]


def random_walk(M, root, ends, distsc, maxlength, fillweight, maxel, minlength):
    if M.dtype != np.float32:
        raise ValueError("Random-walk occupancy must be float32 for in-place updates")
    return _walk(
        M,
        np.asarray(root, np.int32),
        np.asarray(ends, np.int32),
        np.float32(distsc),
        int(maxlength),
        np.float32(fillweight),
        int(maxel),
        int(minlength),
    )


@njit(cache=True)
def _dijkstra(cost, shifts, root):
    count = len(cost)
    distance = np.full(count, np.float32(3.4028234663852886e38), np.float32)
    previous = np.full(count, -1, np.int64)
    tovisit = np.ones(count, np.bool_)
    distance[root] = 0.0
    queue = [(np.float32(0), root)]
    while len(queue):
        weight, current = heapq.heappop(queue)
        for j in range(6):
            nxt = current + shifts[j]
            if nxt > 0 and nxt < count and tovisit[nxt]:
                candidate = np.float32(weight + cost[nxt, j])
                if candidate < distance[nxt]:
                    heapq.heappush(queue, (candidate, nxt))
                    distance[nxt] = candidate
                    previous[nxt] = current
        tovisit[current] = False
    return distance, previous


def dijkstra_grid(cost, dims, root):
    dims = tuple(np.asarray(dims, int))
    root = np.asarray(root, int)
    idx = int(root[0] + dims[0] * root[1] + dims[0] * dims[1] * root[2])
    shift = np.array([1, -1, dims[0], -dims[0], dims[0] * dims[1], -dims[0] * dims[1]])
    distance, previous = _dijkstra(np.asarray(cost, np.float32), shift, idx)
    coords = np.full((np.prod(dims), 3), -1, np.int64)
    valid = previous >= 0
    coords[valid] = np.column_stack(np.unravel_index(previous[valid], dims, order="F"))
    return distance.reshape(dims, order="F"), coords.reshape(dims + (3,), order="F")


def extract_path(previous, node, root, return_pathM=False):
    node = np.asarray(node, int)
    root = np.asarray(root, int)
    path = [node.copy()]
    while not np.array_equal(node, root):
        if np.any(node < 0) or np.any(node >= previous.shape[:3]):
            path = []
            break
        node = previous[tuple(node)].copy()
        path.append(node)
        if len(path) > np.prod(previous.shape[:3]) + 1:
            raise ValueError("Cycle in predecessor field")
    result = np.asarray(path, dtype=int).reshape(-1, 3)
    if return_pathM:
        mask = np.zeros(previous.shape[:3], bool)
        if len(result):
            mask[tuple(result.T)] = True
        return result, mask
    return result
