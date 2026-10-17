"""Closed-surface membership from oriented triangle solid angles.

Uses a direct scalar-triple-product / atan2 formula and an edge-orientation
walk. No external ray-casting implementation is used. Boundary points are
classified outside; meshes must be closed, connected and orientable.
"""

import math
import numpy as np
from numba import njit


def orient_faces(faces, vertex_count):
    tri = np.asarray(faces, np.int64)
    if (
        not len(tri)
        or tri.ndim != 2
        or tri.shape[1] != 3
        or np.any(tri < 0)
        or np.any(tri >= vertex_count)
    ):
        raise ValueError("Invalid triangle indices")
    edges = {}
    adj = [[] for _ in tri]
    for j, row in enumerate(tri):
        if len(set(map(int, row))) != 3:
            raise ValueError("Degenerate triangle indices")
        for u, v in zip(row, np.roll(row, -1)):
            key = tuple(sorted((int(u), int(v))))
            edges.setdefault(key, []).append((j, 1 if u < v else -1))
    for pair in edges.values():
        if len(pair) != 2:
            raise ValueError("Surface must have exactly two faces per edge")
        (a, da), (b, db) = pair
        relation = -da * db
        adj[a].append((b, relation))
        adj[b].append((a, relation))
    sign = np.zeros(len(tri), np.int8)
    sign[0] = 1
    stack = [0]
    while stack:
        j = stack.pop()
        for k, r in adj[j]:
            expected = sign[j] * r
            if sign[k] == 0:
                sign[k] = expected
                stack.append(k)
            elif sign[k] != expected:
                raise ValueError("Surface is not orientable")
    if np.any(sign == 0):
        raise ValueError("Surface must be connected")
    out = tri.copy()
    flip = sign < 0
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


@njit(cache=True)
def _solid_angle_membership(vertices, faces, points):
    result = np.zeros(len(points), np.int8)
    for q in range(len(points)):
        total = 0.0
        boundary = False
        for f in faces:
            a = vertices[f[0]] - points[q]
            b = vertices[f[1]] - points[q]
            c = vertices[f[2]] - points[q]
            la = np.sqrt(np.dot(a, a))
            lb = np.sqrt(np.dot(b, b))
            lc = np.sqrt(np.dot(c, c))
            if min(la, lb, lc) < 1e-13:
                boundary = True
                break
            cross = np.array(
                [b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]]
            )
            numerator = np.dot(a, cross)
            denominator = la * lb * lc + np.dot(a, b) * lc + np.dot(b, c) * la + np.dot(c, a) * lb
            if abs(numerator) < 1e-13 * la * lb * lc and denominator <= 0:
                boundary = True
                break
            total += 2 * math.atan2(numerator, denominator)
        if not boundary and abs(total) > 2 * math.pi:
            result[q] = 1
    return result


def inside_mesh(vertices, triangles, points):
    vertices = np.asarray(vertices, np.float64)
    points = np.asarray(points, np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(vertices).all()
        or not np.isfinite(points).all()
    ):
        raise ValueError("Finite XYZ vertices and points required")
    faces = orient_faces(triangles, len(vertices))
    return _solid_angle_membership(vertices, faces, points)
