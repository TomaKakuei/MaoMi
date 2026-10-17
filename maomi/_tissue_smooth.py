# Derived numerical behavior: local NAOMi checkout, Song and Charles.
import numpy as np
from scipy import interpolate as interp
from .tissue import mround


def smooth_cell_body(allpaths, cell_body, fdims):
    conn_idx_root = np.zeros((len(allpaths), 3), dtype=int)
    empty_idxs = np.zeros(len(allpaths), dtype=bool)
    cell_body_set = set(cell_body)

    for i, path in enumerate(allpaths):
        if len(path) > 0:
            path_ind = path[:, 0] + path[:, 1] * fdims[0] + path[:, 2] * fdims[0] * fdims[1]
            intersect = [idx for idx in path_ind if idx in cell_body_set]
            if intersect:
                first_intersect = np.where(path_ind == intersect[0])[0][0]
                conn_idx_root[i] = path[first_intersect]
            else:
                empty_idxs[i] = True
        else:
            empty_idxs[i] = True

    dist_mat = np.linalg.norm(conn_idx_root[:, None, :] - conn_idx_root[None, :, :], axis=-1)
    dist_mat = (dist_mat == 0).astype(float)
    dist_mat[empty_idxs, :] = np.nan

    dend_groups = []
    for i in range(dist_mat.shape[0]):
        if not np.isnan(dist_mat[i, i]):
            group = np.where(dist_mat[i, :])[0]
            dend_groups.append(group)
            dist_mat[group, :] = np.nan

    offset = 2
    conn_idx = np.zeros((len(dend_groups), 3), dtype=int)
    conn_roots = np.zeros((len(dend_groups), 3), dtype=int)

    for i, group in enumerate(dend_groups):
        path = allpaths[group[0]]
        path_ind = path[:, 0] + path[:, 1] * fdims[0] + path[:, 2] * fdims[0] * fdims[1]
        first_intersect = np.where([ind in cell_body_set for ind in path_ind])[0]
        if len(first_intersect) > 0:
            fi = first_intersect[0]
            idx_to_take = int(fi - mround(offset * np.sqrt(len(group))))
            if idx_to_take >= 0 and idx_to_take < len(path):
                conn_idx[i] = path[idx_to_take]
            else:
                conn_idx[i] = path[0]
            conn_roots[i] = path[fi]
        else:
            conn_idx[i] = path[0]
            conn_roots[i] = path[0]

    cell_mat = np.zeros(fdims, dtype=bool)
    cz = cell_body // (fdims[0] * fdims[1])
    rem = cell_body % (fdims[0] * fdims[1])
    cy = rem // fdims[0]
    cx = rem % fdims[0]
    cell_ind = np.column_stack((cx, cy, cz))

    if len(cell_ind) == 0:
        return np.array([], dtype=int)

    cell_min = np.min(cell_ind, axis=0)
    cell_max = np.max(cell_ind, axis=0)

    cell_mat[cx, cy, cz] = True
    cell_crop = cell_mat[
        cell_min[0] : cell_max[0] + 1, cell_min[1] : cell_max[1] + 1, cell_min[2] : cell_max[2] + 1
    ].copy()

    cell_diff = (
        cell_crop[:-2, 1:-1, 1:-1].astype(int)
        + cell_crop[2:, 1:-1, 1:-1].astype(int)
        + cell_crop[1:-1, :-2, 1:-1].astype(int)
        + cell_crop[1:-1, 2:, 1:-1].astype(int)
        + cell_crop[1:-1, 1:-1, :-2].astype(int)
        + cell_crop[1:-1, 1:-1, 2:].astype(int)
    )

    cell_borders = cell_crop.copy()
    cell_borders[1:-1, 1:-1, 1:-1] = (
        (cell_diff > 0) & (cell_diff < 6) & cell_borders[1:-1, 1:-1, 1:-1]
    )

    cell_borders2 = np.zeros(fdims, dtype=bool)
    cell_borders2[
        cell_min[0] : cell_max[0] + 1, cell_min[1] : cell_max[1] + 1, cell_min[2] : cell_max[2] + 1
    ] = cell_borders

    bx, by, bz = np.where(cell_borders2)
    borders_sub = np.column_stack((bx, by, bz))

    cell_processed = np.zeros(fdims, dtype=bool)
    test_dist = [0, 4, 10]
    numsamp = 20

    for j in range(len(conn_roots)):
        dist_off = min(
            max(test_dist[1], mround(offset * np.sqrt(len(dend_groups[j])))), test_dist[2]
        )
        border_dist = np.linalg.norm(conn_roots[j] - borders_sub, axis=1)
        test_idx = np.where((border_dist < dist_off) & (border_dist > test_dist[0]))[0]

        test_sub = []
        for idx in test_idx:
            pts = np.vstack((conn_roots[j], conn_idx[j], borders_sub[idx])).T
            try:
                steps = np.sqrt((np.diff(pts, axis=1) ** 2).sum(axis=0))
                steps[steps == 0] = 1e-6
                t = np.r_[0, np.cumsum(steps)]
                spline = interp.CubicSpline(
                    t,
                    pts,
                    axis=1,
                    bc_type=(
                        (1, (pts[:, 1] - pts[:, 0]) / steps[0]),
                        (1, (pts[:, 2] - pts[:, 1]) / steps[1]),
                    ),
                )
                dpts = (mround(spline(np.linspace(t[0], t[-1], numsamp)).T + 1) - 1).astype(int)
                test_sub.extend(dpts)
            except:
                pass

        if test_sub:
            test_sub = np.array(test_sub)
            test_sub = np.maximum(test_sub, 0)
            test_sub = np.minimum(test_sub, np.array(fdims) - 1)

            cell_bump = cell_borders2.copy()
            cell_bump[cx, cy, cz] = True
            cell_bump[test_sub[:, 0], test_sub[:, 1], test_sub[:, 2]] = True

            bx2, by2, bz2 = np.where(cell_bump)
            cell_ind2 = np.column_stack((bx2, by2, bz2))
            cell_min2 = np.min(cell_ind2, axis=0)
            cell_max2 = np.max(cell_ind2, axis=0)

            numdiff = np.sum(cell_bump)
            while numdiff > 0:
                cell_crop2 = cell_bump[
                    cell_min2[0] : cell_max2[0] + 1,
                    cell_min2[1] : cell_max2[1] + 1,
                    cell_min2[2] : cell_max2[2] + 1,
                ].copy()
                cell_diff2 = (
                    cell_crop2[:-2, 1:-1, 1:-1].astype(int)
                    + cell_crop2[2:, 1:-1, 1:-1].astype(int)
                    + cell_crop2[1:-1, :-2, 1:-1].astype(int)
                    + cell_crop2[1:-1, 2:, 1:-1].astype(int)
                    + cell_crop2[1:-1, 1:-1, :-2].astype(int)
                    + cell_crop2[1:-1, 1:-1, 2:].astype(int)
                )

                new_center = (cell_diff2 >= 4) | cell_crop2[1:-1, 1:-1, 1:-1]
                prev_sum = np.sum(cell_bump)
                cell_crop2[1:-1, 1:-1, 1:-1] = new_center
                cell_bump[
                    cell_min2[0] : cell_max2[0] + 1,
                    cell_min2[1] : cell_max2[1] + 1,
                    cell_min2[2] : cell_max2[2] + 1,
                ] = cell_crop2
                numdiff = np.sum(cell_crop2) - prev_sum

            cell_processed = cell_processed | cell_bump

    return np.where(cell_processed.ravel("F"))[0]
