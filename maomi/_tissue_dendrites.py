# Derived numerical behavior: local NAOMi checkout, Song and Charles.
"""Main and apical dendritic growth adapted from the original Python port.
Internal indices are zero-based column-major; use tissue API boundary adapters.
"""

import numpy as np
import time
from ._tissue_random import RandomStream
from .tissue import mround
from ._tissue_paths import dijkstra_grid as dendrite_dijkstra2, extract_path as get_dendrite_path2
from ._tissue_growth import dilate_paths as dilate_dendrite_path_all
from ._tissue_smooth import smooth_cell_body


def grow_neuron_dendrites(
    vol_params,
    dend_params,
    neur_soma,
    neur_ves,
    neur_locs,
    gp_nuc,
    gp_soma=None,
    rot_ang=None,
    *,
    random_stream=None,
):
    rng = RandomStream() if random_stream is None else random_stream
    if vol_params.verbose == 1:
        print("Growing out dendrites", end="", flush=True)
    elif vol_params.verbose > 1:
        print("Growing out dendrites...")

    dtParams = np.array(dend_params.dtParams, dtype=float)
    atParams = np.array(dend_params.atParams, dtype=float)
    dweight = dend_params.dweight
    bweight = dend_params.bweight
    thicknessScale = dend_params.thicknessScale
    dims = np.array(dend_params.dims, dtype=float)
    dimsSS = np.array(dend_params.dimsSS, dtype=int)
    rallexp = dend_params.rallexp
    vres = vol_params.vres
    N_neur = int(vol_params.N_neur)
    vol_sz = np.array(vol_params.vol_sz, dtype=int)

    dims = np.minimum(dims, vol_sz / dimsSS)
    fulldims = (vol_sz * vres).astype(int)
    dims = mround(dims * vres).astype(int)
    dtParams[1:3] = dtParams[1:3] * vres
    atParams[1:4] = atParams[1:4] * vres
    thicknessScale = thicknessScale * vres * vres

    vol_depth = int(vol_params.vol_depth * vres)
    # cellVolume initialized
    ves_slice = neur_ves[:, :, vol_depth : vol_depth + int(vol_sz[2] * vres)]
    cellVolume = np.array(neur_soma, dtype=np.uint16, order="F")
    ves_val = vol_params.N_den + vol_params.N_neur + vol_params.N_bg + 1
    cellVolume[ves_slice.astype(bool)] += int(ves_val)

    for kk in range(N_neur):
        flat_nuc = gp_nuc[kk][0]  # assuming gp_nuc is a list of tuples/lists like MATLAB cell array
        # assign 1-based index equivalent
        coords = np.unravel_index(flat_nuc, cellVolume.shape, order="F")
        cellVolume[coords] = kk + 1

    if gp_soma is None:
        gp_soma = []
        for kk in range(N_neur):
            gp_soma.append([np.where(neur_soma.ravel("F") == kk + 1)[0]])

    neur_num = cellVolume.astype(np.uint16).copy()

    cellVolumeIdx = np.zeros_like(neur_soma, dtype=np.uint16, order="F")
    cellVolumeVal = np.zeros_like(neur_soma, dtype=np.float32, order="F")
    cellVolumeAD = np.zeros_like(neur_soma, dtype=np.uint8, order="F")

    allroots = np.ceil(np.maximum(vres * neur_locs, 1e-4)).astype(int) - 1  # 0-based
    fdims = dims * dimsSS
    fdims = np.minimum(fdims, fulldims).astype(int)
    ML = np.empty((fdims[0], fdims[1], fdims[2], 6), dtype=np.float32, order="F")
    M = np.empty(fdims, dtype=np.float32, order="F")

    dendVar = getattr(dend_params, "dendVar", 0.25)

    smallZ = fulldims[2] <= fdims[2]

    for j in range(N_neur):
        if vol_params.verbose > 1:
            t0 = time.time()

        aproot = np.array(
            np.unravel_index(int(np.min(gp_soma[j][0])), tuple(int(x) for x in fulldims), order="F")
        )
        numdt = max(1, int(dtParams[0] + mround(dtParams[4] * rng.randn())))
        borderflag = 0

        try:
            if smallZ:
                rootL = np.array([fdims[0] // 2, fdims[1] // 2, allroots[j, 2]])
                obstruction = cellVolume[
                    allroots[j, 0] - fdims[0] // 2 : allroots[j, 0] + fdims[0] - fdims[0] // 2,
                    allroots[j, 1] - fdims[1] // 2 : allroots[j, 1] + fdims[1] - fdims[1] // 2,
                    :,
                ]
            else:
                rootL = np.array([fdims[0] // 2, fdims[1] // 2, fdims[2] // 2])
                obstruction = cellVolume[
                    allroots[j, 0] - fdims[0] // 2 : allroots[j, 0] + fdims[0] - fdims[0] // 2,
                    allroots[j, 1] - fdims[1] // 2 : allroots[j, 1] + fdims[1] - fdims[1] // 2,
                    allroots[j, 2] - fdims[2] // 2 : allroots[j, 2] + fdims[2] - fdims[2] // 2,
                ]
            if obstruction.shape != tuple(fdims):
                raise ValueError("Shape mismatch")
        except:
            if smallZ:
                rootL = np.array([fdims[0] // 2, fdims[1] // 2, allroots[j, 2]])
                obstruction = np.zeros(fdims, dtype=np.float32, order="F")
                Xlims = [
                    allroots[j, 0] - fdims[0] // 2,
                    allroots[j, 0] + fdims[0] - fdims[0] // 2 - 1,
                ]
                Ylims = [
                    allroots[j, 1] - fdims[1] // 2,
                    allroots[j, 1] + fdims[1] - fdims[1] // 2 - 1,
                ]

                ox_start = max(0, -Xlims[0])
                ox_end = fdims[0] + min(0, fulldims[0] - 1 - Xlims[1])
                oy_start = max(0, -Ylims[0])
                oy_end = fdims[1] + min(0, fulldims[1] - 1 - Ylims[1])

                vx_start = max(0, Xlims[0])
                vx_end = min(fulldims[0], Xlims[1] + 1)
                vy_start = max(0, Ylims[0])
                vy_end = min(fulldims[1], Ylims[1] + 1)

                obstruction[ox_start:ox_end, oy_start:oy_end, :] = cellVolume[
                    vx_start:vx_end, vy_start:vy_end, :
                ]
                borderflag = 1
            else:
                rootL = np.array([fdims[0] // 2, fdims[1] // 2, fdims[2] // 2])
                obstruction = np.zeros(fdims, dtype=np.float32, order="F")
                Xlims = [
                    allroots[j, 0] - fdims[0] // 2,
                    allroots[j, 0] + fdims[0] - fdims[0] // 2 - 1,
                ]
                Ylims = [
                    allroots[j, 1] - fdims[1] // 2,
                    allroots[j, 1] + fdims[1] - fdims[1] // 2 - 1,
                ]
                Zlims = [
                    allroots[j, 2] - fdims[2] // 2,
                    allroots[j, 2] + fdims[2] - fdims[2] // 2 - 1,
                ]

                ox_start = max(0, -Xlims[0])
                ox_end = fdims[0] + min(0, fulldims[0] - 1 - Xlims[1])
                oy_start = max(0, -Ylims[0])
                oy_end = fdims[1] + min(0, fulldims[1] - 1 - Ylims[1])
                oz_start = max(0, -Zlims[0])
                oz_end = fdims[2] + min(0, fulldims[2] - 1 - Zlims[1])

                vx_start = max(0, Xlims[0])
                vx_end = min(fulldims[0], Xlims[1] + 1)
                vy_start = max(0, Ylims[0])
                vy_end = min(fulldims[1], Ylims[1] + 1)
                vz_start = max(0, Zlims[0])
                vz_end = min(fulldims[2], Zlims[1] + 1)

                obstruction[ox_start:ox_end, oy_start:oy_end, oz_start:oz_end] = cellVolume[
                    vx_start:vx_end, vy_start:vy_end, vz_start:vz_end
                ]
                borderflag = 1

        cellBody = np.where(obstruction.ravel("F") == j + 1)[0]
        # set obstruction to 0 for cellbody
        # Wait, flatten returns copy, so we must mutate original:
        flat_obs = obstruction.ravel("F")
        flat_obs[cellBody] = 0
        obstruction = flat_obs.reshape(fdims, order="F")

        root = np.ceil((rootL + 1) / dimsSS).astype(int) - 1
        root = np.minimum(root, dims - 1)

        M = 1 + dweight * rng.rand(dims[0], dims[1], dims[2], 6)

        aprootS = root + mround((aproot - allroots[j, :]) / dimsSS[2]).astype(int)
        aprootS = np.maximum(aprootS, 0)
        aprootS = np.minimum(aprootS, dims - 1)

        if aprootS[0] > root[0]:
            M[root[0] : aprootS[0] + 1, root[1], root[2], 0] = 0
        elif aprootS[0] < root[0]:
            M[aprootS[0] : root[0] + 1, root[1], root[2], 1] = 0

        if aprootS[1] > root[1]:
            M[aprootS[0], root[1] : aprootS[1] + 1, root[2], 2] = 0
        elif aprootS[1] < root[1]:
            M[aprootS[0], aprootS[1] : root[1] + 1, root[2], 3] = 0

        if aprootS[2] > root[2]:
            M[aprootS[0], aprootS[1], root[2] : aprootS[2] + 1, 4] = 0
        elif aprootS[2] < root[2]:
            M[aprootS[0], aprootS[1], aprootS[2] : root[2] + 1, 5] = 0

        M[0, :, :, 0] = np.inf
        M[-1, :, :, 1] = np.inf
        M[:, 0, :, 2] = np.inf
        M[:, -1, :, 3] = np.inf
        M[:, :, 0, 4] = np.inf
        M[:, :, -1, 5] = np.inf

        fillfrac = (
            (obstruction > 0)
            .astype(np.float32)
            .reshape(dimsSS[0], dims[0], dimsSS[1], dims[1], dimsSS[2], dims[2], order="F")
        )
        fillfrac = np.sum(fillfrac, axis=(0, 2, 4)) / np.prod(dimsSS)
        # add to M
        M = M - bweight * np.log(1 - (2 * np.maximum(0, fillfrac - 0.5)))[..., None]

        M_flat = M.reshape((np.prod(dims), 6), order="F")
        _, pathfrom = dendrite_dijkstra2(M_flat, dims, root)

        endsT = np.zeros((numdt, 3))
        for i in range(numdt):
            flag = 1
            distSC = 1.0
            numit = 0
            while flag and numit < 100:
                theta = rng.rand() * 2 * np.pi
                r = np.sqrt(rng.rand()) * dtParams[1] * distSC
                endT = np.floor(
                    [
                        r * np.cos(theta) + rootL[0],
                        r * np.sin(theta) + rootL[1],
                        2 * dtParams[2] * (rng.rand() - 0.5) + rootL[2],
                    ]
                )
                endT = np.minimum(endT, fdims - 1)
                endT = np.maximum(endT, 0)
                endT = endT.astype(int)

                if obstruction[endT[0], endT[1], endT[2]] == 0:
                    endsT[i] = endT
                    flag = 0
                distSC *= 1.01
                numit += 1

        endsTC = np.ceil((endsT + 1) / dimsSS).astype(int) - 1
        num_apical = int(atParams[0])
        endsA = np.zeros((num_apical, 3))
        rootA = np.floor(
            [
                rootL[0] + 2 * atParams[3] * (rng.rand() - 0.5),
                rootL[1] + 2 * atParams[3] * (rng.rand() - 0.5),
                atParams[2],
            ]
        )

        if rot_ang is not None and rot_ang.shape[0] > j:
            rootA2 = rootA + (rootL[2] + 1) * np.sin(np.deg2rad([rot_ang[j, 1], -rot_ang[j, 0], 0]))
        else:
            rootA2 = rootA

        for i in range(num_apical):
            flag = 1
            distSC = 1.0
            numit = 0
            while flag and numit < 100:
                theta = rng.rand() * 2 * np.pi
                r = np.sqrt(rng.rand()) * atParams[1] * distSC
                endA = np.floor(
                    [
                        r * np.cos(theta) + rootA2[0],
                        r * np.sin(theta) + rootA2[1],
                        2 * atParams[2] * (rng.rand() - 0.5) + rootA2[2],
                    ]
                )
                endA = np.minimum(endA, fdims - 1)
                endA = np.maximum(endA, 0)
                endA = endA.astype(int)
                if obstruction[endA[0], endA[1], endA[2]] == 0:
                    endsA[i] = endA
                    flag = 0
                distSC *= 1.01
                numit += 1

        endsAC = np.ceil((endsA + 1) / dimsSS).astype(int) - 1
        ends = np.vstack((endsTC, endsAC))
        endsL = np.vstack((endsT, endsA))
        nends = numdt + num_apical

        paths = np.zeros(dims, dtype=bool, order="F")
        for i in range(nends):
            path = get_dendrite_path2(pathfrom, ends[i], root)
            if len(path) > 0:
                paths[path[:, 0], path[:, 1], path[:, 2]] = True

        rootL_new = mround((root + 1 - 0.5) * dimsSS).astype(int) - 1
        denLocs = np.where(paths.ravel("F"))[0]

        ML.fill(np.inf)

        for i in range(len(denLocs)):
            lz = denLocs[i] // (dims[0] * dims[1])
            rem = denLocs[i] % (dims[0] * dims[1])
            ly = rem // dims[0]
            lx = rem % dims[0]

            temp = 1 + dweight * rng.rand(dimsSS[0], dimsSS[1], dimsSS[2], 6)
            filled = (
                obstruction[
                    lx * dimsSS[0] : (lx + 1) * dimsSS[0],
                    ly * dimsSS[1] : (ly + 1) * dimsSS[1],
                    lz * dimsSS[2] : (lz + 1) * dimsSS[2],
                ]
                * np.inf
            )
            filled[np.isnan(filled)] = 0
            temp += filled[..., None]
            ML[
                lx * dimsSS[0] : (lx + 1) * dimsSS[0],
                ly * dimsSS[1] : (ly + 1) * dimsSS[1],
                lz * dimsSS[2] : (lz + 1) * dimsSS[2],
                :,
            ] = temp

        aprootL = rootL_new + aproot - allroots[j, :]
        aprootL = np.maximum(aprootL, 0)
        aprootL = np.minimum(aprootL, fdims - 1)

        if aprootL[0] > rootL_new[0]:
            ML[rootL_new[0] : aprootL[0] + 1, rootL_new[1], rootL_new[2], 0] = 0
        elif aprootL[0] < rootL_new[0]:
            ML[aprootL[0] : rootL_new[0] + 1, rootL_new[1], rootL_new[2], 1] = 0

        if aprootL[1] > rootL_new[1]:
            ML[aprootL[0], rootL_new[1] : aprootL[1] + 1, rootL_new[2], 2] = 0
        elif aprootL[1] < rootL_new[1]:
            ML[aprootL[0], aprootL[1] : rootL_new[1] + 1, rootL_new[2], 3] = 0

        if aprootL[2] > rootL_new[2]:
            ML[aprootL[0], aprootL[1], rootL_new[2] : aprootL[2] + 1, 4] = 0
        elif aprootL[2] < rootL_new[2]:
            ML[aprootL[0], aprootL[1], aprootL[2] : rootL_new[2] + 1, 5] = 0

        ML[0, :, :, 0] = np.inf
        ML[-1, :, :, 1] = np.inf
        ML[:, 0, :, 2] = np.inf
        ML[:, -1, :, 3] = np.inf
        ML[:, :, 0, 4] = np.inf
        ML[:, :, -1, 5] = np.inf

        ML_flat = ML.reshape((np.prod(fdims), 6), order="F")
        _, pathfromL = dendrite_dijkstra2(ML_flat, fdims, rootL_new)

        for i in range(len(denLocs)):
            lz = denLocs[i] // (dims[0] * dims[1])
            rem = denLocs[i] % (dims[0] * dims[1])
            ly = rem // dims[0]
            lx = rem % dims[0]
            ML[
                lx * dimsSS[0] : (lx + 1) * dimsSS[0],
                ly * dimsSS[1] : (ly + 1) * dimsSS[1],
                lz * dimsSS[2] : (lz + 1) * dimsSS[2],
                :,
            ] = np.inf

        finepathsIdx = np.zeros(fdims, dtype=np.float32, order="F")
        fineIdxs = []
        allpaths = []
        for i in range(numdt):
            path = get_dendrite_path2(pathfromL, endsL[i].astype(int), rootL_new)
            if len(path) == 1:
                path = np.vstack((path[0], path[0], path))
            elif len(path) == 2:
                path = np.vstack((path[0], path))
            allpaths.append(path)
            if len(path) > 0:
                dendSz = max(0, rng.normal(1, dendVar)) ** 2
                diff1 = np.abs(np.diff(path, axis=0))
                diff2 = np.abs(np.diff(diff1, axis=0))
                sum_diff = np.sum(diff2, axis=1) / 2
                pathW_mid = 1 - (1 - 1 / np.sqrt(2)) * sum_diff
                pathW = dendSz * np.concatenate(([1], pathW_mid, [1])).astype(np.float32)

                flat_path_idx = (
                    path[:, 0] + path[:, 1] * fdims[0] + path[:, 2] * fdims[0] * fdims[1]
                )
                flat = finepathsIdx.ravel("F")
                flat[flat_path_idx] = flat[flat_path_idx] + pathW
                fineIdxs.extend(flat_path_idx)

        fineIdxs = np.unique(fineIdxs)
        flat_fidx = finepathsIdx.ravel("F")
        flat_fidx[fineIdxs] = thicknessScale * dtParams[3] * (flat_fidx[fineIdxs] ** (1 / rallexp))
        finepathsIdx = flat_fidx.reshape(fdims, order="F")

        finepathsVal = np.zeros(fdims, dtype=np.float32, order="F")
        fineIdxs2 = []
        for i in range(num_apical):
            path = get_dendrite_path2(pathfromL, endsL[i + numdt].astype(int), rootL_new)
            allpaths.append(path)
            if len(path) > 0:
                if len(path) > 2:
                    dendSz = max(0, rng.normal(1, dendVar))
                    diff1 = np.abs(np.diff(path, axis=0))
                    diff2 = np.abs(np.diff(diff1, axis=0))
                    sum_diff = np.sum(diff2, axis=1) / 2
                    pathW_mid = 1 - (1 - 1 / np.sqrt(2)) * sum_diff
                    pathW = dendSz * np.concatenate(([1], pathW_mid, [1])).astype(np.float32)
                else:
                    pathW = np.ones(len(path), dtype=np.float32)

                flat_path_idx = (
                    path[:, 0] + path[:, 1] * fdims[0] + path[:, 2] * fdims[0] * fdims[1]
                )
                flat = finepathsVal.ravel("F")
                flat[flat_path_idx] = flat[flat_path_idx] + pathW
                fineIdxs2.extend(flat_path_idx)

        fineIdxs2 = np.unique(fineIdxs2)
        finepathsAD = np.zeros(fdims, dtype=bool, order="F")
        flat_fval = finepathsVal.ravel("F")
        flat_fAD = finepathsAD.ravel("F")

        flat_fAD[fineIdxs2] = True
        flat_fval[fineIdxs2] = (
            thicknessScale * atParams[4] * (flat_fval[fineIdxs2] ** (1 / rallexp))
        )

        flat_fval[fineIdxs] += finepathsIdx.ravel("F")[fineIdxs]
        finepathsIdx = np.zeros(fdims, dtype=np.float32, order="F")
        flat_fidx = finepathsIdx.ravel("F")

        rootL_idx = rootL_new[0] + rootL_new[1] * fdims[0] + rootL_new[2] * fdims[0] * fdims[1]
        cellBody = np.concatenate((cellBody, [rootL_idx]))

        if all(len(p) == 0 for p in allpaths):
            cellBodySmoothed = cellBody
        else:
            cellBodySmoothed = smooth_cell_body(allpaths, cellBody, fdims)
            cellBodySmoothed = np.setxor1d(cellBodySmoothed, cellBody)

        fineIdxs3 = np.unique(np.concatenate((fineIdxs, fineIdxs2, cellBodySmoothed)))

        flat_fidx[fineIdxs3] = j + 1
        flat_fidx[cellBodySmoothed] = j + 1
        flat_fval[cellBodySmoothed] += 1

        flat_fidx[cellBody] = 0
        flat_fval[cellBody] = 0
        flat_fAD[cellBody] = False

        finepathsIdx = flat_fidx.reshape(fdims, order="F")
        finepathsVal = flat_fval.reshape(fdims, order="F")
        finepathsAD = flat_fAD.reshape(fdims, order="F")

        if not borderflag:
            xi = fineIdxs3 % fdims[0]
            yi = (fineIdxs3 // fdims[0]) % fdims[1]
            zi = fineIdxs3 // (fdims[0] * fdims[1])

            xi = xi + allroots[j, 0] - fdims[0] // 2
            yi = yi + allroots[j, 1] - fdims[1] // 2
            if smallZ:
                Didxs = xi + yi * fulldims[0] + zi * fulldims[0] * fulldims[1]
            else:
                zi = zi + allroots[j, 2] - fdims[2] // 2
                Didxs = xi + yi * fulldims[0] + zi * fulldims[0] * fulldims[1]
            Fidxs = fineIdxs3
        else:
            xi = fineIdxs3 % fdims[0]
            yi = (fineIdxs3 // fdims[0]) % fdims[1]
            zi = fineIdxs3 // (fdims[0] * fdims[1])

            xi = xi + max(0, Xlims[0]) - max(0, -Xlims[0])
            yi = yi + max(0, Ylims[0]) - max(0, -Ylims[0])
            if smallZ:
                gidxs = (
                    (xi < fulldims[0])
                    & (yi < fulldims[1])
                    & (zi < fulldims[2])
                    & (xi >= 0)
                    & (yi >= 0)
                    & (zi >= 0)
                )
            else:
                zi = zi + max(0, Zlims[0]) - max(0, -Zlims[0])
                gidxs = (
                    (xi < fulldims[0])
                    & (yi < fulldims[1])
                    & (zi < fulldims[2])
                    & (xi >= 0)
                    & (yi >= 0)
                    & (zi >= 0)
                )

            xi = xi[gidxs]
            yi = yi[gidxs]
            zi = zi[gidxs]
            Didxs = xi + yi * fulldims[0] + zi * fulldims[0] * fulldims[1]
            Fidxs = fineIdxs3[gidxs]

        cv_flat = cellVolume.ravel("F")
        cvi_flat = cellVolumeIdx.ravel("F")
        cvv_flat = cellVolumeVal.ravel("F")
        cvad_flat = cellVolumeAD.ravel("F")

        cv_flat[Didxs] += flat_fidx[Fidxs].astype(np.uint16)
        cvi_flat[Didxs] += flat_fidx[Fidxs].astype(np.uint16)
        cvv_flat[Didxs] += flat_fval[Fidxs].astype(np.float32)
        cvad_flat[Didxs] = cvad_flat[Didxs] | flat_fAD[Fidxs]

        if gp_soma is not None:
            # Ensure gp_soma[j] is a Python list (not numpy array)
            if not isinstance(gp_soma[j], list):
                gp_soma[j] = list(gp_soma[j])
            if len(gp_soma[j]) > 1:
                gp_soma[j][1] = cellBodySmoothed
            else:
                gp_soma[j].append(cellBodySmoothed)

        if vol_params.verbose == 1:
            print(".", end="", flush=True)
        elif vol_params.verbose > 1:
            print(f"{j} ({time.time() - t0:.6f} seconds).")

    cvv_flat = cellVolumeVal.ravel("F")
    cvv_flat_int = np.empty_like(cvv_flat, dtype=np.uint16)
    chunk_sz = 10000000
    for i in range(0, cvv_flat.size, chunk_sz):
        end = min(i + chunk_sz, cvv_flat.size)
        c = cvv_flat[i:end]
        cvv_flat_int[i:end] = np.floor(c) + ((c % 1) > rng.rand(c.size))
    cellVolumeVal = cvv_flat_int.reshape(cellVolumeVal.shape, order="F")
    cellVolumeIdx = cellVolumeIdx.astype(np.uint16)
    cellVolumeAD = cellVolumeAD.astype(np.uint16)
    cellVolumeBD = (~cellVolumeAD.astype(bool)).astype(np.uint16)

    _, dendnumAD = dilate_dendrite_path_all(
        cellVolumeVal * cellVolumeAD, cellVolumeIdx * cellVolumeAD, neur_num, random_stream=rng
    )
    _, dendnumBD = dilate_dendrite_path_all(
        cellVolumeVal * cellVolumeBD, cellVolumeIdx * cellVolumeBD, neur_num, random_stream=rng
    )

    dendnumAD_flat = dendnumAD.ravel("F")
    dendnumBD_flat = dendnumBD.ravel("F")

    for kk in range(N_neur):
        dendnumAD_flat[gp_nuc[kk][0]] = 0
        dendnumBD_flat[gp_nuc[kk][0]] = 0
        dendnumAD_flat[gp_soma[kk][0]] = 0
        dendnumBD_flat[gp_soma[kk][0]] = 0

    mask = dendnumAD_flat > 0
    dendnumBD_flat[mask] = dendnumAD_flat[mask]

    neur_num_flat = neur_num.ravel("F")
    mask2 = dendnumBD_flat > 0
    neur_num_flat[mask2] = dendnumBD_flat[mask2]

    for kk in range(N_neur):
        neur_num_flat[gp_nuc[kk][0]] = 0
        neur_num_flat[gp_soma[kk][0]] = kk + 1

    neur_num = neur_num_flat.reshape(neur_num.shape, order="F")
    dendnumAD = dendnumAD_flat.reshape(dendnumAD.shape, order="F")

    if vol_params.verbose >= 1:
        print("done.")

    return neur_num, dendnumAD, dend_params, gp_soma


def grow_apical_dendrites(
    vol_params, dend_params, neur_num, cellVolumeAD, gp_nuc, gp_soma, *, random_stream=None
):
    rng = RandomStream() if random_stream is None else random_stream
    if vol_params.verbose == 1:
        print("Growing out apical dendrites", end="", flush=True)
    elif vol_params.verbose > 1:
        print("Growing out apical dendrites...")

    atParams = np.array(dend_params.atParams2, dtype=float)
    dweight = dend_params.dweight
    bweight = dend_params.bweight
    thicknessScale = dend_params.thicknessScale
    dims = np.array(dend_params.dims, dtype=float)
    dimsSS = np.array(dend_params.dimsSS, dtype=int)
    rallexp = dend_params.rallexp
    vres = vol_params.vres
    N_neur = int(vol_params.N_neur)
    N_den = int(vol_params.N_den)
    vol_sz = np.array(vol_params.vol_sz, dtype=int)

    dims = np.minimum(dims, vol_sz / dimsSS)
    fulldims = (vol_sz * vres).astype(int)
    dims = mround(dims * vres).astype(int)
    atParams[1:4] = atParams[1:4] * vres
    thicknessScale = thicknessScale * vres * vres
    fdims = dims * dimsSS
    fdims = np.minimum(fdims, fulldims).astype(int)

    cellVolume = np.array(neur_num, dtype=np.float32, order="F", copy=True)
    cv_flat = cellVolume.ravel("F")
    for kk in range(N_neur):
        cv_flat[gp_nuc[kk][0]] = kk + 1

    root_den = np.zeros((N_den, 3), dtype=int)

    for j in range(N_den):
        while True:
            root = np.ceil(fulldims[:2] * rng.rand(2)).astype(int) - 1
            root = np.maximum(root, 0)
            if cellVolume[root[0], root[1], 0] == 0:
                root_den[j] = [root[0], root[1], fulldims[2] - 1]
                break

    cellVolumeIdx = np.zeros_like(neur_num, dtype=np.uint16, order="F")
    cellVolumeVal = np.zeros_like(neur_num, dtype=np.float32, order="F")
    ML = np.empty((fdims[0], fdims[1], fdims[2], 6), dtype=np.float32, order="F")
    M = np.empty(fdims, dtype=np.float32, order="F")

    dendVar = getattr(dend_params, "apicalVar", getattr(dend_params, "dendVar", 0.35))

    for j in range(N_den):
        if vol_params.verbose > 1:
            t0 = time.time()

        rootL = np.array([fdims[0] // 2, fdims[1] // 2, root_den[j, 2]])
        borderflag = 0
        try:
            obstruction = cellVolume[
                root_den[j, 0] - fdims[0] // 2 : root_den[j, 0] + fdims[0] - fdims[0] // 2,
                root_den[j, 1] - fdims[1] // 2 : root_den[j, 1] + fdims[1] - fdims[1] // 2,
                :,
            ]
            if obstruction.shape != tuple(fdims):
                raise ValueError("Shape mismatch")
        except:
            obstruction = np.zeros(fdims, dtype=np.float32, order="F")
            Xlims = [root_den[j, 0] - fdims[0] // 2, root_den[j, 0] + fdims[0] - fdims[0] // 2 - 1]
            Ylims = [root_den[j, 1] - fdims[1] // 2, root_den[j, 1] + fdims[1] - fdims[1] // 2 - 1]

            ox_start = max(0, -Xlims[0])
            ox_end = fdims[0] + min(0, fulldims[0] - 1 - Xlims[1])
            oy_start = max(0, -Ylims[0])
            oy_end = fdims[1] + min(0, fulldims[1] - 1 - Ylims[1])

            vx_start = max(0, Xlims[0])
            vx_end = min(fulldims[0], Xlims[1] + 1)
            vy_start = max(0, Ylims[0])
            vy_end = min(fulldims[1], Ylims[1] + 1)

            obstruction[ox_start:ox_end, oy_start:oy_end, :] = cellVolume[
                vx_start:vx_end, vy_start:vy_end, :
            ]
            borderflag = 1

        cellBody = obstruction == j + 1 + N_neur
        obstruction[cellBody] = 0

        root = np.ceil((rootL + 1) / dimsSS).astype(int) - 1
        root = np.minimum(root, dims - 1)
        M = 1 + dweight * rng.rand(dims[0], dims[1], dims[2], 6)

        M[0, :, :, 0] = np.inf
        M[-1, :, :, 1] = np.inf
        M[:, 0, :, 2] = np.inf
        M[:, -1, :, 3] = np.inf
        M[:, :, 0, 4] = np.inf
        M[:, :, -1, 5] = np.inf

        fillfrac = (
            (obstruction > 0)
            .astype(np.float32)
            .reshape(dimsSS[0], dims[0], dimsSS[1], dims[1], dimsSS[2], dims[2], order="F")
        )
        fillfrac = np.sum(fillfrac, axis=(0, 2, 4)) / np.prod(dimsSS)
        M = M - bweight * np.log(1 - (2 * np.maximum(0, fillfrac - 0.5)))[..., None]

        M_flat = M.reshape((np.prod(dims), 6), order="F")
        _, pathfrom = dendrite_dijkstra2(M_flat, dims, root)

        num_apical = int(atParams[0])
        endsA = np.zeros((num_apical, 3))
        rootA = np.floor(
            [
                rootL[0] + 2 * atParams[3] * (rng.rand() - 0.5),
                rootL[1] + 2 * atParams[3] * (rng.rand() - 0.5),
                fdims[2] - 1,
            ]
        )

        for i in range(num_apical):
            flag = 1
            distSC = 1.0
            numit = 0
            while flag and numit < 100:
                theta = rng.rand() * 2 * np.pi
                r = np.sqrt(rng.rand()) * atParams[1] * distSC
                endA = np.floor([r * np.cos(theta) + rootA[0], r * np.sin(theta) + rootA[1], 0])
                endA = np.minimum(endA, fdims - 1)
                endA = np.maximum(endA, 0)
                endA = endA.astype(int)
                if obstruction[endA[0], endA[1], endA[2]] == 0:
                    endsA[i] = endA
                    flag = 0
                distSC *= 1.01
                numit += 1

        endsAC = np.ceil((endsA + 1) / dimsSS).astype(int) - 1
        paths = np.zeros(dims, dtype=int, order="F")
        for i in range(num_apical):
            _, pathM = get_dendrite_path2(pathfrom, endsAC[i], root, return_pathM=True)
            paths += pathM.astype(int)

        ML.fill(np.inf)
        rootL_new = mround((root + 1 - 0.5) * dimsSS).astype(int) - 1
        denLocs = np.where(paths.ravel("F"))[0]
        for i in range(len(denLocs)):
            lz = denLocs[i] // (dims[0] * dims[1])
            rem = denLocs[i] % (dims[0] * dims[1])
            ly = rem // dims[0]
            lx = rem % dims[0]

            temp = 1 + dweight * rng.rand(dimsSS[0], dimsSS[1], dimsSS[2], 6)
            ML[
                lx * dimsSS[0] : (lx + 1) * dimsSS[0],
                ly * dimsSS[1] : (ly + 1) * dimsSS[1],
                lz * dimsSS[2] : (lz + 1) * dimsSS[2],
                :,
            ] = temp

        ML[0, :, :, 0] = np.inf
        ML[-1, :, :, 1] = np.inf
        ML[:, 0, :, 2] = np.inf
        ML[:, -1, :, 3] = np.inf
        ML[:, :, 0, 4] = np.inf
        ML[:, :, -1, 5] = np.inf

        filled = obstruction * np.inf
        filled[np.isnan(filled)] = 0
        ML += filled[..., None]

        ML_flat = ML.reshape((np.prod(fdims), 6), order="F")
        _, pathfromL = dendrite_dijkstra2(ML_flat, fdims, rootL_new)
        # set up distance penalty matrix
        M.fill(1.0)

        finepathsVal = np.zeros(fdims, dtype=np.float32, order="F")
        for i in range(num_apical):
            path = get_dendrite_path2(pathfromL, endsA[i].astype(int), rootL_new)
            if len(path) > 0:
                dendSz = max(0, rng.normal(1, dendVar))
                if len(path) > 2:
                    diff1 = np.abs(np.diff(path, axis=0))
                    diff2 = np.abs(np.diff(diff1, axis=0))
                    sum_diff = np.sum(diff2, axis=1) / 2
                    pathW_mid = 1 - (1 - 1 / np.sqrt(2)) * sum_diff
                    pathW = dendSz * np.concatenate(([1], pathW_mid, [1])).astype(np.float32)
                else:
                    pathW = dendSz * np.ones(len(path), dtype=np.float32)

                flat_path_idx = (
                    path[:, 0] + path[:, 1] * fdims[0] + path[:, 2] * fdims[0] * fdims[1]
                )
                flat = finepathsVal.ravel("F")
                flat[flat_path_idx] = flat[flat_path_idx] + pathW

        mask = finepathsVal > 0
        finepathsVal[mask] = thicknessScale * atParams[4] * (finepathsVal[mask] ** (1 / rallexp))
        finepathsVal[cellBody] = 0
        finepathsIdx = (j + 1 + N_neur) * (finepathsVal > 0).astype(np.float32)

        if not borderflag:
            x_slice = slice(
                root_den[j, 0] - fdims[0] // 2, root_den[j, 0] + fdims[0] - fdims[0] // 2
            )
            y_slice = slice(
                root_den[j, 1] - fdims[1] // 2, root_den[j, 1] + fdims[1] - fdims[1] // 2
            )

            cellVolume[x_slice, y_slice, :] += finepathsIdx
            cellVolumeVal[x_slice, y_slice, :] += finepathsVal.astype(np.float32)
            cellVolumeIdx[x_slice, y_slice, :] += finepathsIdx.astype(np.uint16)
        else:
            vx_start = max(0, Xlims[0])
            vx_end = min(fulldims[0], Xlims[1] + 1)
            vy_start = max(0, Ylims[0])
            vy_end = min(fulldims[1], Ylims[1] + 1)

            ox_start = max(0, -Xlims[0])
            ox_end = fdims[0] + min(0, fulldims[0] - 1 - Xlims[1])
            oy_start = max(0, -Ylims[0])
            oy_end = fdims[1] + min(0, fulldims[1] - 1 - Ylims[1])

            cellVolume[vx_start:vx_end, vy_start:vy_end, :] += finepathsIdx[
                ox_start:ox_end, oy_start:oy_end, :
            ]
            cellVolumeVal[vx_start:vx_end, vy_start:vy_end, :] += finepathsVal[
                ox_start:ox_end, oy_start:oy_end, :
            ].astype(np.float32)
            cellVolumeIdx[vx_start:vx_end, vy_start:vy_end, :] += finepathsIdx[
                ox_start:ox_end, oy_start:oy_end, :
            ].astype(np.uint16)

        if vol_params.verbose == 1:
            print(".", end="", flush=True)
        elif vol_params.verbose > 1:
            print(f"{j} ({time.time() - t0:.6f} seconds).")

    cellVolumeVal = np.ceil(cellVolumeVal).astype(np.float32)
    cellVolumeIdx = cellVolumeIdx.astype(np.float32)
    _, dendnum = dilate_dendrite_path_all(cellVolumeVal, cellVolumeIdx, neur_num, random_stream=rng)

    cellVolumeAD = cellVolumeAD.astype(np.uint16) + dendnum.astype(np.uint16)
    neur_num = neur_num.astype(np.uint16) + dendnum.astype(np.uint16)

    nn_flat = neur_num.ravel("F")
    for kk in range(N_neur):
        nn_flat[gp_nuc[kk][0]] = 0
    for kk in range(N_neur):
        nn_flat[gp_soma[kk][0]] = kk + 1
    neur_num = nn_flat.reshape(neur_num.shape, order="F")

    neur_num_AD = cellVolumeAD.copy()
    nad_flat = neur_num_AD.ravel("F")
    for kk in range(N_neur):
        nad_flat[gp_nuc[kk][0]] = 0
        nad_flat[gp_soma[kk][0]] = 0
    neur_num_AD = nad_flat.reshape(neur_num_AD.shape, order="F")

    chunk_sz = 10000000
    for i in range(0, nad_flat.size, chunk_sz):
        end = min(i + chunk_sz, nad_flat.size)
        mask = (nad_flat[i:end].astype(int) - nn_flat[i:end].astype(int)) > 0
        nad_flat[i:end][mask] = 0

    if vol_params.verbose >= 1:
        print("done.")

    return neur_num, neur_num_AD, dend_params
