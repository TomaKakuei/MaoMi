"""Cell-body/nucleus geometry from the NAOMi model; not full tissue growth."""

import numpy as np
from scipy.spatial import ConvexHull
from types import SimpleNamespace


def spiral_sample_sphere(N=200, vis=False):
    N = int(np.round(N))
    gr = (1 + np.sqrt(5)) / 2
    ga = 2 * np.pi * (1 - 1 / gr)

    i = np.arange(N)
    lat = np.arccos(1 - 2 * i / (N - 1))
    lon = i * ga

    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)

    V = np.column_stack((x, y, z))

    try:
        hull = ConvexHull(V)
        Tri = hull.simplices
    except Exception:
        Tri = []

    return V, Tri


def teardrop_poj(V, p=1):
    rr = np.sqrt(V[:, 0] ** 2 + V[:, 1] ** 2)
    tt = np.pi - np.arctan2(rr, np.abs(V[:, 2])) - (V[:, 2] > 0) * np.pi

    Vtear = np.zeros_like(V)
    rr_safe = np.where(rr == 0, 1.0, rr)  # prevent divide by zero

    Vtear[:, 0] = (V[:, 0] / rr_safe) * np.sin(tt) * (np.sin(0.5 * tt) ** p)
    Vtear[:, 1] = (V[:, 1] / rr_safe) * np.sin(tt) * (np.sin(0.5 * tt) ** p)
    Vtear[:, 2] = -np.cos(tt)

    Vtear[np.isnan(Vtear)] = 0
    return Vtear


def generate_neural_body(neur_params, *, random_draws=None, rng=None):
    if isinstance(neur_params, dict):
        neur_params = SimpleNamespace(**neur_params)
    rng = np.random.default_rng() if rng is None else rng
    uniform_draws = iter(random_draws["uniforms"]) if random_draws is not None else None

    def uniform(*shape):
        if uniform_draws is None:
            return rng.random(shape if shape else None)
        value = np.asarray(next(uniform_draws))
        return value.reshape(shape) if shape else float(value)

    pwr = 1
    nucoff = 3
    if hasattr(neur_params, "S_samp") and hasattr(neur_params, "Tri"):
        V = neur_params.S_samp
        Tri = neur_params.Tri
    else:
        V, Tri = spiral_sample_sphere(neur_params.n_samps, False)

    neur_type = getattr(neur_params, "neur_type", "pyr")

    if neur_type == "pyr":
        Vtear = teardrop_poj(V, 1)
    elif neur_type == "peanut":
        Vtear = teardrop_poj(V, 2)
    else:
        Vtear = V

    p_scale = getattr(neur_params, "p_scale", 1000)
    l_scale = getattr(neur_params, "l_scale", 90)

    if hasattr(neur_params, "dists") and neur_params.dists is not None:
        dists = neur_params.dists
        dists = p_scale * np.exp(-((dists / l_scale) ** pwr))
    else:
        diff = V[:, np.newaxis, :] - V[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        dists = np.clip(dists, -2.0, 2.0)
        dists = 2 * np.arcsin(dists / 2.0)
        dists = p_scale * np.exp(-((dists / l_scale) ** pwr))

    if hasattr(neur_params, "Rtear") and neur_params.Rtear is not None:
        Rtear = neur_params.Rtear
    else:
        if neur_type == "pyr":
            Rtear = np.sqrt(np.sum(Vtear**2, axis=1))
        elif neur_type == "peanut":
            Rtear = np.sqrt(np.sum(V**2, axis=1))
        else:
            Rtear = np.ones(V.shape[0])

    vals = np.linalg.eigvalsh(dists)
    min_eig = np.min(vals) * 1.03
    if min_eig < 0:
        dists = dists + np.abs(min_eig) * np.eye(dists.shape[0])

    exts = getattr(neur_params, "exts", [0.75, 1.7])
    avg_rad = neur_params.avg_rad
    x_bnds = np.array(exts) * avg_rad

    mean_vec = np.zeros_like(Rtear)
    x_base = np.abs(
        (
            rng.multivariate_normal(mean_vec, dists)
            if random_draws is None
            else np.asarray(random_draws["normal"]).ravel()
        )
    )

    x = x_base - np.mean(x_base) + avg_rad * Rtear
    xmin = np.minimum(np.min(x), x_bnds[0])
    xmax = np.maximum(np.max(x), x_bnds[1])
    if xmax != xmin:
        x = (x_bnds[1] - x_bnds[0]) * (x - xmin) / (xmax - xmin) + x_bnds[0]
    else:
        x = np.full_like(x, x_bnds[0])

    if neur_type == "pyr":
        x2 = x_base - np.mean(x_base) + avg_rad
        xmin2 = np.minimum(np.min(x2), x_bnds[0])
        xmax2 = np.maximum(np.max(x2), x_bnds[1])
        if xmax2 != xmin2:
            x2 = (x_bnds[1] - x_bnds[0]) * (x2 - xmin2) / (xmax2 - xmin2) + x_bnds[0]
        else:
            x2 = np.full_like(x2, x_bnds[0])
    else:
        x2 = x.copy()

    eccen = getattr(neur_params, "eccen", [0.35, 0.35, 0.5])
    if isinstance(eccen, (int, float)):
        eccen = np.array([eccen, eccen, eccen])
    else:
        eccen = np.array(eccen)
        if len(eccen) == 1:
            eccen = np.array([eccen[0], eccen[0], eccen[0]])

    eccens = np.ones(3) + eccen * (uniform(3) - np.array([0.5, 0.5, 0]))
    eccens = eccens / (np.prod(eccens) ** (1 / 3))

    if neur_type == "pyr":
        Vetear = Vtear * eccens
        Vetear = Vetear / np.sqrt(np.mean(np.sum(Vetear**2, axis=1)))
    else:
        Vetear = V * eccens
        Vetear = Vetear / np.sqrt(np.mean(np.sum(V**2, axis=1)))

    Vcell = Vetear * x[:, np.newaxis]
    Vcell = Vcell + np.array([0, 0, -nucoff])
    Vnorms = np.sqrt(np.sum(Vcell**2, axis=1))

    Vnuc = V * np.array([1, 1, -1]) * x2[:, np.newaxis]
    Vnorms2 = np.sqrt(np.sum(Vnuc**2, axis=1))

    nexts = getattr(neur_params, "nexts", [0.5, 1.0])
    min_thic = getattr(neur_params, "min_thic", [0.4, 0.4])
    if isinstance(min_thic, (int, float)):
        min_thic = [min_thic, min_thic]

    Vnorms2 = nexts[1] * (nexts[0] * (Vnorms2 - np.min(Vnorms2)) + (1 - nexts[0]) * np.max(Vnorms2))
    Vnorms2 = Vnorms2 + np.min(Vnorms - Vnorms2) - min_thic[0]

    Vnuc = (Vnuc * eccens) * (Vnorms2 / np.sqrt(np.sum(Vnuc**2, axis=1)))[:, np.newaxis]

    lat_ang = uniform() * 2 * np.pi
    lat_shft = (
        (1 - np.abs(uniform() - uniform()))
        * min_thic[1]
        * np.array([np.sin(lat_ang), np.cos(lat_ang)])
    )

    Vcell = Vcell + np.array([0, 0, nucoff])
    Vnuc = Vnuc + np.array([lat_shft[0], lat_shft[1], nucoff])

    if hasattr(neur_params, "nuc_rad") and neur_params.nuc_rad is not None:
        nuc_rad = neur_params.nuc_rad
        try:
            VnucSz = ConvexHull(Vnuc).volume
        except Exception:
            VnucSz = 1.0
        nucsz = (4 / 3) * np.pi * (nuc_rad[0] ** 3)
        if len(nuc_rad) > 1:
            Vnuc = Vnuc * (((nucsz / VnucSz) ** (1 / 3)) ** (1 / nuc_rad[1]))
        else:
            Vnuc = Vnuc * ((nucsz / VnucSz) ** (1 / 3))

    max_ang = getattr(neur_params, "max_ang", 20)
    a = -np.abs(max_ang) + 2 * np.abs(max_ang) * uniform(3)

    def rot_x(ang):
        ang = np.radians(ang)
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(ang):
        ang = np.radians(ang)
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(ang):
        ang = np.radians(ang)
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    Rx, Ry, Rz = rot_x(a[0]), rot_y(a[1]), rot_z(a[2])
    R = Rx @ Ry @ Rz
    Vnuc = Vnuc @ R
    Vcell = Vcell @ R

    return Vcell, Vnuc, Tri, a
