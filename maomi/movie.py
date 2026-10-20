"""Dynamic planar scanning, optical background, row motion and detector noise.

Arrays use X,Y,Z; movies use X,Y,T. Spatial convolution follows MATLAB's
same-crop convention for even kernels. Motion offsets use the explicit upstream
one-based coordinates; z_start is a Python zero-based slice index.
"""

import time
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2


def conv2_same(image, kernel):
    image = np.asarray(image)
    kernel = np.asarray(kernel)
    full = convolve2d(image, kernel, mode="full")
    i, j = np.asarray(kernel.shape) // 2
    return full[i : i + image.shape[0], j : j + image.shape[1]]


def _smooth_length(n):
    while True:
        m = n
        for prime in (2, 3, 5, 7):
            while m % prime == 0:
                m //= prime
        if m == 1:
            return n
        n += 1


def _group_z(a, count):
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("scan_avg must be a positive integer")
    return np.add.reduceat(a, np.arange(0, a.shape[2], count), axis=2) if count > 1 else a


def psf_fft(volume_shape, psf, scan_avg=1):
    psf = np.asarray(psf)
    if psf.ndim == 2:
        psf = psf[..., None]
    grouped = _group_z(psf, scan_avg)
    shape = tuple(_smooth_length(int(volume_shape[j] + psf.shape[j] - 1)) for j in (0, 1))
    return fft2(grouped, s=shape, axes=(0, 1))


def single_scan(volume, psf, *, scan_avg=1, frequency_psf=None, rana=None, z_start=0):
    """One XY frame: sum axial slice convolutions, with optional grouped Z."""
    if rana is not None and rana is not False:
        return rana.scan_slab(volume,scan_avg=scan_avg,z_start=z_start)
    volume = np.asarray(volume)
    psf = np.asarray(psf)
    if volume.ndim != 3 or psf.ndim != 3 or volume.shape[2] != psf.shape[2]:
        raise ValueError("Volume slab and PSF must have the same axial length")
    grouped = _group_z(volume, scan_avg)
    freq = (
        psf_fft(volume.shape, psf, scan_avg) if frequency_psf is None else np.asarray(frequency_psf)
    )
    if freq.shape[2] != grouped.shape[2]:
        raise ValueError("Wrong grouped PSF depth")
    image = ifft2(
        (fft2(grouped, s=freq.shape[:2], axes=(0, 1)) * freq).sum(axis=2), axes=(0, 1)
    ).real
    i, j = np.asarray(psf.shape[:2]) // 2
    return image[i : i + volume.shape[0], j : j + volume.shape[1]]


def blurred_background(
    volume,
    indices,
    kernel,
    *,
    power_ratio=1.0,
    mask=None,
    zscale=None,
    extra_volumes=(),
    frequency=False
):
    """Port of blurredBackComp2; indices are zero-based."""
    volume = np.asarray(volume)
    indices = np.asarray(indices, int).ravel()
    if indices.size == 0:
        raise ValueError("Background requires at least one axial slice")
    if np.any(indices < 0) or np.any(indices >= volume.shape[2]):
        raise ValueError("Invalid background indices")
    scale = (
        np.asarray(zscale).ravel()[: len(indices)]
        if zscale is not None and np.size(zscale) >= len(indices)
        else np.ones(len(indices))
    )
    image = (volume[:, :, indices] * scale).sum(axis=2) / len(indices)
    for other in extra_volumes:
        image += (np.asarray(other)[:, :, indices] * scale).sum(axis=2) / len(indices)
    image = image * power_ratio
    if mask is not None:
        image = image * np.asarray(mask)
    kernel = np.asarray(kernel)
    if kernel.ndim == 3:
        kernel = kernel[:, :, indices].mean(axis=2) if kernel.shape[2] > 1 else kernel[:, :, 0]
    if not frequency:
        return conv2_same(image, kernel)
    transformed = ifft2(fft2(image, s=kernel.shape) * kernel).real
    start = np.ceil((np.asarray(kernel.shape) - np.asarray(volume.shape[:2]) - 1) / 2).astype(int)
    return transformed[start[0] : start[0] + volume.shape[0], start[1] : start[1] + volume.shape[1]]


def row_motion(image, buffer, x_offsets, y_offsets):
    """Match imgSubRowShift, including NaNs beyond its interpolation domain."""
    image = np.asarray(image)
    n, m = image.shape
    if not isinstance(buffer, (int, np.integer)) or buffer < 0 or 2 * buffer >= min(n, m):
        raise ValueError("Invalid motion buffer")
    x = np.broadcast_to(np.asarray(x_offsets).reshape(-1), (n,)) - buffer + np.arange(n) - 1
    y = np.broadcast_to(np.asarray(y_offsets).reshape(-1), (n,)) - buffer
    low = np.floor(x).astype(int)
    high = np.ceil(x).astype(int)
    weight = x - low
    a = np.full((n, m), np.nan, dtype=np.float32)
    b = a.copy()
    valid = (low >= 0) & (low < n)
    a[valid] = image[low[valid]]
    valid = (high >= 0) & (high < n)
    b[valid] = image[high[valid]]
    temp = (a * (1 - weight[:, None]) + b * weight[:, None]).astype(np.float32)
    pad = int(np.ceil(np.max(np.abs(y))))
    temp = np.pad(temp, ((0, 0), (pad, pad)), constant_values=np.nan)
    low = np.floor(y).astype(int) + pad
    high = np.ceil(y).astype(int) + pad
    weight = y - np.floor(y)
    a = temp[np.arange(n)[:, None], low[:, None] + np.arange(m)]
    b = temp[np.arange(n)[:, None], high[:, None] + np.arange(m)]
    result = (a * (1 - weight[:, None]) + b * weight[:, None]).astype(np.float32)
    return result[buffer : n - buffer, buffer : m - buffer]


def detector_noise(clean, parameters=None, *, draws=None, rng=None):
    """Poisson counts, lognormal gain and rounded electronics noise.

    A draw replay contains counts, gain_normal and electronics_normal. Matching
    a replay certifies arithmetic conditional on those draws, not RNG identity.
    """
    p = {"mu": 100.0, "mu0": 0.0, "sigma": 2300.0, "sigma0": 2.7, "darkcount": 0.05}
    if parameters is not None:
        p.update(parameters)
    clean = np.asarray(clean, float)
    if not np.isfinite(clean).all() or np.any(clean + p["darkcount"] < 0):
        raise ValueError("Poisson rate must be finite and nonnegative")
    if p["mu"] <= 0 or min(p["sigma"], p["sigma0"], p["darkcount"]) < 0:
        raise ValueError("Invalid detector parameters")
    if draws is None:
        rng = np.random.default_rng() if rng is None else rng
        draws = {
            "counts": rng.poisson(clean + p["darkcount"]),
            "gain_normal": rng.standard_normal(clean.shape),
            "electronics_normal": rng.standard_normal(clean.shape),
        }
    count, gain_z, electronic_z = [
        np.asarray(draws[k]) for k in ("counts", "gain_normal", "electronics_normal")
    ]
    if (
        any(
            a.shape != clean.shape or not np.isfinite(a).all()
            for a in (count, gain_z, electronic_z)
        )
        or np.any(count < 0)
        or np.any(count != np.floor(count))
    ):
        raise ValueError("Noise draws must match the image and counts must be nonnegative integers")
    count = count.astype(np.float64)
    mean = count * p["mu"]
    variance = count * p["sigma"]
    positive = count > 0
    signal = np.zeros(clean.shape)
    mu = np.log(mean[positive] ** 2 / np.sqrt(variance[positive] + mean[positive] ** 2))
    sd = np.sqrt(np.log1p(variance[positive] / mean[positive] ** 2))
    signal[positive] = np.exp(mu + sd * gain_z[positive])
    result = signal + p["mu0"] + p["sigma0"] * electronic_z
    return np.sign(result) * np.floor(np.abs(result) + 0.5)


def pixel_bleed(frame, probability=0.3, maximum=0.4, *, uniform=None, rng=None):
    if not 0 <= probability <= 1 or not 0 <= maximum <= 1:
        raise ValueError("Bleed parameters must lie in [0,1]")
    frame = np.asarray(frame)
    if probability == 0:
        return frame.copy()
    if uniform is None:
        rng = np.random.default_rng() if rng is None else rng
        uniform = rng.random(frame.shape)
    uniform = np.asarray(uniform)
    if uniform.shape != frame.shape or np.any((uniform < 0) | (uniform > 1)):
        raise ValueError("Invalid bleed draws")
    bleed = maximum * np.maximum(uniform - (1 - probability), 0) / probability
    # The upstream boundary carries the last column of the previous X row.
    carried = np.zeros(frame.size, dtype=np.result_type(frame, bleed))
    carried[1:] = (frame * bleed).ravel(order="C")[:-1]
    return frame - frame * bleed + carried.reshape(frame.shape)


def scan_movie(
    components,
    activity,
    psf,
    *,
    mask=None,
    collection_mask=None,
    z_start=None,
    scan_avg=2,
    sfrac=2,
    buffer=0,
    motion=None,
    tails=(),
    signal_scale=1.0,
    activity_cutoff=0.01,
    noise_parameters=None,
    noise_draws=None,
    bleed_uniforms=None,
    rng=None,
    rana=None,
):
    """Scan dense component volumes K,X,Y,Z into clean and optional noisy movies.

    Tail dictionaries contain kernel, indices, power_ratio and optional mask and
    zscale, in the same units as blurred_background. Activity is K,T. The API
    accepts a supplied motion path and never redraws it behind the caller.
    """
    started = time.perf_counter()
    components = np.asarray(components, np.float32)
    activity = np.asarray(activity, np.float32)
    psf = np.asarray(psf, np.float32)
    if components.ndim != 4 or activity.ndim != 2 or components.shape[0] != activity.shape[0]:
        raise ValueError("Expected KXYZ components and KT activity")
    if not np.isfinite(components).all() or not np.isfinite(activity).all():
        raise ValueError("Inputs must be finite")
    if not isinstance(sfrac, (int, np.integer)) or sfrac < 1:
        raise ValueError("This aligned interface requires integer pixel binning")
    n, m, nz = components.shape[1:]
    nt = activity.shape[1]
    common = np.ones((n, m), np.float32)
    if mask is not None:
        common *= np.maximum(np.asarray(mask, np.float32), 1e-5)
    if collection_mask is not None:
        common *= np.asarray(collection_mask, np.float32)
    weighted = components * common[None, :, :, None]
    baseline = activity.min(axis=1, keepdims=True)
    changes = activity - baseline
    changes[changes < activity_cutoff] = 0
    activity = baseline + changes
    depth = psf.shape[2] if rana is None or rana is False else rana.kernel_shape[2]
    z_start = max(0, int(np.floor((nz - depth) / 2)) - 1) if z_start is None else z_start
    starts = np.broadcast_to(np.asarray(z_start, dtype=int), (nt,))
    if np.any(starts < 0) or np.any(starts + depth > nz):
        raise ValueError("Focal slab outside component volume")
    freq = psf_fft((n, m, psf.shape[2]), psf, scan_avg) if rana is None or rana is False else None
    if rana is not None and rana is not False:
        prepared = []
        for spec in tails:
            spec = dict(spec)
            if spec.get("frequency", False):
                intensity = ifft2(spec["kernel"], axes=(0, 1)).real
                spec["kernel"] = fft2(rana.physical_tail(intensity), axes=(0, 1))
            else:
                spec["kernel"] = rana.physical_tail(spec["kernel"])
            if spec.get("mask") is not None:
                spec["mask"] = rana.physical_tail(spec["mask"])
            prepared.append(spec)
        tails = prepared
    clean = []
    noisy = []
    for t in range(nt):
        volume = np.einsum("kxyz,k->xyz", weighted, activity[:, t], optimize=True)
        z = starts[t]
        frame = single_scan(
            volume[:, :, z : z + depth], psf, scan_avg=scan_avg, frequency_psf=freq, rana=rana, z_start=z
        ) * (signal_scale / (2 * sfrac * sfrac))
        for spec in tails:
            kw = dict(spec)
            kernel = kw.pop("kernel")
            indices = kw.pop("indices")
            frame += blurred_background(volume, indices, kernel, **kw) * (
                signal_scale / (sfrac * sfrac)
            )
        xo = buffer + 1 if motion is None else np.asarray(motion["x_offsets"])[t]
        yo = buffer + 1 if motion is None else np.asarray(motion["y_offsets"])[t]
        frame = row_motion(frame, buffer, xo, yo)
        frame = conv2_same(frame, np.ones((sfrac, sfrac)))[::sfrac, ::sfrac].astype(np.float32)
        clean.append(frame)
        if noise_parameters is not None:
            draws = None if noise_draws is None else noise_draws[t]
            sampled = detector_noise(frame, noise_parameters, draws=draws, rng=rng)
            u = None if bleed_uniforms is None else bleed_uniforms[t]
            sampled = pixel_bleed(
                sampled,
                noise_parameters.get("bleedp", 0.3),
                noise_parameters.get("bleedw", 0.4),
                uniform=u,
                rng=rng,
            )
            noisy.append(sampled.astype(np.float32))
    return {
        "clean": np.stack(clean, axis=2),
        "noisy": np.stack(noisy, axis=2) if noisy else None,
        "seconds": time.perf_counter() - started,
        "frames": nt,
        "motion_supplied": motion is not None,
    }


import math


def check_tpm_params(tpm_params):
    if tpm_params is None:
        tpm_params = {}
    dParams = {
        "nidx": 1.33,
        "nac": 0.8,
        "eta": 0.6,
        "conc": 10,
        "delta": 2,
        "gp": 0.588,
        "f": 80,
        "tau": 150,
        "pavg": 40,
        "lambda": 0.92,
    }
    for k, v in dParams.items():
        if k not in tpm_params:
            tpm_params[k] = v
    if "phi" not in tpm_params or tpm_params["phi"] is None:
        tpm_params["phi"] = (
            0.8 * ((1 - math.sqrt(1 - (tpm_params["nac"] / tpm_params["nidx"]) ** 2)) / 2) * 0.4
        )
    return tpm_params


def tpmSignalscale(tpm_params, psf_params=None):
    nidx = (
        float(psf_params.get("n", tpm_params["nidx"])) if psf_params else float(tpm_params["nidx"])
    )
    nac = (
        float(psf_params.get("objNA", tpm_params["nac"]))
        if psf_params
        else float(tpm_params["nac"])
    )
    lam = (
        float(psf_params.get("lambda", tpm_params["lambda"]))
        if psf_params
        else float(tpm_params["lambda"])
    )

    phi, eta, conc = float(tpm_params["phi"]), float(tpm_params["eta"]), float(tpm_params["conc"])
    delta, gp, f = float(tpm_params["delta"]), float(tpm_params["gp"]), float(tpm_params["f"])
    tau, pavg = float(tpm_params["tau"]), float(tpm_params["pavg"])

    conc = conc * 1e-6 * 6.02e23 * 1e3
    delta = delta * 1e-58
    f = f * 1e6
    tau = tau * 1e-15
    lam = lam * 1e-6
    pavg = 1e-3 * pavg / (6.626e-34 * 3e8 / lam)

    return phi * eta * conc * delta * gp * 8 * nidx * (pavg**2) / (2 * f * tau * math.pi * lam)


def sample_motion(shape, psf_depth, frames, *, buffer=2, zoffset=0, enabled=True, rng=None):
    """Generate the upstream motion law; cross-language RNG sequences may differ."""
    rng = np.random.default_rng() if rng is None else rng
    n, _, nz = shape
    zbase = int(np.floor(0.5 * (nz - psf_depth))) + zoffset
    x = y = buffer + 1
    z = zbase
    result = []
    drift = np.r_[-1, 1, np.zeros(5)].astype(int) if enabled else np.zeros(3, int)
    axial = np.r_[-1, 1, np.zeros(100)].astype(int) if enabled else np.zeros(3, int)
    jump = np.arange(-3, 4) if enabled else np.zeros(3, int)
    bound = 2 if enabled else 0
    for _ in range(frames):
        if rng.random() > (0.05 if enabled else 0):
            x = np.clip(x + rng.choice(jump), 1, 2 * buffer + 1)
            y = np.clip(y + rng.choice(jump), 1, 2 * buffer + 1)
        xx = np.clip(x + rng.choice(drift), 1, 2 * buffer + 1)
        yy = np.clip(y + rng.choice(drift), 1, 2 * buffer + 1)
        z = np.clip(z + rng.choice(axial), zbase - bound, zbase + bound)
        z = int(np.clip(z + rng.choice(drift), 1, nz - psf_depth + 1))
        prefix = np.zeros(rng.integers(1, max(2, int(np.floor(2 * n / 5)) + 1)))
        length = int(np.floor(rng.random() * 3 * n / 5 + 0.5))
        shear = np.r_[
            prefix,
            np.linspace(0, 1, length) * (2 * (rng.random() - 0.5)) * (n / 200 if enabled else 0),
        ]
        shear = np.pad(shear, (0, max(0, n - len(shear))), mode="edge")[:n]
        yrow = np.clip(yy + shear + rng.choice(drift, size=n), 1, 2 * buffer + 1)
        result.append({"x": int(xx), "y": np.floor(yrow + 0.5), "z": z})
    return result


def scan_volume(
    neurons,
    optics,
    activity,
    scan=None,
    noise=None,
    spikes=None,
    tpm=None,
    *,
    motion=None,
    noise_draws=None,
    bleed_uniforms=None,
    rng=None,
    add_noise=True
):
    """Collect iter_scan_volume frames into the traditional in-memory result."""
    clean = []
    noisy = []
    positions = []
    last = None
    for frame in iter_scan_volume(
        neurons,
        optics,
        activity,
        scan,
        noise,
        spikes,
        tpm,
        motion=motion,
        noise_draws=noise_draws,
        bleed_uniforms=bleed_uniforms,
        rng=rng,
        add_noise=add_noise,
    ):
        clean.append(frame["clean"])
        positions.append(frame["motion"])
        last = frame
        if frame["noisy"] is not None:
            noisy.append(frame["noisy"])
    if last is None:
        raise ValueError("At least one activity frame is required")
    return dict(
        clean=np.stack(clean, axis=2),
        noisy=np.stack(noisy, axis=2) if noisy else None,
        seconds=last["seconds"],
        frames=last["frames"],
        signal_scale=last["signal_scale"],
        motion=positions,
    )


def iter_scan_volume(
    neurons,
    optics,
    activity,
    scan=None,
    noise=None,
    spikes=None,
    tpm=None,
    *,
    motion=None,
    noise_draws=None,
    bleed_uniforms=None,
    rng=None,
    add_noise=True
):
    """Aligned cytosolic or nuclear scan_volume interface with sparse NAOMi components.

    Sparse gp_vals/gp_nuc/bg_proc use one-based MATLAB column-major indices.
    Soma/dendrite writes preserve the upstream MEX overwrite semantics; axonal
    background accumulates. A supplied motion list uses upstream one-based
    x/y/z positions. Returned arrays can be exported separately.
    """
    if "channels" in optics:
        yield from _iter_multibeam(
            neurons,
            optics,
            activity,
            scan,
            noise,
            spikes,
            tpm,
            motion=motion,
            noise_draws=noise_draws,
            bleed_uniforms=bleed_uniforms,
            rng=rng,
            add_noise=add_noise,
        )
        return
    started = time.perf_counter()
    p = {"scan_buff": 10, "scan_avg": 2, "sfrac": 2, "motion": True, "zoffset": 0}
    p.update(scan or {})
    nuclear = p.get("nuc_label", 0) >= 1
    if not np.isfinite(p["sfrac"]) or p["sfrac"] <= 0:
        raise ValueError("sfrac must be positive")
    if p.get("fsimPath") or p.get("fsimCleanPath"):
        raise NotImplementedError("Use returned arrays for export")
    sf = float(p["sfrac"])
    buffer = int(p["scan_buff"])
    psf = np.asarray(optics["psf"], np.float32)
    rana = optics.get('rana')
    depth = psf.shape[2] if rana is None else rana.kernel_shape[2]
    shape = tuple(np.asarray(neurons["neur_vol"]).shape)
    n, m, nz = shape
    common = None
    if optics.get("mask") is not None:
        common = np.maximum(np.asarray(optics["mask"], np.float32), 1e-5)
        if optics.get("colmask") is not None:
            common = common * np.asarray(optics["colmask"], np.float32)
    mask_flat = None if common is None else common.ravel(order="F")

    def component(indices, values, use_mask=True):
        indices = np.asarray(indices, dtype=int).ravel() - 1
        values = np.asarray(values, np.float32).ravel()
        if np.any(indices < 0) or np.any(indices >= np.prod(shape)):
            raise ValueError("Invalid one-based component index")
        if values.size == 1:
            values = np.full(len(indices), values[0], np.float32)
        if len(values) != len(indices):
            raise ValueError("One component value per index is required")
        if mask_flat is not None and use_mask:
            # Preserve upstream max(1,mod(index,N1*N2)) at the last voxel of a plane.
            values = values * mask_flat[np.maximum(1, (indices + 1) % (n * m)) - 1]
        return indices, values

    soma = []
    dend = []
    raw_cells = neurons["gp_vals"]
    # scipy.io.loadmat(simplify_cells=True) squeezes a single MATLAB cell row.
    if isinstance(raw_cells, np.ndarray) and raw_cells.shape == (3,):
        raw_cells = [raw_cells]
    for ids, values, issoma in raw_cells:
        ids = np.asarray(ids).ravel()
        values = np.asarray(values).ravel()
        issoma = np.asarray(issoma, bool).ravel()
        soma.append(component(ids[issoma], values[issoma]))
        dend.append(component(ids[~issoma], values[~issoma]))
    raw_bg = neurons.get("bg_proc", [])
    # A MATLAB cell row may simplify to a flat two-entry sequence.
    if isinstance(raw_bg, np.ndarray) and raw_bg.shape == (2,):
        raw_bg = [raw_bg]
    bg = [component(ids, values) for ids, values in raw_bg]
    soma_activity = np.atleast_2d(activity["soma"]).astype(np.float32)
    nt = soma_activity.shape[1]
    activities = [
        soma_activity,
        np.atleast_2d(activity.get("dend", soma_activity)).astype(np.float32),
        np.atleast_2d(activity.get("bg", np.zeros((len(bg), nt)))).astype(np.float32),
    ]
    if (
        activities[0].shape[0] != len(soma)
        or activities[1].shape[0] != len(dend)
        or activities[2].shape[0] != len(bg)
    ):
        raise ValueError("Component/activity counts differ")
    nucleus_activity = activities[0].copy() if nuclear else None
    if nuclear:
        activities = [np.zeros_like(a) for a in activities]
    mins = [a.min(axis=1) for a in activities]
    delta = [a - mi[:, None] for a, mi in zip(activities, mins)]
    for a in delta:
        a[a < 0.01] = 0
    base = np.zeros(np.prod(shape), np.float32)
    for (ids, values), amp in zip(soma, mins[0]):
        base[ids] = values * amp
    raw_nuc = neurons.get("gp_nuc", [])
    if isinstance(raw_nuc, np.ndarray) and raw_nuc.shape == (2,):
        raw_nuc = [raw_nuc]
    for j, (ids, values) in enumerate(raw_nuc):
        ids, values = component(ids, values, use_mask=False)
        if len(ids) and np.any(values):
            base[ids] = values * mins[0][j]
    for (ids, values), amp in zip(dend, mins[1]):
        base[ids] = values * amp
    for (ids, values), amp in zip(bg, mins[2]):
        np.add.at(base, ids, values * amp)
    nucleus = []
    nucleus_delta = None
    if nuclear:
        # Upstream subtracts bg_min here, not nuc_min; retain that behavior.
        nucleus_delta = (
            nucleus_activity.copy()
        )  # Background is identically zero in this mode; its row count is irrelevant.
        nucleus_delta[nucleus_delta < 0.01] = 0
        for j, (ids, values) in enumerate(raw_nuc):
            ids, values = component(ids, 1.0 if common is not None else values)
            nucleus.append((ids, values))
            base[ids] = values * nucleus_activity[j].min()
    base = base.reshape(shape, order="F")
    defaults = _noise_defaults(noise)
    scale = (
        tpmSignalscale(check_tpm_params(dict(tpm or {})))
        * (spikes or {}).get("dt", 1 / 30)
        * sf
        * sf
        / (250 * 250)
    )
    freq = psf_fft(shape, psf, int(p["scan_avg"])) if rana is None else None
    tails = []
    for name in ["psfT", "psfB"]:
        spec = optics.get(name)
        if spec is None:
            continue
        kernel = np.asarray(spec["convmask"], np.float32)
        if rana is not None:
            kernel = rana.physical_tail(kernel)
        kernel = kernel / kernel.sum()
        tm = np.asarray(spec["mask"], np.float32)
        if rana is not None:
            tm = rana.physical_tail(tm)
        tm = tm / tm.mean()
        tails.append(
            {
                "kernel": psf_fft(shape, kernel),
                "weight": float(spec["weight"]),
                "mask": tm / (common if common is not None else 1),
                "zscale": spec.get("psfZ"),
            }
        )
    if len(tails) not in (0, 2):
        raise ValueError("Both optical tails are required")
    if motion is None:
        motion = sample_motion(
            shape,
            depth,
            nt,
            buffer=buffer,
            zoffset=p["zoffset"],
            enabled=p["motion"],
            rng=rng,
        )
    if len(motion) != nt:
        raise ValueError("One motion record per frame is required")
    for t in range(nt):
        temp = np.zeros(np.prod(shape), np.float32)
        for groups, changes, add in [
            (soma, delta[0], False),
            (nucleus, nucleus_delta, False),
            (dend, delta[1], False),
            (bg, delta[2], True),
        ]:
            for j, (ids, values) in enumerate(groups):
                if changes[j, t] > 0:
                    if add:
                        np.add.at(temp, ids, values * changes[j, t])
                    else:
                        temp[ids] = values * changes[j, t]
        temp = temp.reshape(shape, order="F")
        position = motion[t]
        z = int(position["z"]) - 1
        if z < 0 or z + depth > nz:
            raise ValueError("Motion places focal slab outside volume")
        frame = single_scan(
            temp[:, :, z : z + depth] + base[:, :, z : z + depth],
            psf,
            scan_avg=int(p["scan_avg"]),
            frequency_psf=freq,
            rana=rana,
            z_start=z,
        ) * (scale / (2 * sf * sf))
        if "g_blur" in optics:
            frame += conv2_same(frame, optics["blur"])
        for j, spec in enumerate(tails):
            indices = np.arange(z) if j == 0 else np.arange(z + depth, nz)
            zs = spec["zscale"]
            if len(indices) == 0:
                indices = np.arange(nz)
                zs = None
            frame += blurred_background(
                temp,
                indices,
                spec["kernel"],
                power_ratio=spec["weight"],
                mask=spec["mask"],
                zscale=zs,
                extra_volumes=[base],
                frequency=True,
            ) * (scale / (sf * sf))
        frame = row_motion(frame, buffer, position["x"], np.floor(np.asarray(position["y"]) + 0.5))
        if sf == int(sf):
            si = int(sf)
            frame = conv2_same(frame, np.ones((si, si)))[::si, ::si].astype(np.float32)
        else:
            frame = (sf * sf * _resize_bilinear(frame, 1 / sf)).astype(np.float32)
        noisy_frame = None
        if add_noise:
            noisy_frame = detector_noise(
                frame, defaults, draws=None if noise_draws is None else noise_draws[t], rng=rng
            )
            noisy_frame = pixel_bleed(
                noisy_frame,
                defaults["bleedp"],
                defaults["bleedw"],
                uniform=None if bleed_uniforms is None else bleed_uniforms[t],
                rng=rng,
            )
            noisy_frame = noisy_frame.astype(np.float32)
        yield {
            "clean": frame,
            "noisy": noisy_frame,
            "index": t,
            "seconds": time.perf_counter() - started,
            "frames": nt,
            "signal_scale": scale,
            "motion": position,
        }


def _resize_bilinear(image, scale):
    """MATLAB imresize bilinear: half-pixel centres, symmetric antialiasing."""
    out = np.asarray(image)
    for axis in (0, 1):
        n = out.shape[axis]
        m = int(np.ceil(n * scale))
        a = min(scale, 1.0)
        u = (np.arange(m) + 0.5) / scale - 0.5
        width = 2 / a
        idx = np.floor(u - width / 2).astype(int)[:, None] + np.arange(int(np.ceil(width)) + 2)
        w = np.maximum(0, 1 - np.abs((u[:, None] - idx) * a)) * a
        w = (w / w.sum(axis=1, keepdims=True)).astype(out.dtype)
        idx %= 2 * n
        idx = np.where(idx < n, idx, 2 * n - 1 - idx)
        out = np.moveaxis((np.moveaxis(out, axis, 0)[idx] * w[..., None]).sum(axis=1), 0, axis)
    return out


def _noise_defaults(noise):
    return dict(
        {
            "mu": 100.0,
            "mu0": 0.0,
            "sigma": 2300.0,
            "sigma0": 2.7,
            "darkcount": 0.05,
            "bleedp": 0.3,
            "bleedw": 0.4,
        },
        **(noise or {})
    )


def _iter_multibeam(
    neurons,
    optics,
    activity,
    scan,
    noise,
    spikes,
    tpm,
    *,
    motion,
    noise_draws,
    bleed_uniforms,
    rng,
    add_noise
):
    """Scan each incoherent beam, share motion, then apply the detector once."""
    channels = optics["channels"]
    weights = optics.get("beam_weights", {k: 1.0 for k in channels})
    if (
        not channels
        or set(weights) != set(channels)
        or not all(np.isfinite(w) and w >= 0 for w in weights.values())
        or sum(weights.values()) <= 0
    ):
        raise ValueError("Invalid beam weights")
    p = {"scan_buff": 10, "scan_avg": 2, "sfrac": 2, "motion": True, "zoffset": 0}
    p.update(scan or {})
    shape = np.asarray(neurons["neur_vol"]).shape
    nt = np.atleast_2d(activity["soma"]).shape[1]
    depths = {o['rana'].kernel_shape[2] if o.get('rana') is not None else np.asarray(o['psf']).shape[2] for o in channels.values()}
    if len(depths) != 1:
        raise ValueError("Beams must share focal depth and acquisition geometry")
    rng = np.random.default_rng() if rng is None else rng
    if motion is None:
        motion = sample_motion(
            shape,
            next(iter(depths)),
            nt,
            buffer=int(p["scan_buff"]),
            zoffset=p["zoffset"],
            enabled=p["motion"],
            rng=rng,
        )
    streams = {
        key: iter_scan_volume(
            neurons, value, activity, p, spikes=spikes, tpm=tpm, motion=motion, add_noise=False
        )
        for key, value in channels.items()
    }
    defaults = _noise_defaults(noise)
    started = time.perf_counter()
    for t in range(nt):
        items = {k: next(v) for k, v in streams.items()}
        clean = sum(weights[k] * v["clean"] for k, v in items.items()).astype(np.float32)
        noisy = None
        if add_noise:
            noisy = detector_noise(
                clean, defaults, draws=None if noise_draws is None else noise_draws[t], rng=rng
            )
            noisy = pixel_bleed(
                noisy,
                defaults["bleedp"],
                defaults["bleedw"],
                uniform=None if bleed_uniforms is None else bleed_uniforms[t],
                rng=rng,
            ).astype(np.float32)
        first = next(iter(items.values()))
        yield dict(
            first,
            clean=clean,
            noisy=noisy,
            seconds=time.perf_counter() - started,
            channel_clean={k: v["clean"] for k, v in items.items()},
        )
