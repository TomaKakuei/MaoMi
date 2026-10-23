"""Connect explicit exposure clocks to MaoMi's sparse anatomy and optics."""
from dataclasses import asdict, replace
import time
import numpy as np
from scipy.ndimage import shift
from .acquisition import AcquisitionClock
from ._movie_buffers import ActivitySource, SparseMovieBuffer


def _rate_operator(operator):
    """Resolve declared integrated kernels to rates without changing archives."""
    if operator is None:
        return None
    from .rana import DepthPSFBank, RanaOperator
    fields, changed = [], False
    for field in operator.fields:
        meta = field.provenance
        duration = meta.get("kernel_exposure_s", 1.)
        if "kernel_exposure_s" not in meta and meta.get("model") == "RANA Richards-Wolf layered pupil":
            duration = meta["acquisition"]["dwell_seconds"]
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Declared PSF kernel exposure must be finite and positive")
        changed |= duration != 1.
        fields.append(field if duration == 1. else replace(field, kernels=field.kernels/duration,
                      provenance=dict(meta, timing_rate_divisor_s=duration)))
    if not changed:
        return operator
    source = DepthPSFBank(operator.bank.depths_um, tuple(fields)) if operator.bank is not None else fields[0]
    return RanaOperator(source, native=operator.native, origin_um=operator.origin_um,
                        z_origin_um=operator.z_origin_um, device=operator.device)


def timing_settings(scan, shape, dt):
    """Validate and resolve settings without preparing optics or output files."""
    spec = dict(scan["timing"])
    velocity = np.asarray(spec.pop("velocity_voxels_s", (0., 0., 0.)), float)
    gain = float(spec.pop("photon_rate_scale", 1.))
    if velocity.shape != (3,) or not np.isfinite(velocity).all():
        raise ValueError("velocity_voxels_s must contain finite XYZ specimen velocities")
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("photon_rate_scale and frame interval must be positive and finite")
    clock = AcquisitionClock(**spec)
    buffer, sf = scan.get("scan_buff", 10), scan.get("sfrac", 2)
    if (not isinstance(buffer, (int, np.integer)) or buffer < 0
            or int(sf) != sf or sf < 1):
        raise ValueError("Timed acquisition requires a nonnegative integer buffer and integer binning")
    crop = np.asarray(shape[:2])-2*buffer
    if (crop <= 0).any() or (crop % int(sf) != 0).any():
        raise ValueError("Timed acquisition crop must contain complete pixel bins")
    output_shape = tuple(crop//int(sf))
    offsets = clock.offsets(output_shape)
    if offsets.max()-offsets.min()+clock.exposure_s > dt*(1+1e-12):
        raise ValueError("Acquisition window must fit the frame interval")
    return clock, velocity, gain, buffer, int(sf), output_shape


def timing_metadata(scan, shape, dt):
    clock, velocity, gain, _, _, output_shape = timing_settings(scan, shape, dt)
    return dict(clock=asdict(clock), output_shape_xy=list(output_shape),
                frame_interval_s=dt, frame_midpoints="index * frame_interval_s",
                velocity_voxels_s=velocity.tolist(), photon_rate_scale=gain,
                activity_interpolation="linear at frame midpoints; endpoint hold",
                motion="rigid fluorescence translation before fixed optics and masks; zero outside volume",
                units="integrated expected photons; photon_rate_scale calibrates optical response",
                detector="NAOMi-compatible detector noise, sampled once after integration",
                temporal_model="piecewise-linear optical rate snapshots",
                kernel_exposure="declared RANA dwell or kernel_exposure_s divided out before timing integration",
                axes="XY, slow scan Y, fast scan X")


def iter_timed_volume(neurons, optics, activity, scan, noise, spikes, *, motion=None,
                      noise_draws=None, bleed_uniforms=None, rng=None, add_noise=True,
                      start_frame=0, frame_seed=None):
    from .movie import single_scan, psf_fft, blurred_background, conv2_same
    from .movie import detector_noise, pixel_bleed, _noise_defaults
    from .rana.operator import RanaOperator
    if motion is not None:
        raise ValueError("Timed acquisition uses timing.velocity_voxels_s; omit legacy motion records")
    started = time.perf_counter()
    shape = tuple(neurons["neur_vol"].shape)
    dt = float((spikes or {}).get("dt", 1/30))
    clock, velocity, gain, buffer, sf, output_shape = timing_settings(scan, shape, dt)
    plan = clock.plan(output_shape)
    nuclear = scan.get("nuc_label", 0) >= 1
    rows = neurons["gp_vals"]
    if isinstance(rows, np.ndarray) and rows.shape == (3,):
        rows = [rows]

    def component(ids, values):
        ids = np.asarray(ids, np.int64).ravel()-1
        values = np.broadcast_to(np.asarray(values, np.float32).ravel(), ids.shape).copy()
        if (ids < 0).any() or (ids >= np.prod(shape)).any():
            raise ValueError("Invalid one-based component index")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Fluorescence components must be finite and nonnegative")
        return ids, values

    soma, dend = [], []
    for ids, values, is_soma in rows:
        ids, values = np.asarray(ids).ravel(), np.asarray(values).ravel()
        selection = np.asarray(is_soma, bool).ravel()
        soma.append(component(ids[selection], values[selection]))
        dend.append(component(ids[~selection], values[~selection]))
    bg_rows, nuc_rows = neurons.get("bg_proc", []), neurons.get("gp_nuc", [])
    if isinstance(bg_rows, np.ndarray) and bg_rows.shape == (2,):
        bg_rows = [bg_rows]
    if isinstance(nuc_rows, np.ndarray) and nuc_rows.shape == (2,):
        nuc_rows = [nuc_rows]
    bg = [component(*row) for row in bg_rows]
    nuc = [component(*row) for row in nuc_rows] if nuclear else []
    if nuclear and len(nuc) != len(soma):
        raise ValueError("Timed nuclear acquisition requires one nuclear component per cell")
    source = ActivitySource(activity, (len(soma), len(dend), len(bg)),
                            chunk_frames=scan.get("chunk_frames", 256))
    if not isinstance(start_frame, (int, np.integer)) or not 0 <= start_frame <= source.frames:
        raise ValueError("start_frame must be between zero and frame count")
    groups = ((nuc, False),) if nuclear else ((soma, False), (dend, False), (bg, True))
    writer = SparseMovieBuffer(shape, groups, backend=scan.get("sparse_backend", "auto"))
    psf = np.asarray(optics["psf"])
    rana = _rate_operator(optics.get("rana"))
    depth = psf.shape[2] if rana is None else rana.kernel_shape[2]
    z = (shape[2]-depth)//2 + int(scan.get("zoffset", 0))
    if z < 0 or z+depth > shape[2]:
        raise ValueError("Focal slab outside volume")
    avg = scan.get("scan_avg", 2)
    if not isinstance(avg, (int, np.integer)) or avg < 1:
        raise ValueError("scan_avg must be a positive integer")
    freq = psf_fft(shape, psf, avg) if rana is None else None
    common = np.ones(shape[:2], np.float32)
    for key in ("mask", "colmask"):
        if optics.get(key) is not None:
            mask = np.asarray(optics[key])
            if mask.shape != shape[:2] or not np.isfinite(mask).all() or (mask < 0).any():
                raise ValueError("Optical source masks must be finite nonnegative XY arrays")
            common *= mask
    tails = []
    for j, name in enumerate(("psfT", "psfB")):
        if name not in optics:
            continue
        spec = optics[name]
        kernel = RanaOperator.physical_tail(np.asarray(spec["convmask"]))
        tm = RanaOperator.physical_tail(np.asarray(spec["mask"]))
        indices = np.arange(z) if j == 0 else np.arange(z+depth, shape[2])
        if len(indices):
            tails.append(dict(indices=indices, kernel=kernel/kernel.sum(),
                              power_ratio=spec["weight"], mask=tm/tm.mean(), zscale=spec.get("psfZ")))
    if ("psfT" in optics) != ("psfB" in optics):
        raise ValueError("Both optical tails are required")
    crop = tuple(slice(buffer, n-buffer) for n in shape[:2])
    stationary = not np.any(velocity)
    cached_amplitudes, cached_rate = None, None

    def render_rate(seconds):
        nonlocal cached_amplitudes, cached_rate
        amplitudes = source.at_time(seconds/dt)
        if stationary and cached_amplitudes is not None and all(
                np.array_equal(a, b) for a, b in zip(amplitudes, cached_amplitudes)):
            return cached_rate
        density = writer.frame([amplitudes[0]] if nuclear else amplitudes)
        if np.any(velocity):
            density = shift(density, velocity*seconds, order=1, mode="constant",
                            cval=0., prefilter=False)
        weighted = density*common[..., None]
        frame = single_scan(weighted[:, :, z:z+depth], psf, scan_avg=avg,
                            frequency_psf=freq, rana=rana, z_start=z)
        if "g_blur" in optics:
            frame += conv2_same(frame, optics["blur"])
        for tail in tails:
            frame += 2*blurred_background(density, **tail)
        if not np.isfinite(frame).all() or frame.min() < -1e-5*max(1., float(frame.max())):
            raise ValueError("Optical model returned negative or nonfinite photon rates")
        frame = np.maximum(frame[crop], 0.)
        rate = gain*frame.reshape(output_shape[0], sf, output_shape[1], sf).sum(axis=(1, 3))
        if stationary:
            cached_amplitudes, cached_rate = amplitudes, rate
        return rate

    defaults = _noise_defaults(noise)
    for index in range(start_frame, source.frames):
        clean = plan.integrate(render_rate, center_s=index*dt).astype(np.float32)
        noisy = None
        frame_rng = np.random.default_rng(np.random.SeedSequence([frame_seed, index])) if frame_seed is not None else rng
        if add_noise:
            noisy = detector_noise(clean, defaults, draws=None if noise_draws is None else noise_draws[index], rng=frame_rng)
            noisy = pixel_bleed(noisy, defaults["bleedp"], defaults["bleedw"],
                                uniform=None if bleed_uniforms is None else bleed_uniforms[index], rng=frame_rng).astype(np.float32)
        yield dict(clean=clean, noisy=noisy, index=index, seconds=time.perf_counter()-started,
                   frames=source.frames, signal_scale=gain, motion=None, time_s=index*dt)
