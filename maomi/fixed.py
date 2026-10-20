"""MATLAB-compatible native optics, with explicit full/focal-only outputs."""

import time
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
import torch
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.distance import cdist
from .optics import prepare_fields, pupil, collection_mask, legacy
from .numerics import v4_interpolate
from .scanning import zstack


def fixed_locations(config, count_per_axis=65):
    """Uniform physical XY locations snapped to the supersampled lattice."""
    if isinstance(count_per_axis, bool) or not isinstance(count_per_axis, int) or count_per_axis < 2:
        raise ValueError("count_per_axis must be an integer >= 2")
    ticks = np.asarray(config.volume_um[:2], float) * config.vres * config.ss
    if (
        not np.all(np.isfinite(ticks))
        or np.any(ticks < count_per_axis - 1)
        or not np.allclose(ticks, np.round(ticks))
    ):
        raise ValueError(
            f"Fixed {count_per_axis}x{count_per_axis} requires an integer optical lattice "
            f"with at least {count_per_axis} positions per axis; choose count_per_axis explicitly"
        )
    axes = [np.round(np.linspace(0, int(round(n)), count_per_axis)) / (config.vres * config.ss) for n in ticks]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
    if len(np.unique(points, axis=0)) != count_per_axis ** 2:
        raise ValueError("Grid contains duplicate locations")
    return points


def field_shapes(config):
    v, p = config.parameters()
    r = config.vres
    n = int(round((v.vasc_sz[0] - v.vol_sz[0]) * r))
    n2 = (
        int(round(min(max(legacy.gaussian_beam_size(p, config.psf_um[2] / 2, 3)) * r * 2, n)))
        if config.propcrop
        else n
    )
    za = int(round((config.depth_um + config.volume_um[2] / 2 - config.psf_um[2] / 2) * r))
    zb = int(round(config.psf_um[2] * r))
    pc = int(round(config.prop_um * r))
    nc = int(round(config.tail_um * r))
    if min(za, zb, pc, nc) <= 0:
        raise ValueError("Optical A/B/C propagation extents must be positive")
    xy = np.round(np.asarray(config.volume_um[:2]) * r).astype(int)
    return [
        (xy[0] + n, xy[1] + n, int(np.ceil(za / pc))),
        (xy[0] + n2, xy[1] + n2, zb),
        (xy[0] + n, xy[1] + n, int(np.ceil(nc / pc))),
    ]


def _cubic(d):
    return d * d * d


class CubicField:
    """Affine-augmented cubic RBF; stream query bases to limit host memory."""

    def __init__(self, config, points, shape):
        self.extent = np.asarray(config.volume_um[:2], float)
        self.p = np.asarray(points) / self.extent
        self.shape = tuple(shape)
        n = len(points)
        poly = np.c_[np.ones(n), self.p]
        self.system = np.block([[_cubic(cdist(self.p, self.p)), poly], [poly.T, np.zeros((3, 3))]])
        lattice = np.round(self.extent * config.vres * config.ss).astype(int)
        a, b = [np.arange(k + 1) / k for k in lattice]
        targets = np.stack(np.meshgrid(a, b, indexing="ij"), -1).reshape(-1, 2)
        mean = np.zeros(n)
        for i in range(0, len(targets), 1024):
            mean += _cubic(cdist(targets[i : i + 1024], self.p)).sum(0)
        self.mean_basis = np.r_[mean / len(targets), 1.0, 0.5, 0.5]
        self.factor = lu_factor(self.system)
        self.mean_weights = lu_solve(self.factor, self.mean_basis)[:n]
        axes = [(np.arange(k) + 0.5) / k for k in shape]
        self.pixels = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
        self.pixel_basis = None
        if len(self.pixels) * n <= 2_000_000:
            self.pixel_basis = np.c_[
                _cubic(cdist(self.pixels, self.p)), np.ones(len(self.pixels)), self.pixels
            ]

    def reconstruct(self, values):
        values = np.asarray(values, np.float64)
        if values.ndim == 1:
            values = values[:, None]
        solution = lu_solve(self.factor, np.r_[values, np.zeros((3, values.shape[1]))])
        if self.pixel_basis is not None:
            return self.mean_weights, (self.pixel_basis @ solution).reshape(
                *self.shape, values.shape[1]
            )
        blocks = []
        for i in range(0, len(self.pixels), 1024):
            q = self.pixels[i : i + 1024]
            basis = np.c_[_cubic(cdist(q, self.p)), np.ones(len(q)), q]
            blocks.append(basis @ solution)
        fields = np.concatenate(blocks).reshape(*self.shape, values.shape[1])
        return self.mean_weights, fields


@lru_cache(maxsize=2)
def _cached_field(extent, vres, ss, shape):
    cfg = SimpleNamespace(volume_um=extent, vres=vres, ss=ss)
    return CubicField(cfg, fixed_locations(cfg, 32), shape)


def clear_geometry_cache():
    """Clear the two-entry geometry cache, e.g. for cold-geometry timing."""
    _cached_field.cache_clear()


def assemble(config, out, shape, *, mask_mode="matlab_v4", z_positions=None, cut=None):
    """Assemble native outputs. Legacy cubic aggregation is diagnostic-only.

    Full-field cubic imaging lives in maomi.enhanced and never averages PSFs.
    """
    points = out["points_um"]
    psfs = out["psfs"]
    energies = psfs.sum((1, 2, 3), dtype=np.float64)
    values = [energies]
    tails = bool(out["tails_computed"])
    if tails:
        values.extend(out[k]["weight"] for k in ("top", "bottom"))
    values = np.stack(values, axis=1)
    cache_hit = False
    if mask_mode == "cubic":
        if len(points) == 1024 and np.array_equal(points, fixed_locations(config, 32)):
            before = _cached_field.cache_info().hits
            model = _cached_field(tuple(config.volume_um[:2]), config.vres, config.ss, tuple(shape))
            cache_hit = _cached_field.cache_info().hits > before
        else:
            model = CubicField(config, points, shape)
        weights, masks = model.reconstruct(values)
    elif mask_mode == "matlab_v4":
        axes = [(np.arange(k) + 0.5) for k in shape]
        q = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
        masks = v4_interpolate(points * config.vres, values, q)
        weights = np.full(len(points), 1 / len(points))
    else:
        raise ValueError("mask_mode must be cubic or matlab_v4")
    mask = masks[..., 0]
    mean = float(mask.mean())
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Nonpositive or invalid mean excitation energy")
    result = {
        "psf": (weights @ np.abs(psfs).reshape(len(points), -1)).reshape(psfs.shape[1:]),
        "psf_raw": (weights @ psfs.reshape(len(points), -1)).reshape(psfs.shape[1:]),
        "psf_3d": psfs,
        "points_um": points,
        "energies": energies,
        "mask_raw": mask,
        "mask": mask / mean,
        "mask_mean": mean,
        "mask_mode": mask_mode,
        "geometry_cache_hit": cache_hit,
        "tails_computed": tails,
        "unique_optical_paths": len(points),
        "negative_resampling_mass_fraction": float(
            -np.minimum(psfs, 0).sum(dtype=np.float64)
            / max(np.abs(psfs).sum(dtype=np.float64), 1e-30)
        ),
    }
    if tails:
        for j, (native, name) in enumerate([("top", "psfT"), ("bottom", "psfB")]):
            data = out[native]
            profile = data["profile"].sum(0, dtype=np.float64)
            actual_z = z_positions[0][cut:] if j == 0 else z_positions[2]
            actual_z = actual_z - actual_z[0]
            axial = np.arange(round(actual_z[-1] * 1e6 * config.vres) + 1) / config.vres
            interp = np.interp(axial, actual_z * 1e6, profile)
            tm = masks[..., j + 1]
            tm_mean = float(tm.mean())
            result[name] = {
                "convmask": data["convmask"].mean(0),
                "weight": float(data["weight"].mean()),
                "psfZ": interp / interp.mean() if interp.mean() > 0 else interp,
                "mask_raw": tm,
                "mask": tm / tm_mean if tm_mean > 0 else tm,
                "mask_mean": tm_mean,
                "weights_per_point": data["weight"],
                "profiles_per_point": data["profile"],
            }
    return result


def run_fixed(
    config,
    volume,
    *,
    vessels=None,
    raw_fields=None,
    scatter=True,
    custom_phase=None,
    phase_units="radians",
    phase_aperture="objective",
    input_pupil=None,
    beam_weights=None,
    tails=True,
    batch_size=32,
    count_per_axis=65,
    mask_mode="matlab_v4",
    device="cuda",
    rana=False,
):
    """MATLAB v4 masks and arithmetic PSF averaging; 65x65 sites per beam by default.

    Tails are returned as descriptors, not added as functional-movie background.
    Collection absorption is applied to the image when vessels are supplied.
    CPU/windowed reference optics remain in maomi.optics.run_optics/propagate.
    """
    if rana is not False and rana is not None:
        options = locals().copy()
        options['rana'] = False
        from .rana_addon import attach_rana, RanaAddon
        from .rana_ui import RanaStatus
        overall = time.perf_counter()
        with RanaStatus(rana.status_window if isinstance(rana,RanaAddon) else True):
            native = run_fixed(**options)
            start = time.perf_counter()
            result = attach_rana(native,rana,config=config,volume=volume,device=device)
            result['timing_seconds'] = dict(native.get('timing_seconds',{}), rana=time.perf_counter()-start)
            result['native_total_seconds'] = native['total_seconds']
            result['total_seconds'] = time.perf_counter()-overall
            return result
    if mask_mode != "matlab_v4":
        raise ValueError("run_fixed requires matlab_v4; use maomi.enhanced.CubicFullFieldRenderer "
                         "with a complete custom FullFieldPSF for cubic imaging")
    points = fixed_locations(config, count_per_axis)
    if isinstance(input_pupil, dict):
        if custom_phase is not None:
            raise ValueError("Compose channel pupil phases explicitly")
        if not input_pupil:
            raise ValueError("At least one beam required")
        weights = {k: 1.0 for k in input_pupil} if beam_weights is None else dict(beam_weights)
        if (
            set(weights) != set(input_pupil)
            or not all(np.isfinite(w) and w >= 0 for w in weights.values())
            or sum(weights.values()) <= 0
        ):
            raise ValueError("Nonnegative detected-intensity weight per beam required")
        start = time.perf_counter()
        raw = (
            prepare_fields(config, vessels, scatter=scatter, device="cpu")
            if raw_fields is None
            else raw_fields
        )
        channels = {
            name: run_fixed(
                config,
                volume,
                vessels=vessels,
                raw_fields=raw,
                input_pupil=field,
                tails=tails,
                batch_size=batch_size,
                count_per_axis=count_per_axis,
                mask_mode=mask_mode,
                device=device,
            )
            for name, field in input_pupil.items()
        }
        return {
            "channels": channels,
            "beam_weights": weights,
            "image": sum(weights[k] * v["image"] for k, v in channels.items()),
            "points_um": points,
            "unique_optical_paths": len(points) * len(channels),
            "unique_optical_positions": len(points),
            "total_seconds": time.perf_counter() - start,
            "settings": {
                "combination": "incoherent detected-intensity sum",
                "scaling": config.scaling,
                "paths_per_position": len(channels),
                "count_per_axis": count_per_axis,
                "mask_mode": mask_mode,
            },
        }
    if beam_weights is not None:
        raise ValueError("beam_weights requires a mapping of pupils")
    if not str(device).startswith("cuda"):
        raise ValueError(
            "Fixed fast pipeline requires CUDA; use maomi.optics for CPU reference calculations"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch and an NVIDIA GPU are required for run_fixed")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if phase_units not in ("radians", "waves") or phase_aperture not in ("objective", "excitation"):
        raise ValueError("Invalid pupil phase units/aperture")
    if input_pupil is not None and custom_phase is not None:
        raise ValueError("Supply either input_pupil or custom_phase; compose the field explicitly")
    volume = np.asarray(volume, np.float32)
    expected = tuple(np.round(np.asarray(config.volume_um) * config.vres).astype(int))
    if volume.shape != expected or not np.isfinite(volume).all():
        raise ValueError(f"Expected finite volume of shape {expected}")
    started = time.perf_counter()
    timings = {}
    supplied = raw_fields is not None
    t = time.perf_counter()
    raw = (
        prepare_fields(config, vessels, scatter=scatter, device="cpu")
        if raw_fields is None
        else raw_fields
    )
    expected_fields = field_shapes(config)
    if len(raw) != 3 or any(tuple(a.shape) != tuple(s) for a, s in zip(raw, expected_fields)):
        raise ValueError(f"Raw A/B/C fields must have shapes {expected_fields}")
    if any(
        torch.is_complex(torch.as_tensor(a)) or not bool(torch.isfinite(torch.as_tensor(a)).all())
        for a in raw
    ):
        raise ValueError("Raw refractive-index fields must be finite and real")
    col = (
        collection_mask(vessels, config)
        if vessels is not None
        else np.ones(volume.shape[:2], np.float32)
    )
    timings["fields_and_collection"] = time.perf_counter() - t
    # Imports are lazy: the CPU reference package has no CuPy dependency.
    try:
        import cupy as cp
        from maomi.backends.cuda.batched import OpticalPlan
    except ImportError as exc:
        raise RuntimeError('Install the CUDA extra: pip install ".[cuda]"') from exc
    with torch.cuda.device(device), cp.cuda.Device(torch.cuda.current_device()):
        t = time.perf_counter()
        u = (
            pupil(
                config,
                custom_phase=custom_phase,
                phase_units=phase_units,
                phase_aperture=phase_aperture,
                device=device,
            )
            if input_pupil is None
            else torch.as_tensor(input_pupil, dtype=torch.complex64, device=device)
        )
        expected_n = int(
            round(
                (config.parameters()[0].vasc_sz[0] - config.volume_um[0]) * config.vres * config.ss
            )
        )
        if (
            tuple(u.shape) != (expected_n, expected_n)
            or not bool(torch.isfinite(u).all())
            or not bool(torch.any(u.abs() > 0))
        ):
            raise ValueError(
                "Input pupil must be a finite nonzero complex field on the optical grid"
            )
        plan = OpticalPlan(config, raw, uin=u, device=device)
        torch.cuda.synchronize()
        timings["plan"] = time.perf_counter() - t
        t = time.perf_counter()
        outputs = plan.evaluate(points, batch_size=batch_size, tails=tails)
        timings["propagation"] = time.perf_counter() - t
        t = time.perf_counter()
        result = assemble(
            config,
            outputs,
            volume.shape[:2],
            mask_mode=mask_mode,
            z_positions=plan.z_positions,
            cut=plan.cut,
        )
        result["colmask"] = col
        timings["aggregation"] = time.perf_counter() - t
        t = time.perf_counter()
        result["image"] = (
            zstack(
                volume,
                result["psf"].astype(np.float32),
                mask=result["mask"].astype(np.float32),
                collection_mask=col,
                device=device,
            )
            .cpu()
            .numpy()
        )
        timings["static_scan"] = time.perf_counter() - t
    result["timing_seconds"] = timings
    result["total_seconds"] = time.perf_counter() - started
    result["settings"] = {
        "seed": config.seed,
        "scaling": config.scaling,
        "fields": "provided" if supplied else "generated",
        "scatter": None if supplied else bool(scatter),
        "phase_units": phase_units,
        "phase_aperture": phase_aperture,
        "input_pupil": "provided" if input_pupil is not None else "gaussian",
        "collection_absorption": vessels is not None,
        "tail_background_added_to_image": False,
        "count_per_axis": count_per_axis,
        "adaptive": False,
        "low_order_subtraction": False,
        "mask_mode": mask_mode,
        "psf_average": "arithmetic mean of absolute per-point PSFs",
    }
    return result
