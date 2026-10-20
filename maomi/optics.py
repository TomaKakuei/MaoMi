"""Optical subset: explicit spatial samples, A/B/C paths, tails and collection.

Derived from Song & Charles' MIT NAOMi MATLAB optical routines (2016-2021),
and the preserved July 2026 local Python port. See LICENSES/NAOMi-MIT.txt.
This module does not implement MATLAB FM.fineSamp's interpolated wavefront pass.
Instead it evaluates explicitly requested full optical paths.
"""

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
from .numerics import fresnel, resize_xy, resize_xy_region, translate_xy, v4_interpolate

from . import _optical_math as legacy
from .excitation import excitation_order, axial_profile
from ._propagation import geometry, edge_apodization, center_slice as _center


@dataclass
class OpticalConfig:
    volume_um: tuple = (32.0, 32.0, 10.0)
    depth_um: float = 30.0
    vres: float = 1.0
    NA: float = 0.6
    objNA: float = 0.8
    n: float = 1.35
    wavelength_um: float = 0.92
    focal_length_mm: float = 4.5
    ss: int = 2
    psf_um: tuple = (8.0, 8.0, 10.0)
    prop_um: float = 5.0
    tail_um: float = 20.0
    n_diff: float = 0.02
    hemoabs: float = 0.0155194235
    propcrop: bool = True
    scatter_scales_um: tuple = (0.51, 1.56, 4.52, 14.78)
    scatter_weights: tuple = (0.57, 0.29, 0.19, 0.15)
    seed: int = 20260927
    scaling: str = "two-photon"
    temporal_length_um: float = 10.0
    zernike_weights: tuple = ()

    def parameters(self):
        excitation_order(self)
        if not np.isfinite(self.temporal_length_um) or self.temporal_length_um <= 0:
            raise ValueError("Positive temporal length required")
        p = legacy.PsfParams(
            NA=self.NA,
            objNA=self.objNA,
            n=self.n,
            lambda_exc=self.wavelength_um,
            obj_fl=self.focal_length_mm,
            ss=self.ss,
            psf_sz=np.array(self.psf_um),
            prop_sz=self.prop_um,
            taillength=self.tail_um,
            n_diff=self.n_diff,
            propcrop=self.propcrop,
            fastmask=False,
            zernikeWt=np.asarray(self.zernike_weights,dtype=float).ravel() if len(self.zernike_weights) else None,
        )
        v = legacy.VolParams(
            vol_sz=np.array(self.volume_um), vres=self.vres, vol_depth=self.depth_um, verbose=0
        )
        v.vasc_sz = (
            legacy.gaussian_beam_size(p, self.depth_um + self.volume_um[2] / 2)
            + v.vol_sz
            + np.array([0, 0, self.depth_um])
        )
        return v, p


def locations(volume_um, count_per_axis):
    """Explicit (X,Y) microns; single sample is FOV CENTER, never an implicit corner."""
    x = (
        np.linspace(0, volume_um[0], count_per_axis)
        if count_per_axis > 1
        else np.array([volume_um[0] / 2])
    )
    y = (
        np.linspace(0, volume_um[1], count_per_axis)
        if count_per_axis > 1
        else np.array([volume_um[1] / 2])
    )
    return np.stack(np.meshgrid(x, y, indexing="ij"), -1).reshape(-1, 2)


def pupil(
    config, *, custom_phase=None, phase_units="radians", phase_aperture="objective", device="cpu"
):
    """Phase has explicit units and pupil coordinates; no low-order subtraction."""
    v, p = config.parameters()
    if custom_phase is not None:
        p.zernikeWt = None  # MATLAB custom_phase replaces its default Zernike screen.
    u = legacy.generate_ba(v, p, device=torch.device(device))
    if custom_phase is None:
        return u
    if phase_units not in ("radians", "waves") or phase_aperture not in ("objective", "excitation"):
        raise ValueError("Explicit phase_units / phase_aperture required")
    h, w = u.shape
    fl = config.focal_length_mm / 1000
    d1 = max(legacy.gaussian_beam_size(p, fl * 1e6)) / 1e6 / min(h, w)
    rad = (
        np.tan(np.arcsin((config.objNA if phase_aperture == "objective" else config.NA) / config.n))
        * fl
    )
    y, x = torch.meshgrid(
        torch.arange(-h / 2, h / 2, device=device) * d1,
        torch.arange(-w / 2, w / 2, device=device) * d1,
        indexing="ij",
    )
    grid = torch.stack((x / rad, y / rad), -1)[None]
    phase = torch.as_tensor(custom_phase, dtype=u.real.dtype, device=device)[None, None]
    phase = F.grid_sample(phase, grid, mode="nearest", align_corners=True)[0, 0]
    return u * torch.exp(1j * phase * (2 * np.pi if phase_units == "waves" else 1))


def prepare_fields(config, vessels=None, *, device="cpu", scatter=True):
    """Create reproducible raw A/B/C refractive-index occupancy fields.

    vessels, when supplied, covers the padded vascular XY domain from z=0.
    Fixed Python RNG is reproducible within this backend; MATLAB's shuffled GP
    draws are not identical. Shared raw fields are required for exact comparisons.
    """
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
    xy = np.round(np.array(config.volume_um[:2]) * r).astype(int)
    shapes = [
        (xy[0] + n, xy[1] + n, int(np.ceil(za / pc))),
        (xy[0] + n2, xy[1] + n2, zb),
        (xy[0] + n, xy[1] + n, int(np.ceil(nc / pc))),
    ]
    raw = []
    # fork_rng preserves the caller's random stream, including CUDA state.
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        scales = (
            torch.tensor(config.scatter_scales_um, device=device, dtype=torch.float32)[
                :, None
            ].repeat(1, 3)
            * r
        )
        weights = torch.tensor(config.scatter_weights, device=device, dtype=torch.float32)
        for i, shape in enumerate(shapes):
            ls = scales.clone()
            scale = 1 if i == 1 else pc
            ls[:, 2] /= scale
            raw.append(
                legacy.masked_3DGP_test(shape, ls, scale, l_weights=weights, device=device)
                if scatter
                else torch.zeros(shape, device=device)
            )
    # Partial first/top and last/bottom slabs have proportionate optical thickness.
    if za % pc:
        raw[0][..., 0] *= (za % pc) / pc
    if nc % pc:
        raw[2][..., -1] *= (nc % pc) / pc
    if vessels is not None:
        if not hasattr(vessels,"shape"):
            vessels = np.asarray(vessels)
        required = (shapes[0][0], shapes[0][1], za + zb + nc)
        if any(a < b for a, b in zip(vessels.shape, required)):
            raise ValueError(f"Padded vessels need at least {required}, got {tuple(vessels.shape)}")
        def read(x,y,z):
            a = vessels[x,y,z]
            if isinstance(a,torch.Tensor):
                return a.to(device=device,dtype=torch.float32)
            return torch.as_tensor(np.asarray(a), device=device, dtype=torch.float32)
        xs,ys = slice(0,required[0]),slice(0,required[1])
        chunks = []
        end = za % pc or pc
        start = 0
        while start < za:
            chunks.append(read(xs,ys,slice(start,end)).sum(-1))
            start = end
            end = min(za, end + pc)
        occA = torch.stack(chunks, -1)
        off = (n - n2) // 2
        occB = read(slice(off,off+shapes[1][0]),slice(off,off+shapes[1][1]),slice(za,za+zb))
        occC = torch.stack([read(xs,ys,slice(z,min(z+pc,za+zb+nc))).sum(-1)
                            for z in range(za+zb,za+zb+nc,pc)],-1)
        for i, occ in enumerate((occA, occB, occC)):
            raw[i] = occ + raw[i] * (1 - occ / (1 if i == 1 else pc))
    return raw


def propagate(
    config,
    raw_fields,
    points_um,
    *,
    uin=None,
    device="cpu",
    save_full=False,
    phase_mode="auto",
    phase_memory_limit_mb=256,
):
    """Return absolute PSFs, raw/relative masks, tails, and exact sample coordinates.

    Small XY fields are resized before windowing. Large fields use an exactly
    equivalent global-coordinate interpolation halo, retaining raw fields on CPU.
    The optimized trajectory saves only output crops; save_full=True is a
    numerically equivalent reference with the original larger trajectory.
    """
    r, ss = config.vres, config.ss
    g = geometry(config, raw_fields)
    n, n2, d1, d2, fl, wvl, nr = g.n, g.n2, g.d1, g.d2, g.fl, g.wvl, g.nr
    u = pupil(config, device=device) if uin is None else torch.as_tensor(uin, device=device)
    sg, sg2 = [edge_apodization(k, device, u.real.dtype) for k in (n, n2)]
    base = fresnel(
        u,
        wvl,
        [d1, d2],
        [0, np.float32(fl - (config.depth_um + config.volume_um[2] / 2) * 1e-6)],
        torch.stack((sg, sg), -1),
        nr,
    )
    base = base / torch.linalg.vector_norm(base)
    phase_scale = config.n_diff * 2 * np.pi / (config.wavelength_um * r)
    estimate = sum(np.prod(a.shape) for a in raw_fields) * ss**2 * 8
    if phase_mode == "auto":
        phase_mode = "window" if estimate > phase_memory_limit_mb * 1024**2 else "full"
    if phase_mode not in ("full", "window"):
        raise ValueError("phase_mode must be auto/full/window")
    phase = (
        [
            torch.exp(
                1j
                * resize_xy(torch.as_tensor(a, device=device, dtype=u.real.dtype), ss)
                * phase_scale
            )
            for a in raw_fields
        ]
        if phase_mode == "full"
        else None
    )
    za_pos, zb_pos, zc_pos = g.z
    cut, c, fc = g.cut, g.crop, g.focal_crop
    outputs = []
    tails = [[], []]
    weights = [[], []]
    profiles = [[], []]
    pts = np.asarray(points_um, float)
    if np.any(pts < 0) or np.any(pts > np.array(config.volume_um[:2])):
        raise ValueError("PSF coordinates outside FOV")
    for xy in pts:
        offset = xy * r * ss
        if not np.allclose(offset, np.round(offset)):
            raise ValueError("Locations must land on the supersampled grid")
        x, y = np.round(offset).astype(int)
        if phase is not None:
            pa = phase[0][x : x + n, y : y + n] * sg[..., None]
            pb = phase[1][x : x + n2, y : y + n2] * sg2[..., None]
            pcfield = phase[2][x : x + n, y : y + n] * sg[..., None]
        else:
            chunks = [
                torch.exp(
                    1j
                    * resize_xy_region(
                        torch.as_tensor(a, dtype=u.real.dtype), ss, (x, y), (m, m), device=device
                    )
                    * phase_scale
                )
                for a, m in zip(raw_fields, (n, n2, n))
            ]
            pa = chunks[0] * sg[..., None]
            pb = chunks[1] * sg2[..., None]
            pcfield = chunks[2] * sg[..., None]
        ua, top = fresnel(
            base,
            wvl,
            np.full(len(za_pos), d2),
            za_pos,
            pa,
            nr,
            return_all=True,
            crop=None if save_full else (c, c),
            save_from=0 if save_full else cut,
        )
        ub, focus = fresnel(
            ua[c, c],
            wvl,
            np.full(len(zb_pos), d2),
            zb_pos,
            pb,
            nr,
            return_all=True,
            crop=None if save_full else fc,
        )
        padded = F.pad(ub, ((n - n2) // 2,) * 4)
        _, bottom = fresnel(
            padded,
            wvl,
            np.full(len(zc_pos), d2),
            zc_pos,
            pcfield,
            nr,
            return_all=True,
            crop=None if save_full else (c, c),
        )
        if save_full:
            top = top[c, c, cut:]
            focus = focus[fc[0], fc[1]]
            bottom = bottom[c, c]
        intensity = ss**2 * focus[..., :-1].abs() ** (2 * excitation_order(config))
        intensity *= torch.as_tensor(axial_profile(config, intensity.shape[-1]), device=device)
        final = (
            ss**2
            * resize_xy(translate_xy(intensity, ss / 2 - 0.5), 1 / ss)
            * (r * (1e6 * wvl) ** 1.5)
            / (np.pi * nr)
        )
        outputs.append(final.cpu().numpy())
        for j, field in enumerate((top, bottom)):
            intensity = ss**2 * field.abs() ** (2 * excitation_order(config))
            intensity *= torch.as_tensor(
                axial_profile(config, field.shape[-1], "top" if j == 0 else "bottom"), device=device
            )
            profiles[j].append(intensity.sum((0, 1)).cpu().numpy())
            intensity[..., 0] *= 0.5
            intensity[..., -1] *= 0.5
            weights[j].append(float(intensity.sum().cpu()) * config.prop_um)
            tails[j].append(
                (ss**2 * resize_xy(translate_xy(intensity.sum(-1), ss / 2 - 0.5), 1 / ss))
                .cpu()
                .numpy()
            )
    psfs = np.stack(outputs)
    energies = psfs.sum((1, 2, 3), dtype=np.float64)
    xyshape = np.round(np.array(config.volume_um[:2]) * r).astype(int)
    q = np.stack(
        np.meshgrid(np.arange(xyshape[0]) + 0.5, np.arange(xyshape[1]) + 0.5, indexing="ij"), -1
    )
    mask = v4_interpolate(pts * r, energies, q)
    mean = float(mask.mean())
    result = {
        "psf_3d": psfs,
        "psf_raw": psfs.mean(0),
        "psf": np.abs(psfs).mean(0),
        "mask_raw": mask,
        "mask_mean": mean,
        "mask": mask / mean if mean > 0 else mask,
        "points_um": pts,
        "energies": energies,
        "mask_mode": "explicit_full_paths_v4",
        "scattering_replay": "provided_raw_fields",
        "phase_mode": phase_mode,
        "estimated_full_phase_bytes": int(estimate),
        "negative_resampling_mass_fraction": float(
            -np.minimum(psfs, 0).sum(dtype=np.float64) / np.abs(psfs).sum(dtype=np.float64)
        ),
    }
    for j, name in enumerate(("psfT", "psfB")):
        profile = np.stack(profiles[j]).sum(0)
        actual_z = (za_pos[cut:] - za_pos[cut]) if j == 0 else zc_pos
        out_z = np.arange(round(actual_z[-1] * 1e6 * r) + 1) / r
        interp = np.interp(out_z, actual_z * 1e6, profile)
        tail_mask = v4_interpolate(pts * r, weights[j], q)
        tail_mean = float(tail_mask.mean())
        result[name] = {
            "convmask": np.mean(tails[j], 0),
            "weight": float(np.mean(weights[j])),
            "psfZ": interp / interp.mean() if interp.mean() > 0 else interp,
            "mask_raw": tail_mask,
            "mask_mean": tail_mean,
            "mask": tail_mask / tail_mean if tail_mean > 0 else tail_mask,
        }
    return result


def collection_mask(vessels, config):
    """Port of NAOMi's depth-dependent collection disk and hemoglobin absorption."""
    a = vessels if hasattr(vessels,"shape") else np.asarray(vessels)
    r = config.vres
    pc = int(round(config.prop_um * r))
    total = np.zeros(a.shape[:2], np.float32)
    for start in range(0, a.shape[-1], pc):
        depth = (start / pc + 0.5) * config.prop_um
        radius = (
            r
            * np.tan(np.arcsin(config.objNA / config.n))
            * (config.depth_um + config.volume_um[2] / 2 - depth)
        )
        if radius <= 0:
            continue
        # Match original full-grid disk then tight bounding-box autocrop.
        x = np.arange(-a.shape[0] / 2, a.shape[0] / 2)[:, None]
        y = np.arange(-a.shape[1] / 2, a.shape[1] / 2)[None, :]
        disk = (x * x + y * y <= radius * radius).astype(np.float32)
        nz = np.argwhere(disk)
        if len(nz) == 0:
            continue
        lo = nz.min(0)
        hi = nz.max(0) + 1
        disk = disk[lo[0] : hi[0], lo[1] : hi[1]]
        disk /= disk.sum()
        # Disk is odd-sized, so scipy's same crop equals MATLAB's.
        total += fftconvolve(np.asarray(a[..., start : start + pc],np.float32).sum(-1), disk, mode="same")
    shape = np.round(np.array(config.volume_um[:2]) * r).astype(int)
    offsets = (np.array(total.shape) - shape) // 2
    total = total[offsets[0] : offsets[0] + shape[0], offsets[1] : offsets[1] + shape[1]]
    return 10 ** (-total / r * config.hemoabs)


def run_optics(
    config,
    points_um,
    *,
    vessels=None,
    scatter=True,
    custom_phase=None,
    phase_units="radians",
    phase_aperture="objective",
    device="cpu",
    rana=False,
):
    """Convenience pipeline with explicit physics and no stage-dependent factors.

    Vessels cover the padded domain described by prepare_fields. An absent
    vessel volume means zero vessel absorption; Gaussian scattering remains an
    independent, explicitly enabled model component.
    """
    fields = prepare_fields(config, vessels, scatter=scatter, device="cpu")
    u = pupil(
        config,
        custom_phase=custom_phase,
        phase_units=phase_units,
        phase_aperture=phase_aperture,
        device=device,
    )
    out = propagate(config, fields, points_um, uin=u, device=device)
    out["colmask"] = (
        collection_mask(vessels, config) if vessels is not None else np.ones_like(out["mask"])
    )
    out["settings"] = {
        "seed": config.seed,
        "scatter": scatter,
        "phase_units": phase_units,
        "phase_aperture": phase_aperture,
        "low_order_subtraction": False,
    }
    if rana is not False and rana is not None:
        from .rana_addon import attach_rana
        out = attach_rana(out,rana,config=config,device=device)
    return out
