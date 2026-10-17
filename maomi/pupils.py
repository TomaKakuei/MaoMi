"""Single and independent Vtwins pupils matching the identified MATLAB cores."""

import numpy as np
from .optics import legacy


def make_pupil(config, *, kind="gaussian", length_um=60.0, profile="gaussian", cylscale=(1.0, 1.0)):
    x, y, rad, objrad, k, fl, nr = _pupil_grid(config)
    rho2 = x * x + y * y
    phase = np.exp(np.complex64(-1j) * k / (2 * fl) * rho2)
    if kind == "gaussian":
        amp = np.exp(-rho2 / (rad * rad)) * (rho2 < objrad * objrad)
    elif kind == "cylindrical":
        scale = np.asarray(cylscale, float)
        if scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Positive two-axis cylindrical scale required")
        # Original generateCylindricalBA ultimately masks the Gaussian pupil.
        scaled = (x / scale[0]) ** 2 + (y / scale[1]) ** 2
        support = (np.exp(-scaled / (rad * rad)) * (scaled < objrad * objrad)) > 0
        amp = np.exp(-rho2 / (rad * rad)) * (rho2 < objrad * objrad) * support
    elif kind == "bessel":
        if not np.isfinite(length_um) or length_um <= 0:
            raise ValueError("Positive Bessel length required")
        width = np.float32(
            1.4e-10 * np.tan(np.arcsin(np.float32(config.NA) / nr)) / (fl * 1e-6 * 2 * length_um)
        )
        dr = np.sqrt(rho2) - rad
        amp = _bessel_amplitude(dr, width, profile)
    else:
        raise ValueError("Supported single-field kinds: gaussian, bessel, cylindrical")
    field = np.asarray(amp * phase, np.complex64)
    if not np.isfinite(field).all() or not np.any(field):
        raise ValueError("Pupil has no finite nonzero energy on this grid")
    return field


def _pupil_grid(config):
    v, p = config.parameters()
    fl = np.float32(config.focal_length_mm / 1000)
    d2 = np.float32(1e-6 / config.vres / config.ss)
    n = np.float32(1e-6 * (v.vasc_sz[:2] - v.vol_sz[:2]) / d2)
    d1 = np.float32(max(legacy.gaussian_beam_size(p, float(fl * 1e6))) / 1e6 / min(n))
    nr = np.float32(config.n)
    rad = np.float32(np.tan(np.arcsin(np.float32(config.NA) / nr)) * fl)
    objrad = np.float32(np.tan(np.arcsin(np.float32(config.objNA) / nr)) * fl)
    k = np.float32(2 * nr * np.pi / np.float32(config.wavelength_um * 1e-6))
    x, y = np.meshgrid(
        np.arange(-n[0] / 2, n[0] / 2, dtype=np.float32) * d1,
        np.arange(-n[1] / 2, n[1] / 2, dtype=np.float32) * d1,
    )
    return x, y, rad, objrad, k, fl, nr


def make_vtwins_pupils(
    config,
    *,
    beam_na=0.12,
    theta_deg=20.0,
    separation_um=6.0,
    kind="gaussian",
    length_um=50.0,
    profile="gaussian",
    clip_to_objective=True
):
    """Separate left/right incident fields; never a coherent field sum.

    Parameters reproduce generateVtwinsBA. clip_to_objective=True corresponds
    to its default-profile Bessel branch; the explicit-bprof branch omits it.
    """
    if (
        not np.isfinite([beam_na, theta_deg, separation_um, length_um]).all()
        or not 0 < beam_na < config.n
        or abs(theta_deg) >= 180
        or length_um <= 0
    ):
        raise ValueError("Invalid Vtwins beam parameters")
    x, y, _, aperture, k, fl, nr = _pupil_grid(config)
    radius = np.float32(np.tan(np.arcsin(np.float32(beam_na) / nr)) * fl)
    displacement = np.float32(np.tan(np.deg2rad(theta_deg / 2))) * fl
    rho2 = x * x + y * y
    focus = np.exp(np.complex64(-1j) * k / (2 * fl) * rho2)
    result = {}
    for label, sign in [("left", 1), ("right", -1)]:
        xx = x - sign * displacement
        if kind == "gaussian":
            amplitude = np.exp(-(xx * xx + y * y) / (radius * radius)) * (
                rho2 < aperture * aperture
            )
        elif kind == "bessel":
            width = np.float32(
                1.4e-10 * np.tan(np.arcsin(np.float32(config.NA) / nr)) / (fl * 1e-6 * length_um)
            )
            dr = np.sqrt(xx * xx + y * y) - radius
            amplitude = _bessel_amplitude(dr, width, profile)
            if clip_to_objective:
                amplitude *= rho2 < aperture * aperture
        else:
            raise ValueError("Vtwins kind must be gaussian or bessel")
        phase = (
            np.float32(config.wavelength_um * 1e-6 * separation_um / 8)
            * np.float32(2)
            * x
            / aperture
        ).astype(np.float32)
        phase -= phase.mean(dtype=np.float32)
        result[label] = np.asarray(
            amplitude * focus * np.exp(np.complex64(1j) * k * sign * phase), np.complex64
        )
    return result


def _bessel_amplitude(dr, width, profile):
    if profile == "gaussian":
        width = width / (2 * np.sqrt(np.float32(np.log(2))))
        return np.exp(-((dr / width) ** 2))
    if profile == "uniform":
        return (dr < width / 2) & (dr > -width / 2)
    if profile in ("posax", "negax"):
        width = width / 1.133
        z = (dr if profile == "posax" else -dr) + width / np.sqrt(np.float32(2))
        return z * np.exp(-((z / width) ** 2)) * (z > 0)
    raise ValueError("Unknown Bessel profile")
