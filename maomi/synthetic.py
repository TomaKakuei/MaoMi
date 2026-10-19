"""Small procedural-cell and paired-photoelectron-data prototype.

Arrays use XY axes. These stylized thin-section cells require no photographs;
they are not calibrated biological cell models or a complete 3-D cell editor.
"""
import numpy as np
from .imaging import render_image


def procedural_cells(shape, pixel_size_um, cells):
    """Render elliptical soma, nuclear depression and optional tapered branches.

    Cell dictionaries contain center_um, radii_um, angle_deg, brightness,
    nucleus_fraction (0..1), and branches (each a polyline in world micrometres).
    Returns the additive fluorescence density and per-cell components; overlap
    is retained rather than assigning an arbitrary single instance ID.
    """
    if len(shape) != 2 or any(int(v) != v or v < 2 for v in shape):
        raise ValueError("shape must contain two integer dimensions >= 2")
    pitch = np.asarray(pixel_size_um, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
        raise ValueError("Specify positive XY pixel sizes in micrometres")
    x, y = np.meshgrid((np.arange(shape[0])+.5)*pitch[0],
                       (np.arange(shape[1])+.5)*pitch[1], indexing='ij')
    components = []
    for cell in cells:
        center = np.asarray(cell['center_um'], float)
        radii = np.asarray(cell['radii_um'], float)
        angle = float(cell.get('angle_deg', 0.))
        bright = float(cell.get('brightness', 1.))
        nucleus = float(cell.get('nucleus_fraction', .35))
        if center.shape != (2,) or radii.shape != (2,) or not np.isfinite([*center, *radii, angle, bright, nucleus]).all() or min(radii) <= 0 or bright < 0 or not 0 <= nucleus <= 1:
            raise ValueError("Invalid cell geometry, brightness or nucleus fraction")
        a = np.deg2rad(angle)
        dx, dy = x-center[0], y-center[1]
        u, v = np.cos(a)*dx+np.sin(a)*dy, -np.sin(a)*dx+np.cos(a)*dy
        q = (u/radii[0])**2 + (v/radii[1])**2
        # Smooth boundaries reduce aliasing; geometry remains in physical units.
        soma = np.clip((1.08-q)/.16, 0., 1.) * (1.-nucleus*np.exp(-q/.15))
        width = float(cell.get('branch_width_um', 1.))
        if not np.isfinite(width) or width <= 0:
            raise ValueError("branch_width_um must be positive")
        for branch in cell.get('branches', []):
            points = np.asarray(branch, float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
                raise ValueError("Each branch must be a finite XY polyline")
            for k, (p, r) in enumerate(zip(points[:-1], points[1:])):
                delta = r-p
                length2 = float(delta @ delta)
                if length2 == 0:
                    raise ValueError("Branch segments must have nonzero length")
                t = np.clip(((x-p[0])*delta[0]+(y-p[1])*delta[1])/length2, 0., 1.)
                d2 = (x-p[0]-t*delta[0])**2 + (y-p[1]-t*delta[1])**2
                local_width = width*(1.-.45*(k+t)/(len(points)-1))
                soma = np.maximum(soma, .6*np.exp(-.5*d2/local_width**2))
        components.append(bright*soma)
    if not components:
        raise ValueError("Supply at least one cell")
    components = np.stack(components)
    return components.sum(axis=0), components


def expected_measurement(density, psf, *, pixel_size_um, photons_per_unit=32., background_e=2.,
                         origin_um=(0., 0.)):
    """MaoMi convolution followed by an explicit photon-budget calibration.

    Despite the convenient photons_per_unit name, the budget is *detected*
    photoelectrons per unit density; throughput and quantum efficiency must be
    applied before choosing it. Zero padding permits light to leave the FOV.
    """
    a = np.asarray(density, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Density must be a finite, nonnegative XY image")
    if not np.allclose(pixel_size_um, psf.pixel_size_um, rtol=1e-10, atol=1e-12):
        raise ValueError("Scene and specimen-plane PSF sampling must match; no implicit resampling")
    if not np.isfinite([photons_per_unit, background_e]).all() or min(photons_per_unit, background_e) < 0:
        raise ValueError("Photon budget and background must be finite and nonnegative")
    image = render_image(a, psf, pixel_size_um=pixel_size_um, origin_um=origin_um)
    # Round-off can create tiny negative FFT residuals; physical density cannot.
    if image.min() < -1e-10*max(1., float(a.max())):
        raise ValueError("Convolution produced a material negative intensity")
    return photons_per_unit*np.maximum(image, 0.) + background_e


def noisy_measurement(clean_e, *, rng, read_sigma_e=1.):
    """Poisson photoelectrons plus zero-mean Gaussian read noise, without clipping."""
    a = np.asarray(clean_e, dtype=float)
    if not np.isfinite(a).all() or (a < 0).any() or not np.isfinite(read_sigma_e) or read_sigma_e < 0:
        raise ValueError("Invalid expected counts or read-noise standard deviation")
    return rng.poisson(a).astype(float) + rng.normal(0., read_sigma_e, size=a.shape)
