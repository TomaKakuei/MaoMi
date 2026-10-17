"""RayOptics RayGrid -> sampled scalar PSF. RayOptics is an optional dependency.

Tested with RayOptics 0.9.5. Near-axis, central-wavelength, finite-conjugate
models only: paraxial exit-pupil mapping is not a high-NA vector model.
"""
from importlib.metadata import version
import numpy as np
from ..psf import SampledPSF, PSFField, pupil_to_intensity, resample_psf


def rayoptics_psf(model, *, num_rays=33, fft_size=256, field_index=0,
                 magnification=None, focus_offset=0.):
    """Trace the model's central wavelength and return an object-space PSF.

    focus_offset uses the RayOptics model's native length unit. Pixel pitch is
    derived from the actual traced grid spacing, not an assumed [-1, 1] grid.
    An explicit magnification is required for infinite-conjugate models.
    """
    from rayoptics.raytr.analyses import RayGrid
    if int(num_rays) != num_rays or num_rays < 5:
        raise ValueError("num_rays must be an integer >= 5")
    if not np.isfinite(focus_offset):
        raise ValueError("focus_offset must be finite")
    model.update_model()
    g = RayGrid(model, f=field_index, num_rays=int(num_rays), foc=float(focus_offset),
                clip_rays=True, value_if_none=np.nan)
    x, y, opd = np.asarray(g.grid, dtype=float)
    # RayGrid arrays use X as axis 0 and Y as axis 1.
    ux = np.nanmedian(x, axis=1)
    uy = np.nanmedian(y, axis=0)
    deltas = []
    for coord in (ux, uy):
        steps = np.diff(coord)
        step = float(np.median(steps))
        if step <= 0 or not np.isfinite(step) or not np.allclose(steps, step, rtol=1e-6, atol=1e-12):
            raise ValueError("RayOptics returned a nonuniform or reversed pupil grid")
        deltas.append(step)
    if not np.allclose(x, ux[:, None], equal_nan=False) or not np.allclose(y, uy[None, :], equal_nan=False):
        raise ValueError("RayOptics pupil coordinates are not rectilinear XY")
    fod = model['analysis_results']['parax_data'].fod
    mag = abs(float(fod.m if magnification is None else magnification))
    if not np.isfinite(mag) or mag <= 0:
        raise ValueError("Specify a finite, positive lateral magnification")
    radius = abs(float(g.fld.ref_sphere[2]))
    pupil_radius = abs(float(fod.exp_radius))
    if not np.isfinite([radius, pupil_radius]).all() or min(radius, pupil_radius) <= 0:
        raise ValueError("Finite reference sphere and exit pupil are required")
    wavelength = float(g.wvl)
    # Vacuum wavelength converted to the image medium before Fraunhofer scaling.
    n_image = abs(float(fod.n_img))
    if not np.isfinite(n_image) or n_image <= 0:
        raise ValueError("Invalid image-space refractive index")
    unit_um = 1. / float(model.nm_to_sys_units(1000.))
    lambda_system = float(model.nm_to_sys_units(wavelength)) / n_image
    pitch = tuple(lambda_system * radius / (pupil_radius * fft_size * du) * unit_um / mag for du in deltas)
    psf = pupil_to_intensity(opd, fft_size=fft_size)
    return SampledPSF(psf, pitch, wavelength, {
        "provider": "RayOptics", "version": version("rayoptics"), "axes": "XY",
        "model": "scalar diffraction; paraxial exit pupil; central wavelength",
        "field_index": int(field_index), "num_rays": int(num_rays), "fft_size": int(fft_size),
        "valid_rays": int(np.isfinite(opd).sum()), "magnification": mag,
        "image_refractive_index": n_image, "focus_offset_model_units": float(focus_offset),
        "model_length_unit": str(model['system_spec'].dimensions),
        "reference_sphere_radius_model_units": radius, "exit_pupil_radius_model_units": pupil_radius,
        "pupil_grid_step": deltas, "opd_range_waves": [float(np.nanmin(opd)), float(np.nanmax(opd))],
        "normalization": "unit sum; throughput is specified separately in photoelectrons"
    })


def rayoptics_psf_field(model, x_um, y_um, field_indices, *, pixel_size_um, shape,
                       num_rays=33, fft_size=256, magnification=None, focus_offset=0.):
    """Trace configured RayOptics fields and explicitly resample to a common grid.

    field_indices[i,j] maps specimen position (x_um[i], y_um[j]) to an existing
    model field. The caller supplies this physical mapping; no field-of-view
    coordinates are inferred. Every trace retains its provider/resampling record.
    """
    indices = np.asarray(field_indices)
    if (indices.shape != (len(x_um), len(y_um)) or not np.isfinite(indices).all()
            or (indices < 0).any() or not np.equal(indices, np.floor(indices)).all()):
        raise ValueError("field_indices must be a nonnegative integer array matching the field grid")
    lookup = {(float(x), float(y)): int(indices[i, j])
              for i, x in enumerate(x_um) for j, y in enumerate(y_um)}
    cache = {}
    def provider(x, y):
        index = lookup[x, y]
        if index not in cache:
            p = rayoptics_psf(model, field_index=index, num_rays=num_rays, fft_size=fft_size,
                             magnification=magnification, focus_offset=focus_offset)
            cache[index] = resample_psf(p, pixel_size_um, shape)
        return cache[index]
    return PSFField.from_provider(provider, x_um, y_um)


def demo_lens(*, wavelength_nm=550., entrance_pupil_mm=8., curvature_mm=50.):
    """Air/glass/air finite-conjugate singlet with a constant glass index.

    A demonstration optical prescription, not a fitted microscope objective.
    """
    from rayoptics.optical.opticalmodel import OpticalModel
    from rayoptics.raytr.opticalspec import PupilSpec, FieldSpec, WvlSpec
    if not np.isfinite([wavelength_nm, entrance_pupil_mm, curvature_mm]).all() or min(wavelength_nm, entrance_pupil_mm, curvature_mm) <= 0:
        raise ValueError("Lens parameters must be positive and finite")
    model = OpticalModel()
    model.radius_mode = True
    seq, spec = model['seq_model'], model['optical_spec']
    spec['pupil'] = PupilSpec(spec, key=['object', 'epd'], value=entrance_pupil_mm)
    spec['fov'] = FieldSpec(spec, key=['object', 'height'], flds=[0.], value=1., is_relative=True)
    spec['wvls'] = WvlSpec([(wavelength_nm, 1.)], ref_wl=0)
    seq.gaps[0].thi = 100.
    seq.add_surface([curvature_mm, 4., 1.517], sd=8.)
    seq.set_stop()
    seq.add_surface([-curvature_mm, 100.], sd=8.)
    model.update_model()
    seq.gaps[-1].thi = float(model['analysis_results']['parax_data'].fod.img_dist)
    model.update_model()
    return model
