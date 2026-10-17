# Public API

## Platform extension API

`maomi.psf.SampledPSF` (also exported through the existing `maomi.integrations` path) is the provider-neutral 2-D intensity contract:
unit sum, XY axes, origin at `shape // 2`, specimen-plane pixel pitch in
micrometres, wavelength in nanometres, and JSON provenance. Use `.normalized()`
for explicit normalization and `.save()` / `.load()` for portable NPZ storage.
An intensity PSF is not a complex optical field.

`maomi.integrations.rayoptics.rayoptics_psf(model, num_rays=33, fft_size=256,
field_index=0, magnification=None, focus_offset=0.)` traces the central wavelength
of a RayOptics model. It assumes a near-axis, scalar, paraxial exit-pupil model;
the returned PSF is sampled in the specimen plane. Focus offsets use the lens
model's units. RayOptics is an optional dependency, tested at version 0.9.5.

`maomi.synthetic.procedural_cells(shape, pixel_size_um, cells)` returns summed
XY fluorescence density and per-cell components from an editable cell recipe.
`expected_measurement(density, psf, pixel_size_um=..., photons_per_unit=32.,
background_e=2.)` uses the public PSF renderer and returns expected
detected photoelectrons. `noisy_measurement(clean_e, rng=..., read_sigma_e=1.)`
adds Poisson and Gaussian noise without clipping. The clean expected image is
the denoising target; the latent density is a different reconstruction target.

See the [runnable integration tutorial](PLATFORM_EXAMPLE.md) and
[platform roadmap](TECHNICAL_REPORT.md#4-whats-next). These new modules are separate from
the native 3-D engine.

### Spatially varying PSFs

`maomi.psf.PSFField.from_provider(provider, x_um, y_um)` samples a callable returning
`SampledPSF` on a physical source grid. `.save()` and `.load()` store the field.
`maomi.imaging.PSFRenderer(psf, shape, pixel_size_um=..., origin_um=(0,0))`
prepares a reusable CPU operator; call it with each density image. `render_image`
is the one-shot form. `expected_measurement` also accepts a field and an origin.

`maomi.integrations.rayoptics.rayoptics_psf_field(model, x_um, y_um, field_indices,
pixel_size_um=..., shape=..., ...)` explicitly maps configured model fields to
specimen positions and resamples them onto a common intensity grid. A full
[contract, equation and runnable examples](PSF_INTERFACE.md) explains source
weighting, physical sampling, boundary behavior and the 2-D scope.

## Existing anatomy, optics and movie API

All spatial arrays use X,Y,Z; movies use X,Y,time. Sparse tissue component indices
use one-based MATLAB column-major indexing, as documented by `scan_volume`.

```python
from maomi import OpticalConfig, run_fixed, make_vtwins_pupils
from maomi.movie import scan_volume

cfg = OpticalConfig(volume_um=tuple(volume.shape), depth_um=10, tail_um=5)
optics = run_fixed(cfg, volume, vessels=vessels)
movie = scan_volume(neurons, optics, activity, scan, add_noise=False)
```

`vessels` must cover the padded domain from depth zero. `cfg.parameters()[0].vasc_sz`
exposes that domain in micrometres. `raw_fields=[A,B,C]` optionally supplies the
refractive-index fields for exact cross-backend comparisons.

Excitation uses `cfg.scaling`: `one-photon`, `two-photon` (default),
`three-photon`, or `temporal-focusing`. Temporal focusing also takes
`temporal_length_um`; the source-scoped envelope is applied before integration
and resampling, including top and bottom tails.

```python
beams = make_vtwins_pupils(cfg, kind="gaussian", beam_na=.12,
                           theta_deg=20, separation_um=6)
optics = run_fixed(cfg, volume, vessels=vessels, input_pupil=beams,
                   beam_weights={"left": .7, "right": 1.3})
movie = scan_volume(neurons, optics, activity, scan, add_noise=True)
```

Beams are propagated separately. `beam_weights` weights detected intensities;
it is not an incident-field amplitude or a nonlinear laser-power fraction.
Each pupil is normalized by the propagator. Motion is shared between beams and
noise/bleed are applied once to the combined frame. Dual output has `channels`,
`beam_weights`, `image`, `unique_optical_positions=1024` and
`unique_optical_paths=2048`. Single-beam outputs retain the v3 public contract.

`make_pupil` supports Gaussian, cylindrical and four Bessel amplitude profiles.
`make_vtwins_pupils(kind="bessel")` returns two fields; `clip_to_objective=True`
reproduces the original default-profile aperture, while false exposes the
original explicit-profile branch's omitted clipping.

`run_fixed` returns a focal static Z-stack and top/bottom optical descriptors.
Use `scan_volume` or `iter_scan_volume` to include dynamic out-of-focus activity.
For large movies, feed the iterator to `maomi.movie_io.write_movie`; it supports
NPY, TIFF, FITS and calibrated FFV1 AVI. Activity arrays remain resident.

The top-level package keeps `OpticalConfig`, `run_fixed` and `zstack`.
The private former `maomi_v2` backend is now `maomi.backends.cuda`;
the duplicate legacy propagation entry points are retired. Public activity,
tissue, scanning and analysis modules remain available.
