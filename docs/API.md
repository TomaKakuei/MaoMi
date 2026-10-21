# Public API

## Complete scene workflow

`Scene`, `load_naomi_mat`, `RunConfig`, `run_scene` and `render_scene` connect imported or
generated source tissue to standard/RANA acquisition on CPU/CUDA.
`prepare_optics` and `render_static` separate propagation from rendering;
`save_scene` / `load_scene` and `save_optics` / `load_optics` provide portable
HDF5 archives. `MovieDataset` reads resumable clean/noisy frame pairs lazily.
`tissue_parameters` and `generate_scene` provide complete editable morphology
presets. `RunConfig(rana=True)` selects RANA for either task; `RanaAddon`
supplies a custom provider. CLI users use the same [commands](CLI.md).
See [data contracts](WORKFLOWS.md) and [architecture](ARCHITECTURE.md).

`maomi.integrations.volume.PSFVolume` and `volume_psf_field` adapt XYZ intensity,
complex vector fields or effective excitation from wave programs.
`maomi.integrations.rayoptics_volume` supplies `rayoptics_psf_volume`,
`rayoptics_volume_field` and `rayoptics_native_pupil`.

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

## RANA acquisition add-on

`run_fixed(..., rana=True)` and `run_optics(..., rana=True)` attach complete
native local PSFs. Default `rana=False` retains the original behavior.
`attach_rana(optics, RanaAddon(...), config=cfg, volume=None)` reuses
previously propagated optics. The optional volume also produces a static image.
`RanaAddon(field=None, voxel_size_um=None, origin_um=(0,0), z_origin_um=0,
device=None, status_window=True)` uses native PSFs by default; `field` accepts
`FullFieldPSF`, `DepthPSFBank`, any `PSFProvider` (including `RanaConfig`), or a
per-channel mapping. Providers implement `build_field()` and return a sampled
field or bank; no inheritance or registration is required. See the
[provider contract and runnable example](PROVIDERS.md).
Pitch must match the source volume. Custom banks use explicit depth coordinates.

The returned mapping retains native masks, tails and descriptors and adds
`optics["rana"]`. `scan_volume` / `iter_scan_volume` consume it automatically.
For dense inputs use `scan_movie(..., rana=optics["rana"])`; for static scanning
use `zstack(..., rana=optics["rana"], mask=..., collection_mask=...)`.
`settings["rana"]` records the local grid, interpolation, gain and provider.
`run_fixed` adds an attachment/render time under `timing_seconds["rana"]`;
`total_seconds` includes it.

The common workflow uses `maomi.rana.RanaOperator` with compact positive
cubic weights. `maomi.rana.reference.render_reference` exposes the historical
global cubic RBF operator by name. Older `simulate(mode="rana")` calls are
deprecated compatibility calls to that reference. New acquisitions use the
scene workflow. `MAOMI_STATUS_WINDOW=0` disables the desktop indicator.

`maomi.FullFieldPSF` carries non-normalized effective XY or XYZ kernels on a
complete XY field. `maomi.CubicFullFieldRenderer(field, shape,
voxel_size_um=..., origin_um=(0,0), device="cpu")` interpolates whole local
kernels and renders on CPU or CUDA. An explicit detection/excitation operator,
full-FOV coverage and physical kernels are required. No shared-PSF fallback.
See [contracts, equations and runnable example](ENHANCED_PSF.md).

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
`beam_weights`, `image`, `unique_optical_positions=4225` and
`unique_optical_paths=8450` at the default 65x65 grid. Single-beam outputs retain the v3 public contract.

`make_pupil` supports Gaussian, cylindrical and four Bessel amplitude profiles.
`make_vtwins_pupils(kind="bessel")` returns two fields; `clip_to_objective=True`
reproduces the original default-profile aperture, while false exposes the
original explicit-profile branch's omitted clipping.

`run_fixed(..., count_per_axis=65, mask_mode="matlab_v4")` defaults to 4225
sites, MATLAB v4 masks, and arithmetic PSF averaging. Smaller grids, such as
`count_per_axis=32`, remain explicit choices with scene-dependent accuracy.
The supersampled lattice must support the selected count; no automatic reduction.
`mask_mode="cubic"` is rejected at this public entry. The old low-level
`assemble(..., mask_mode="cubic")` is retained only for legacy diagnostics and
is not the enhanced route.

`run_fixed` returns a focal static Z-stack and top/bottom optical descriptors.
Use `scan_volume` or `iter_scan_volume` to include dynamic out-of-focus activity.
For large movies, feed the iterator to `maomi.movie_io.write_movie`; it supports
NPY, TIFF, FITS and calibrated FFV1 AVI. Sparse movie activity is read in bounded
chunks and can use lazy MAT/HDF5 or memory-mapped arrays. Use `run_scene` for
resumable paired output and full source/optics metadata.

The top-level package keeps `OpticalConfig`, `run_fixed` and `zstack`.
The private former `maomi_v2` backend is now `maomi.backends.cuda`;
the duplicate legacy propagation entry points are retired. Public activity,
tissue, scanning and analysis modules remain available.


## RANA physics and depth banks

`RanaConfig(...).build_field()` constructs a complete depth bank.
`RanaConfig.from_dict(specification)` accepts physics in JSON form;
see [the executable config](../examples/rana_physics.json).
Pass `RanaAddon(field=config)` or `RanaAddon(field=bank)` to `run_scene` or
`render_scene`. `maomi.rana.RanaOperator(bank).static(volume)` uses the core
directly, independently of the host pipeline.

`PSFProvider` and `resolve_field` are exported from `maomi` and `maomi.rana`.
`RanaOperator(provider)` resolves a provider once at construction. Host
attachment resolves it during optical preparation. Saved optics contain only
numerical fields and provenance, so resume does not need the provider code.

- `VectorialOptics`: complex vector field, polarization, Fresnel interfaces,
  custom pupil amplitude/OPD and radial/azimuth quadrature.
- `StratifiedMedium`: immersion/specimen/glass indices and actual/design glass.
- `Acquisition`: source-voxel quadrature and arbitrary weighted scan offsets;
  `Acquisition.line` constructs a line-dwell quadrature.
- `PulseSpectrum`: coherent wavelength-resolved fields and temporal integration
  at fixed input fluence; fs FWHM and fs² GDD.
- `EmissionResponse`: incoherent photon spectrum and collection/detector QE.
- `DepthPSFBank.save/load`: portable complete-field NPZ with model provenance.
- `DepthFullFieldRenderer`: cubic XY, positive linear depth, CPU/CUDA operators.

The global RBF reference renderer's timing fields separate `status_start`, `optics`, `preparation` and
`render`; `simulation_seconds` includes the last three, excluding the indicator.
See [runnable add-on guide](RANA_MODE.md) and [equations/units](RANA_PHYSICS.md).
