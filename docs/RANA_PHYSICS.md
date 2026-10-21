# RANA physics and numerical contracts

**RANA — Ray Aware Nonuniform Acquisition — provides theoretically higher
accuracy through local optical fields, depth-dependent aberration, vectorial
focusing, finite-volume acquisition and coherent pulse integration.**

All four extensions compose in `RanaConfig`. Attach them to the existing
acquisition pipeline with `RanaAddon(field=config)` for tissue, activity,
movie scans, noise and export. The standalone complete-field CPU/CUDA renderer
also remains available. See the [add-on guide](RANA_MODE.md).

## 1. Depth and refractive-index interfaces

`StratifiedMedium` specifies immersion, coverglass and specimen indices, plus
actual/design coverglass thickness and index. The entrance pupil coordinate is
`q = n_immersion sin(theta)`. The layer OPD, in micrometres, is

```text
W(q) = depth sqrt(ns^2-q^2) + di sqrt(ni^2-q^2)
       + tg sqrt(ng^2-q^2) - tg_design sqrt(ng_design^2-q^2)
di = -ni (depth/ns + tg/ng - tg_design/ng_design)
phase = 2 pi [W(q)-W(0)] / wavelength_um
```

This is a layered-path model with an explicit paraxial refocus convention:
`di` cancels the quadratic defocus term; subtracting `W(0)` removes piston.
Matched layers reproduce the unaberrated pupil at every depth. The remaining
depth-dependent phase includes spherical aberration. Indices are constant over
the modeled band; additional spectral phase is supplied by `PulseSpectrum.gdd_fs2`
or a wavelength-dependent pupil OPD callback. Rays must propagate in every layer.

`DepthPSFBank` holds a complete XY field at each absolute focus depth. Full
voxel-edge coverage is required in XYZ. Interpolation is cubic in XY and
positive linear interpolation in depth. Detection chooses the source depth;
raster excitation chooses the scan depth. Kernels retain their gains.

```python
bank = config.build_field()
bank.save("bank.npz")
bank = DepthPSFBank.load("bank.npz")
from maomi.rana import RanaOperator
image = RanaOperator(bank, z_origin_um=50, device="cpu").static(volume)
# For movies: run_scene(..., rana=RanaAddon(field=bank, z_origin_um=50)).
```

## 2. High-NA vectorial wave optics

`VectorialOptics` integrates the Richards–Wolf field over entrance-pupil angles.
Gauss–Legendre radial nodes and periodic azimuth nodes include the solid-angle
Jacobian and aplanatic `sqrt(cos(theta))` factor. Separate s/p Fresnel amplitudes
are transported through the immersion/glass/specimen interfaces. A normalized
incident Jones vector generates complex `Ex`, `Ey` and `Ez` in the specimen.
`electric_field(...)` exposes these components; `vectorial=False` gives the
scalar control. NA, indices, polarization and angular sampling are explicit.

There is one common entrance-field scale and wavelength-dependent `1/lambda`
amplitude prefactor. Each local PSF retains the resulting throughput; it is not
rescaled to unit sum. Absolute photon counts are set by an explicit exposure /
fluorescence calibration after the optical response.

The optional `opd(rho, phi, wavelength_nm, position_xyz_um)` callback returns
pupil OPD in micrometres; `amplitude(...)` returns nonnegative pupil amplitude.
Both see the scan position, including its within-dwell displacement. These
hooks accept field-dependent pupil models. External sampled PSFs, including
RayOptics-derived fields, also enter through `FullFieldPSF` / `DepthPSFBank`.

## 3. Finite voxels and scan dwell

`Acquisition` integrates a piecewise-constant source density over its physical
voxel and over a prescribed scan trajectory. For voxel-center offset `delta`,
source offset `u`, scan offset `s` and central scan position `r`,

```text
K(delta; r) = dwell_seconds * integral_voxel integral_scan
                effective_response(delta + u - s; r + s) dw(s) du
```

Integration happens on the optical field's sampling coordinates before kernel
discretization. The local optics are evaluated at `r+s`, not only at the pixel
center. Tensor Gauss–Legendre quadrature handles source voxels;
`Acquisition.line(...)` supplies a line trajectory. Arbitrary nonuniform
`scan_offsets_um` and `dwell_weights` are supported. Weights are nonnegative
fractions summing to one; total dwell time is separate.

Voxel volume and dwell time are retained in the kernel. Input density therefore
represents fluorescence per cubic micrometre. `voxel_order=1` and a zero-length
trajectory provide the center-sample control. Outside-volume density is zero.
Arbitrary per-voxel dwell scaling can additionally multiply the noiseless raster
response before photon noise; this keeps exposure and optical shape distinct.

## 4. Pulse and spectral response

`PulseSpectrum` samples a coherent Gaussian pulse on a uniform angular-frequency
grid. Its parameters are vacuum center wavelength, transform-limited intensity
FWHM in fs, GDD in fs², frequency samples and temporal oversampling.

```text
A(Omega) = C exp[-tau0^2 Omega^2 / (8 ln 2)] exp[i GDD Omega^2 / 2]
E(t,r) = inverse_Fourier[A(Omega) E_optical(Omega,r)]
I(t,r) = |Ex|^2 + |Ey|^2 + |Ez|^2
effective_response(r) = integral I(t,r)^m dt
```

Frequency components interfere before the nonlinear power. `C` fixes incident
fluence, so stretching a pulse lowers its two-photon response. Optical fields
are evaluated at each frequency's wavelength. The Gaussian limit is
`tau = tau0 sqrt(1 + (4 ln(2) GDD / tau0²)²)` and
`integral I² dt = sqrt(2 ln(2)/pi) / tau` for unit fluence. Orders 1, 2, 3 and
higher integer orders share the same temporal operator. An insufficient time
window raises an explicit sampling error.

Fluorescence emission is incoherent. `EmissionResponse` separately integrates
photon spectral density times collection/detector efficiency over wavelength.
It supplies a collection fraction for non-descanned fluorescence acquisition;
it is not treated as a coherent excitation spectrum. The final relative kernel
units are `um³ * dwell_seconds * fs^(1-m)` with a pulse, and
`um³ * dwell_seconds` for a monochromatic response.

## Numerical controls and comparison

The tests use independent Bessel quadrature, explicit source/scan sums,
polarization rotation, matched-interface limits, analytic Gaussian pulses and
polynomial voxel/scan integrals. CPU/CUDA comparisons use identical fields.
The [image comparison](RANA_COMPARISON.md) adds full-volume ablations, a higher
quadrature reference and serial timings. Reproduce it with:

```sh
python examples/rana_comparison.py --device cuda --output artifacts/rana_comparison
```

The supplied comparison uses spatially uniform entrance optics in XY and
depth-varying kernels. The [custom full-field example](../examples/enhanced_full_field.py)
separately exercises varying XY shapes and gains. Neither route substitutes a
shared mean PSF for a supplied local field.

## Model references

The implementation follows the field integral and interface transmission in
[PSF Generator's Richards–Wolf theory](https://psf-generator.readthedocs.io/src/theory/basic.html).
Its layered phase model is a paraxially refocused specialization of the optical
path construction described in
[PSF Generator's correction factors](https://psf-generator.readthedocs.io/src/theory/correction_factors.html).
These equations were implemented directly; no code from that project is bundled.
