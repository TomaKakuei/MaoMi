# RANA full-field PSF contract

MaoMi's optional RANA add-on retains a complete local PSF at every position.
This page describes `FullFieldPSF` and the standalone global cubic RBF
`CubicFullFieldRenderer`. The native acquisition add-on uses the same field
contract with compact positive cubic interpolation.

**Theoretically higher accuracy** through position-dependent local PSFs.
See the [add-on switch and movie integration](RANA_MODE.md).

## Run it

```sh
python -m pip install ".[examples]"
python examples/enhanced_full_field.py --output artifacts/enhanced
# Optional CUDA PyTorch backend; this external-PSF route does not need CuPy:
python examples/enhanced_full_field.py --device cuda --output artifacts/enhanced_cuda
```

The self-contained example constructs a 3x3 field of effective two-photon
excitation kernels, editable fluorescence and clean/noisy observations. It
preserves the varying gain. The example supplies an illustrative prescribed two-photon response. `clean_e.npy` is the denoising target;
`density.npy` is the distinct latent reconstruction target. Preparation and
render times are recorded separately; they are example timings, not speed claims.

For a wave solver, a measured field, or another provider:

```python
from maomi import FullFieldPSF, CubicFullFieldRenderer

# effective_kernel(x_um, y_um) returns an XY or XYZ intensity response array.
# Nodes cover the whole 64 x 64 um field, including its edges.
field = FullFieldPSF.from_provider(
    effective_kernel, [0, 32, 64], [0, 32, 64],
    voxel_size_um=(.5, .5), wavelength_nm=920, operator="excitation",
    provenance={"provider": "my optical solver", "units": "my calibrated response"},
)
field.save("full_field.npz")
render = CubicFullFieldRenderer(field, (128, 128), voxel_size_um=(.5, .5))
image = render(density)
```

Saved custom data can use the same example runner:

```sh
python examples/enhanced_full_field.py --field full_field.npz --density density.npy --output artifacts/custom
```

## Full-field contract

- At least two distinct, increasing nodes on **each** XY axis; the grid must
  cover the image's full physical extent, including edges. No edge extension.
- Arrays use `field_X, field_Y, kernel_X, kernel_Y[, kernel_Z]`. Image arrays
  use XY or XYZ. All kernels share sampling, support and wavelength. The
  discrete kernel origin is `shape // 2`, including even sizes.
- Finite nonnegative effective intensity kernels with positive local gain.
  Kernel sums are retained, not silently normalized. Supply throughput, power
  and collection factors according to your forward model and record their units.
- Explicit `operator="detection"` or `operator="excitation"`. For two-photon
  excitation, supply the effective intensity-squared response. Already effective
  measured two-photon PSFs must **not** be squared again.
- Image and kernel pitches must match. `origin_um` is the XY lower image edge;
  the first pixel center is half a pitch inward. Outside-image fluorescence is
  zero, so finite-FOV loss is possible.
- Cubic interpolation is checked at **every rendered XY pixel center**. Negative
  values below a relative roundoff tolerance are rejected. The renderer never
  hides missing coverage or nonphysical kernels by falling back to a mean PSF.
  Checks are evaluated on the rendering grid.

`FullFieldPSF.from_psf_field(field, operator=..., gains=...)` adapts existing
normalized `PSFField` data, including RayOptics adapter output. `gains` is one
positive value per node; omitted gains explicitly mean unit gain. A 1x3 field
such as the basic RayOptics line example is insufficient for this route: trace
a complete XY grid using `rayoptics_psf_field` and the corresponding model field
indices. Conversion does not create unmeasured positions, convert a detection
PSF to excitation, or recover throughput lost by prior normalization.

## Which position selects the PSF?

For fluorescence density `rho(s)`, emitter position `s`, and observation/scan
position `r`, the two incoherent operators are:

```text
Detection:  y(r) = sum_s rho(s) h(r - s; s)
Excitation: y(r) = sum_s rho(s) E(s - r; r)
```

Detection weights the source before convolution. Raster excitation weights the
scan position after correlation. The latter reverses kernel offsets and uses
the corresponding even-kernel crop. Squaring an intensity kernel alone does
not turn the detection operator into a raster two-photon excitation operator.

## Why theoretically higher accuracy

The enhanced route retains the affine-augmented cubic RBF (`r^3`) used by the
earlier interpolation work, but applies it to **every element of the local PSF**,
not just energy masks or weights of an average kernel. At normalized XY position
`p`, the whole kernel is `H(delta; p) = sum_j b_j(p) C_j(delta)`. The augmented
system interpolates the sampled kernels and reproduces affine spatial fields
up to numerical precision; its polynomial augmentation is affine.

Linearity gives an exact basis implementation of this interpolated operator:

```text
Detection:  sum_j convolution(rho * b_j, C_j)
Excitation: sum_j b_j * correlation(rho, C_j)
```

No local kernels are averaged away and no rank truncation is used. FFTs reduce
work compared with a separate convolution for every output location. CPU and
CUDA use float64; kernel transforms are streamed to avoid storing one full
image FFT per node. Dense field fitting still costs cubic time in node count
and quadratic storage; this route is intended for deliberately sampled custom
fields with an explicit sampling budget.

**RANA Mode is theoretically more accurate** because it retains local PSF
width, asymmetry, aberration and gain instead of reducing them to a common
shape. Cubic interpolation represents smooth spatial changes between the
supplied samples. Local-kernel validity checks reject nonphysical overshoot;
refine the supplied field when a check fails. Complete local kernels and gains
are retained throughout image formation.

`FullFieldPSF` describes one XY field with XY or XYZ offset kernels.
`DepthPSFBank` adds absolute focal-depth dependence; `DepthFullFieldRenderer`
combines complete fields with positive linear depth interpolation. `RanaConfig`
can generate these banks using layered vectorial optics, finite-voxel and scan
integration, coherent pulse spectra and emission collection. See
[RANA physics](RANA_PHYSICS.md). Native tissue propagation and tail-background
movies continue through the standard APIs.

The tests compare the implementation with explicit sums for detection and
excitation, asymmetric odd/even kernels, XY/XYZ data, and CPU/CUDA. They also
check node interpolation, affine reproduction, gain preservation, full-FOV
coverage, and rejection of cubic overshoot. These checks cover the numerical and physical-data contracts.
