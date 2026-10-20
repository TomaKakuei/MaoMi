# Open PSF interfaces and spatially varying image formation

MaoMi separates three responsibilities: an optical provider generates kernels;
`maomi.psf` carries sampled physical data; `maomi.imaging` forms images. Exposure,
background and detector noise remain in `maomi.synthetic`. Providers need no
registration, inheritance or dependency on the renderer.

The APIs below describe the **basic bilinear detection renderer**. For strict
full-field cubic imaging, including raster two-photon excitation and 3-D offset
kernels, use [RANA Mode](ENHANCED_PSF.md). The contracts intentionally differ:
basic fields permit edge extension and unit-sum shapes; RANA requires full
coverage and retains local gain.

## Choose the smallest interface you need

| Interface | Purpose |
| --- | --- |
| `SampledPSF` | One shift-invariant, unit-sum 2-D intensity kernel |
| `PSFProvider` | A callable `(x_um, y_um) -> SampledPSF` for any optical program |
| `PSFField.from_provider(...)` | Sample a rectilinear grid of local kernels |
| `PSFField` | Store a bank of kernels, source positions and physical metadata |
| `PSFRenderer` | Prepare kernel FFTs once and reuse across images |
| `render_image(...)` | Convenient one-shot image formation |
| `resample_psf(...)` | Explicit linear intensity resampling onto common sampling |

The existing `maomi.integrations.SampledPSF` and
`maomi.integrations.psf.SampledPSF` imports remain supported; they refer to the
same public class. Existing single-PSF NPZ files remain readable.

## A complete example without an optical-design dependency

```python
import numpy as np
from maomi.psf import SampledPSF, PSFField
from maomi.imaging import PSFRenderer

def optics(x_um, y_um):
    # Replace this illustrative intensity kernel with your program's output.
    x, y = np.meshgrid(np.arange(21)-10, np.arange(21)-10, indexing="ij")
    sx = 1.5 + x_um / 64
    intensity = np.exp(-.5*((x/sx)**2 + (y/2)**2))
    return SampledPSF.normalized(intensity, (.5, .5), 920,
                                 {"provider": "illustrative Gaussian"})

field = PSFField.from_provider(optics, [0, 32, 64], [0, 32, 64])
field.save("field.npz")
renderer = PSFRenderer(field, (128, 128), pixel_size_um=(.5, .5))
density = np.zeros((128, 128))
density[32, 32] = density[96, 96] = 1
image = renderer(density)
```

For a scalar wave-optics example with position-dependent pupil aberrations:

```sh
python examples/spatial_psf.py --output artifacts/spatial_psf
python tools/verify_psf_field.py --folder artifacts/spatial_psf
```

For real RayOptics wavefront traces at three object heights:

```sh
python -m pip install ".[optics]"
python examples/rayoptics_field.py --output artifacts/rayoptics_field.npz
```

Load the result using `PSFField.load(...)` and pass it to the same renderer or to
`expected_measurement(...)`. The RayOptics example is a one-by-three field grid:
Y varies over 0, 50 and 100 micrometres; X is constant. Its scalar, paraxial,
near-axis assumptions are explicit. It demonstrates the connection, not a
calibrated microscope objective or a high-NA vector model.

## What a wave-optics or measured-PSF adapter must specify

- Nonnegative, finite intensity; XY axis order; origin at `shape // 2`.
- Specimen-plane pixel pitches in micrometres and wavelength in nanometres.
- Unit-sum redistribution kernels. Throughput and detector efficiency are separate.
- For a field: strictly increasing physical source coordinates, matching kernel
  dimensions, pixel pitches and wavelength, plus JSON-compatible provenance.

Use `SampledPSF.normalized` to explicitly normalize an array. A third-party solver
may return a different axis order, sampling or amplitude convention; its adapter
must convert those deliberately. `resample_psf` uses linear intensity interpolation,
zero support extension and explicit renormalization, and records its mass estimate
before normalization. Use the recorded sampling and retained support when selecting a target grid. Complex amplitudes must be converted to intensity only where
that conversion matches the intended incoherent measurement model.

## Position-dependent imaging, not output blending

For source position `s` and detector position `r`, the field defines `h(r-s; s)`.
MaoMi interpolates the local kernels bilinearly at **source positions**:

```text
h(displacement; source) = sum_k w_k(source) h_k(displacement)
image = sum_k convolution(density * w_k, h_k)
```

Weights are nonnegative and sum to one. Nearest-edge extension is used outside
the supplied field grid; a singleton axis is constant. `origin_um=(0, 0)` denotes
the lower edge of the first pixel; its center is at half the pixel pitch.
Convolution uses zero padding, so light can leave the field of view. Even and odd
kernel origins follow the existing MaoMi/MATLAB convention.

Kernel FFTs are cached in an explicit reusable renderer. Each frame performs one
forward transform per active local kernel and **one inverse transform total**.
There is no global cache or hidden persistence. Memory scales with active kernels
times padded image size; choose the field grid for the optical problem.

## Scope

The external renderer handles spatially varying 2-D intensity PSFs. Run the
self-contained example and checker above to verify your generated output.
Choose and validate the PSF field grid for your optical problem. The native
3-D NAOMi-derived propagation/movie route remains separate; automatic coupling,
depth/spectral kernels and coherent fields are future extensions.

## XYZ providers and native pupil coupling

The [workflow guide](WORKFLOWS.md#connect-rayoptics-or-a-wave-solver) describes
`PSFVolume`, effective multiphoton conversion, RayOptics XYZ fields and complex
pupil coupling to native tissue scattering. The 2-D contracts above remain
unchanged.
