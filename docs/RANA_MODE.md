# RANA — Ray Aware Nonuniform Acquisition

RANA is an **optional optical add-on to MaoMi**: theoretically higher accuracy
through complete local PSFs and explicit image-formation physics. It attaches
to the existing acquisition pipeline; anatomy, activity, scanning, noise and
export retain their existing interfaces.

## Enable it on your existing pipeline

```python
from maomi import run_fixed
from maomi.movie import scan_volume

# The only added argument is rana=True.
optics = run_fixed(cfg, volume, vessels=vessels, rana=True)
movie = scan_volume(neurons, optics, activity, scan, add_noise=True)
```

`rana=False` is the default. With it off, native propagation, MATLAB v4 masks
and mean-PSF scanning remain unchanged. Automatic RANA reuses every computed
local PSF without rerunning propagation or reducing the point count.
For CPU propagation use `run_optics(cfg, points, rana=True, ...)`.

Attach to previously computed optics:

```python
from maomi import attach_rana, RanaAddon

local_optics = attach_rana(optics, RanaAddon(device="cpu"), config=cfg)
movie = scan_volume(neurons, local_optics, activity, scan, add_noise=True)
```

Without `config`, supply `RanaAddon(voxel_size_um=(dx,dy,dz))`.
Use `volume=volume` in `attach_rana` to also render a static image.
Arrays are XYZ; distances are micrometres.

## Shared capabilities

| Existing capability | RANA integration |
| --- | --- |
| Scattering, vessels, custom pupil/phase | Reuses native per-position kernels and masks |
| One-, two-, three-photon, temporal focusing | Retains the selected native propagation model |
| Multiple beams, vTwINS / Bessel pupils | Local field per beam; shared motion; intensity sum; detector once |
| Generated or imported sparse tissue | Same `neur_vol`, `gp_vals`, `gp_nuc`, `bg_proc` |
| Calcium, AR, voltage and background activity | Same component activity inputs |
| Cytosolic / nuclear scans | Same compartment and labeling semantics |
| Dense scans | `scan_movie(..., rana=optics["rana"])` |
| Motion, scan averaging, pixel binning | Same settings; focal depth follows the attached field |
| Top/bottom background, collection, extra blur | Existing processing retained |
| Noise and bleed | Existing detector, including draw replay |
| Ideal component images | `ideal_components` uses the same scanner |
| Streaming / NPY, TIFF, FITS, FFV1 | Existing `iter_scan_volume` and `write_movie` |

## Native kernels and custom providers

Automatic attachment replaces the shared focal **shape** with complete local
shapes. Native kernels are scaled to the mean-PSF energy because the existing
source mask already carries local excitation gain. This avoids counting that
gain twice. Collection masks and both tail descriptors are retained.
When tail resampling contains negative intensity ringing, RANA projects it to
nonnegative intensity while preserving integrated gain. The original
descriptors and default scanner remain unchanged; background is still included.

The acquisition operator uses compact nonnegative cubic tensor weights:
between adjacent nodes `s(t)=3t²−2t³`, with weights `1−s(t)` and `s(t)`.
They sum to one and retain complete local kernels. Compact support permits
tiled linear convolutions. Native kernels keep MATLAB's orientation and
even-kernel crop. There is no shared-PSF fallback.

```python
from maomi import RanaAddon, attach_rana, DepthPSFBank

bank = DepthPSFBank.load("bank.npz")
local_optics = attach_rana(
    optics, RanaAddon(field=bank, z_origin_um=50., device="cuda"), config=cfg)
movie = scan_volume(neurons, local_optics, activity, scan, add_noise=True)
```

`field` accepts an XYZ-kernel `FullFieldPSF`, `DepthPSFBank` or `RanaConfig`.
For multiple beams pass one field or a dictionary keyed by channel.
The custom provider supplies the focal optical model; it is not automatically
propagated through a second, unrelated native pupil. Its response units,
sample pitch and excitation/detection meaning remain explicit. Existing
masks, tails and detector processing continue in the same pipeline.

`RanaConfig` supplies four composable extensions:

1. Layered refractive-index / coverglass aberration versus depth.
2. Vectorial high-NA focusing, polarization and interface transmission.
3. Finite source-voxel and scan-dwell quadrature.
4. Coherent pulse/spectral and incoherent emission/collection response.

Depth interpolation is positive and linear. Excitation fields vary with scan
position; detection fields vary with source position. Set `z_origin_um` to
the lower volume edge in the bank's coordinates, covering the motion range.
Custom kernels retain their gain; no exposure fitting is hidden.
See [physics and units](RANA_PHYSICS.md).

## Run a complete movie example

```sh
python -m pip install ".[examples]"
python examples/rana_movie.py --output artifacts/default_movie
python examples/rana_movie.py --rana --output artifacts/local_movie
python examples/rana_movie.py --rana --physics --nuclear --output artifacts/physics_movie
```

These runs share generated cells, AR activity, motion, tails, collection,
binning, detector noise and export. Outputs include clean/noisy movies,
latent density, spikes, activity and settings/timings. Clean acquisition is
the denoising target; latent density is a separate reconstruction target.
Streaming is checked against the collected movie. Use `--device cuda`
with CUDA-enabled PyTorch and the `cuda` extra.

For the native CLI, add `--rana` to the existing command:

```sh
python -m maomi --scene scattering --rana --output artifacts/native_local
```

The earlier `--mode rana` / `simulate(..., mode="rana")` static entry remains
available for existing scripts and optical experiments. It uses the global
cubic `r³` RBF renderer in the [full-field contract](ENHANCED_PSF.md).
That renderer and the compact positive acquisition interpolator are distinct;
the [published static comparison](RANA_COMPARISON.md) measures the former.

See [API](API.md) and [OSF datasets](DATASETS.md).
