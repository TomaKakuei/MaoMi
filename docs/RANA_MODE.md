# RANA: Ray Aware Nonuniform Acquisition

RANA is an optional add-on within MaoMi's common workflow and the project's
primary area for future simulator development. It provides **theoretically
higher accuracy** through complete local PSFs and composable physical models.

## One switch, the same workflow

```sh
python -m pip install ".[mat,tissue]"
maomi run --config examples/scene.json --output artifacts/standard
maomi run --config examples/scene.json --rana --output artifacts/rana
maomi render --config examples/scene.json --rana --output artifacts/static
maomi resume artifacts/rana
```

`--rana` enables local-field acquisition. Input, activity, device, sampling,
output and checkpoint options are shared with standard MaoMi. Resume restores
the saved configuration. See [all commands](CLI.md).

```python
from maomi import RunConfig, run_scene, render_scene
from maomi.rana import RanaAddon, DepthPSFBank

settings = RunConfig(device="auto", rana=True)
data = run_scene(scene, settings=settings, output="artifacts/movie")
image = render_scene(scene, settings=settings)["image"]

bank = DepthPSFBank.load("bank.npz")
provider = RanaAddon(field=bank, z_origin_um=scene.config.depth_um)
data = run_scene(scene, settings=settings, rana=provider, output="artifacts/custom")
```

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

## Providers and development

```sh
maomi run --config examples/scene.json --rana --psf-field bank.npz --output artifacts/custom
maomi run --demo --config examples/rana_physics.json --rana --output artifacts/physics
```

Fields must match voxel pitch and cover the entire FOV. Depth banks also cover
the acquisition depth range, including motion. `rana.physics` in JSON accepts
`RanaConfig.from_dict`. See [configuration](CLI.md#rana-optical-providers).

Low-level `run_fixed(..., rana=True)` and `run_optics(..., rana=True)` remain
supported. `attach_rana` reuses existing propagation. Without an OpticalConfig,
declare `RanaAddon(voxel_size_um=(dx,dy,dz))`.

RANA's canonical implementation lives in `maomi/rana/`; native host coupling is
in `maomi/rana/adapters/naomi.py`. The [development map](ARCHITECTURE.md) describes
how this supports gradually replacing NAOMi-derived components and eventually
releasing RANA as an independent successor.

`CubicFullFieldRenderer`, `DepthFullFieldRenderer` and `render_reference` expose
the global cubic RBF reference operator used by the
[historical static comparison](RANA_COMPARISON.md). They are named numerical
components, not another RANA CLI mode. See the [field contract](ENHANCED_PSF.md)
and [migration guide](CLI.md#migration-from-older-snapshots).
