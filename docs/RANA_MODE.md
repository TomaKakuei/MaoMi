# RANA — Ray Aware Nonuniform Acquisition

**Theoretically higher accuracy through complete local PSFs and explicit
image-formation physics.** RANA retains local shape and gain throughout
rendering and implements all four extensions: depth-dependent layered optics,
vectorial high-NA focusing, finite-voxel/scan integration and pulse/spectral response.

| Mode | Image formation | Entry point |
| --- | --- | --- |
| Standard MaoMi | Native CUDA propagation, MATLAB v4 masks, arithmetic PSF averaging | `--mode standard` (default) |
| RANA | Complete local fields; cubic XY and positive linear depth interpolation; CPU/CUDA | `--mode rana` |

```sh
python -m pip install ".[examples]"
# Quick custom full-field demo; no external data required.
python -m maomi --mode rana --demo --output artifacts/rana
# All four physics extensions, 128 x 128 x 9 volume.
python -m maomi --mode rana --physics-demo --output artifacts/rana_physics
# Your saved FullFieldPSF or DepthPSFBank and fluorescence density.
python -m maomi --mode rana --psf-field bank.npz --volume density.npy --device cuda --output artifacts/custom
```

The CLI saves the noiseless image and density in NPZ and settings/timings in
JSON. `--z-origin-um` specifies the lower Z edge; its default is the first depth
in a bank. GPU rendering uses CUDA-enabled PyTorch; the vectorial PSF provider
runs on the CPU. GUI, optics construction, renderer preparation and image
rendering are timed separately.

```python
from maomi import (RanaConfig, VectorialOptics, StratifiedMedium, Acquisition,
                   PulseSpectrum, EmissionResponse, simulate)

cfg = RanaConfig(
    x_um=(0, 10, 20), y_um=(0, 10, 20), depths_um=(50, 55, 60),
    voxel_size_um=(.2, .2, .5), kernel_shape=(17, 17, 9),
    optics=VectorialOptics(na=1.1, medium=StratifiedMedium(specimen_index=1.38)),
    acquisition=Acquisition.line((.15, 0, 0), samples=3, voxel_order=3),
    pulse=PulseSpectrum(fwhm_fs=100, gdd_fs2=1800, samples=33),
    emission=EmissionResponse((500, 525, 550), (0, 1, 0), (.5, .8, .6)))
# density.shape = (100, 100, 20), fluorescence per cubic micrometre.
result = simulate(density, mode="rana", config=cfg, z_origin_um=50, device="cpu")
```

To supply your own optical solver, use `FullFieldPSF.from_provider` or
`DepthPSFBank.from_provider`. Coverage, sampling, excitation/detection meaning
and local gains stay explicit. See the [PSF contract](ENHANCED_PSF.md),
[four physics models](RANA_PHYSICS.md) and [measured comparisons](RANA_COMPARISON.md).

## Rana desktop indicator

On a desktop, the existing pixel head of the project owner's cat rotates in a
small independent window. Its text is:

> MaoMi-Rana is working in background

The window closes when the run ends. Closing it manually dismisses only the
indicator; computation continues. The cat image is unchanged. Standard mode
has no indicator; servers and CI use the same numerical operator headlessly.

Disable it with `--no-status-window`, `status_window=False`, or
`MAOMI_STATUS_WINDOW=0`. Pillow is a package dependency; Tk comes with many
Python distributions. Linux may need the matching `python3-tk` or Conda `tk`.

```sh
python -m maomi.rana_ui --seconds 10
```
