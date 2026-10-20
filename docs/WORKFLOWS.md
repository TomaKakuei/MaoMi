# Source tissue to paired movies

Author: Yueze Wang

`Scene` holds anatomy, activity, optics and acquisition inputs. `RunConfig`
selects CPU/CUDA, optional RANA, noise, sampling and output chunks.

## Run a complete example

```sh
python -m pip install ".[mat,tissue,examples]"
python -m maomi run --config examples/scene.json --output artifacts/scene
# Optional CUDA:
python -m pip install ".[cuda]"
python -m maomi run --config examples/scene.json --device cuda --output artifacts/scene_cuda
```

The example explicitly selects a compact morphology preset and 3-by-3 optical
grid. `RunConfig` otherwise defaults to 65-by-65 sites. `rana=False` retains the
standard acquisition. RANA is **Ray Aware Nonuniform Acquisition**: theoretically
higher accuracy through complete local PSFs and optional physical extensions.

`device="auto"` selects CUDA when CUDA-enabled PyTorch and CuPy are available,
otherwise CPU. Explicit device requests are honored. The existing
`simulate(..., device="cpu")` also supports static native imaging.
CUDA accelerates native optical propagation and RANA local-PSF convolution.
Standard movie convolution, motion, detector noise and file output currently
run on the CPU.

`generate_scene(..., preset="naomi")` enables vessels, dendrites, background
and axons. Default dendrite block widths adapt to divide the requested voxel
dimensions; tissue size and voxel pitch stay as requested. Explicit
`overrides={"dend": {"dimsSS": [...]}}` values are retained.

## Import original NAOMi MAT files

```python
from maomi import load_naomi_mat, RunConfig, run_scene

scene = load_naomi_mat("source_tissue.mat")
# Explicitly generate activity if the source contains anatomy only:
scene.generate_activity(frames=300, fps=30, rate_hz=0.5, seed=19)
data = run_scene(scene, settings=RunConfig(device="auto", rana=True, seed=19),
                 output="artifacts/imported_movie")
```

```sh
python -m maomi inspect source_tissue.mat
python -m maomi run --input source_tissue.mat --activity-mat traces.mat --activity-key neur_act --rana --output artifacts/imported_movie
```

The importer supports MATLAB v5-v7.2 through SciPy and v7.3 through HDF5. It
recognizes `vol_out`, `vol_params`, `psf_params`, scan/noise/spike/TPM parameters,
and `activity` or `neur_act`. `tissue_key`, `activity_key` and `activity_path`
select other variable names or a separate activity file. Missing activity
remains `None` until explicitly supplied or generated.

`roi=((x0,x1),(y0,y1),(z0,z1))` uses zero-based, half-open source voxel bounds.
XYZ order, voxel pitch and absolute tissue depth are retained. `gp_vals`,
`gp_nuc` and `bg_proc` indices are remapped in **one-based MATLAB column-major
order**. All component IDs/rows, including empty cropped components, remain
aligned with activity. v7.3 volumes, activity and cell rows can stay lazy;
v5-v7.2 decoding is eager. `save_scene` converts either to a lazy HDF5 archive.

Optical fields populate `OpticalConfig`. Defaults used for missing fields and
the original optical parameters are recorded in `scene.metadata`.
Fixed Zernike weights retain NAOMi's coefficient convention and enter native
propagation. Non-Gaussian prescriptions or spatial MATLAB callbacks require
explicit `input_pupil`/`custom_phase`; they are never silently replaced by an
unaberrated Gaussian beam.

Vessels retain optical padding and surface-to-focus depth. If stored anatomy
does not cover the requested focal slab and lower tail, propagation reports
the required/available shapes. Select suitable `optical_overrides` at import;
missing anatomy is not fabricated. CLI JSON accepts `"optical_overrides"`.
A mean-PSF-only file cannot supply missing local shapes for RANA.

## Prepare optics once

```python
from maomi import prepare_optics, render_static, save_optics, load_optics
from maomi.movie import iter_scan_volume

optics = prepare_optics(scene.config, vessels=scene.vessels,
                        device="auto", rana=True, count_per_axis=65)
image = render_static(scene.neurons["neur_vol"], optics, device="auto")
frames = iter_scan_volume(scene.neurons, optics, scene.activity, scene.scan)
save_optics(optics, "optics.h5")
restored = load_optics("optics.h5", device="cpu")
```

`prepare_optics` computes no static image; `run_fixed(..., render_image=False)`
provides the same separation. RANA no longer first renders a discarded shared
PSF image. The shared preparation API accepts pupils, custom phase, raw A/B/C
fields, excitation modes and incoherent multiple beams. CPU propagation
currently computes tails even when `tails=False` omits their descriptors.

RANA uses compact-support batched FFTs and resident device weights. Kernel
spectra are cached within a 256 MiB budget per operator; temporary FFT batch
sizes are bounded separately. CPU uses scalar batches to avoid stacking
overhead. `optics["rana"].cache_info` reports reuse; `.clear_cache()` releases
caches. Local shapes/gains and source-versus-scan weighting are preserved.

Sparse movies reuse packed buffers and optionally compile ordered writes
with Numba. Set `scene.scan["sparse_backend"]="numpy"` for NumPy.
Overwrite/additive semantics, duplicate indices and activity thresholds remain.
Activity minima span the whole recording, computed in bounded chunks.
Lazy activity/vessels avoid full input copies; the current scanner still keeps
a dense working volume and the propagated optical arrays.

For long recordings, supply `output=...` to `run_scene` or consume
`iter_scan_volume` with a streaming exporter. `scan_volume` and
`run_scene(output=None)` collect the full movie in RAM. Generated activity and
the motion trajectory are still prepared for the whole recording; output
chunking bounds image buffers, not every source array.

Automatically generated motion is bounded to the row scanner's valid crop.
Explicit motion retains the MATLAB coordinate convention; its X and Y
offsets have different origins.

## Resume and train

```sh
python -m maomi run --config examples/scene.json --output artifacts/checkpoint --max-frames 3
python -m maomi resume artifacts/checkpoint
```

```python
from maomi import MovieDataset

data = MovieDataset("artifacts/checkpoint", verify=True)
pair = data[0]           # clean/noisy YX arrays and frame index
source = data.scene()    # lazy tissue, activity and sparse cell labels
metadata = data.metadata
# torch.utils.data.DataLoader(data, batch_size=8) is supported.
```

An output folder contains `scene.h5` (anatomy, labels, vessels, activity and
supplied spikes), `optics.h5` (PSFs, masks, tails and local fields), `run.json`
(configuration, version, full motion path, seed and target definitions),
atomic `frames_*.npz` TYX chunks, and `dataset.json` with progress/checksums.

Resume verifies saved chunks and restores archived inputs/optics. It starts
at the first uncommitted frame without re-propagating or replaying earlier
frames. Frame-indexed detector randomness preserves interrupted/uninterrupted
agreement. Interruption during initial scene/optics preparation precedes this
movie checkpoint. Existing TIFF/FITS/NPY/AVI export remains available.

The clean target includes optical blur, background and motion. Latent density
is a separate reconstruction target. Keep all variants of one tissue scene
within the same train/validation/test split.

## Connect RayOptics or a wave solver

```sh
python -m pip install ".[optics,mat]"
python examples/rayoptics_movie.py --route field --output artifacts/ray_field
python examples/rayoptics_movie.py --route pupil --output artifacts/ray_pupil
```

The field route traces a scalar pupil, propagates an XYZ angular-spectrum
intensity volume and constructs effective multiphoton kernels. `field_indices`
maps specimen positions to lens model fields. The small example explicitly
repeats one on-axis prescription; supply distinct model fields for varying
aberrations. The pupil route maps the traced normalized pupil to the declared
objective aperture, with matching excitation wavelengths, then enters native
A/B/C propagation. Aberration and scattering compose before intensity and
multiphoton excitation. The RayOptics scalar/paraxial model and RANA vectorial
provider retain their distinct physical models.

```python
from maomi.integrations.volume import PSFVolume, volume_psf_field
from maomi import RanaAddon

def provider(x_um, y_um):
    samples = my_wave_solver(x_um, y_um)  # XYZ or XYZ-by-polarization
    return PSFVolume(samples, (0.5, 0.5, 1.0), 920, kind="complex_field",
                     provenance={"solver": "my solver"})

field = volume_psf_field(provider, [0, 50, 100], [0, 50, 100],
                         photon_order=2, normalization="reference_sum")
addon = RanaAddon(field=field)
```

Use `"intensity"` for linear intensity or `"effective_excitation"` for an
already computed multiphoton response; the latter is not exponentiated again.
`reference_sum` applies one common divisor and preserves relative node gains;
`none` retains amplitudes. Pitch, wavelength, axes and center (`shape//2`) are
explicit. `DepthPSFBank` combines independently supplied fields at multiple
specimen depths.
