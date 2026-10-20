# MaoMi

**Modular anatomy and optics for Microscopy.** Faster simulation, open optics, programmable ground truth.

**Author: Yueze Wang**

MaoMi = **M**odular **a**natomy and **o**ptics for **Mi**croscopy.

Built on **NAOMi (Neural Anatomy and Optical Microscopy)** by Song et al. (2021).
See [citation and upstream credit](#citation-and-upstream-credit).

## 1. Speed

**49.1× measured optical-stage acceleration. 0.00667% image difference from MATLAB.**

| Implementation | Propagation sites | Optical-stage time |
| --- | --- | --- |
| Original MATLAB CPU | 4225 | 179.644 s |
| MaoMi Standard CUDA | 4225 | 3.662 s |


Same 512 x 512 x 60 tissue, scattering input, pupil and 4225 sites. Full-volume relative L2: 0.00667%. Three serial calls, median; Ryzen 9 7950X / RTX 3080, CUDA batch 8. Optical stage includes phase preparation, propagation, both tails and mask assembly; input generation, static scanning, startup and I/O excluded. Standard-mode comparison, measured 19 October 2026 UTC.

## 2. Extensibility

RANA (Ray Aware Nonuniform Acquisition) is an optional optical add-on to MaoMi.
It retains local PSFs inside the existing anatomy, activity and acquisition
pipeline. Standard mode uses the MATLAB-compatible optical assembly.

**RANA: theoretically higher accuracy through four implemented extensions: refractive-index/depth aberration, high-NA vectorial optics, finite-voxel/scan integration and coherent pulse/spectral response.**

| Area | MaoMi | NAOMi.jl |
| --- | --- | --- |
| Execution | Batched CUDA propagation | CPU / FFTW |
| Native tissue optics | Scattering and multiple excitation modes | Cortical driver deferred; Gaussian example |


Julia: [pinned snapshot, 17 May 2026](https://github.com/JuliaNeuroscience/NAOMi.jl/tree/b657ac7cd1416af0274a25a3c230b21429da4d7f); source comparison, with no Julia timing claim.

## 3. Connections and capabilities

| Connection | Available now |
| --- | --- |
| Original NAOMi MAT files | Direct v5/v7.3 import, lazy volume/activity reads and ROI-safe sparse indices. |
| RayOptics / wave solvers | XYZ multiphoton PSF fields, complex-pupil coupling to tissue scattering and portable field banks. |
| RANA physics | Layered vector fields, polarization, voxel/scan dwell, dispersion and spectra. |
| Cells and paired data | Complete tissue presets, activity, resumable clean/noisy movies, cell labels and latent GT. |


RANA standalone static control (256 x 256 x 16): 0.1030% full-volume difference from higher quadrature; CPU/CUDA rendering 0.206/0.088 s (2.3x). PSF construction: 23.42 s CPU, measured separately.

See [RANA add-on](docs/RANA_MODE.md), [physics and units](docs/RANA_PHYSICS.md),
[comparison and individual ablations](docs/RANA_COMPARISON.md), [PSF interfaces](docs/PSF_INTERFACE.md) and [API](docs/API.md).


Enable RANA when local optical fields are needed:

| Configuration | Purpose |
| --- | --- |
| Default MaoMi | MATLAB compatibility and native CUDA speed |
| MaoMi + RANA | **Theoretically higher accuracy** with local PSFs and optional physics providers |

```sh
python -m pip install ".[examples]"
# Complete CPU movie pipeline: cells, activity, local optics, noise and GT.
python examples/rana_movie.py --rana --output artifacts/local_movie
# Same pipeline with all four physics extensions and nuclear labeling.
python examples/rana_movie.py --rana --physics --nuclear --output artifacts/physics_movie
# Native CUDA workflow: enable RANA by adding one flag.
python -m pip install ".[cuda]"
python -m maomi --scene scattering --rana --output artifacts/native_local
```

In Python, add `rana=True` to `run_fixed(...)`, then call the same
`scan_volume(neurons, optics, activity, scan)` as before. Omit the flag for the
default MATLAB-compatible workflow. Existing optical results can be enhanced
with `attach_rana`; custom PSFs and depth banks use `RanaAddon(field=...)`.
See the [integration guide](docs/RANA_MODE.md).

One workflow for imported or generated tissue, CPU/CUDA and optional RANA:

```sh
python -m pip install ".[mat,tissue,examples]"
python -m maomi run --config examples/scene.json --output artifacts/scene
python -m maomi inspect source_tissue.mat
python -m maomi run --input source_tissue.mat --frames 300 --rana --output artifacts/imported
# Continue a saved movie without repeating optical propagation:
python -m maomi resume artifacts/imported
```

`prepare_optics` reuses optics across acquisitions; local-PSF FFT caches and
packed sparse buffers accelerate repeated frames. See the
[complete workflow guide](docs/WORKFLOWS.md) for ROI import, custom optics,
configuration, checkpoints and lazy training-data access.

RayOptics and paired-data examples:

```sh
python -m pip install ".[optics,mat,examples]"
python examples/rayoptics_custom_psf.py --cells examples/cell_recipe.json --output artifacts/my_cells
python examples/enhanced_full_field.py --no-status-window --output artifacts/paired_data
python examples/rana_comparison.py --device cuda --output artifacts/comparison
python examples/rayoptics_movie.py --route field --output artifacts/ray_movie
```

Python 3.11+; Windows/Linux CPU CI covers 3.11 and 3.12. The optics extra pins
RayOptics 0.9.5 with `opticalglass>=1.1,<2`. Standard mode defaults to 65×65
sites and `matlab_v4` assembly. Existing basic PSF and native movie APIs remain.

Read the [one-page overview](docs/TECHNICAL_REPORT.pdf) and [release notes](docs/RELEASE_NOTES.md).

The original NAOMi simulated datasets are available at **[OSF: NAOMi data](https://osf.io/863j9/)**
([browse files](https://osf.io/863j9/files/)). See [downloading and using the data](docs/DATASETS.md)
for MATLAB volume/activity files and movie-only datasets.

## 4. What's next

Extend editable 3-D cells and dynamics, calibrate optical/tissue parameters against real data, and generate paired ground truth for denoising and reconstruction. Add optical providers and acquisition models through the same Python interfaces.

Built on [NAOMi, Song et al. (2021)](https://doi.org/10.1016/j.jneumeth.2021.109173).

MIT licensed. See [source attribution](THIRD_PARTY_NOTICES.md).

## Citation and upstream credit

Please cite the original NAOMi paper when using MaoMi in research:

> Song, A., Gauthier, J. L., Pillow, J. W., Tank, D. W., & Charles, A. S. (2021).
> Neural anatomy and optical microscopy (NAOMi) simulation for evaluating calcium imaging methods.
> *Journal of Neuroscience Methods*, **358**, 109173.
> [doi:10.1016/j.jneumeth.2021.109173](https://doi.org/10.1016/j.jneumeth.2021.109173)

```bibtex
@article{song2021naomi,
  author  = {Song, Alexander and Gauthier, Jeff L. and Pillow, Jonathan W. and Tank, David W. and Charles, Adam S.},
  title   = {Neural anatomy and optical microscopy ({NAOMi}) simulation for evaluating calcium imaging methods},
  journal = {Journal of Neuroscience Methods},
  volume  = {358},
  pages   = {109173},
  year    = {2021},
  doi     = {10.1016/j.jneumeth.2021.109173}
}
```

Original MATLAB implementation: [NAOMi-Sim](https://bitbucket.org/adamshch/naomi_sim/).
To identify the MaoMi implementation used, also report
[this repository](https://github.com/TomaKakuei/MaoMi) and its release tag or commit.
Upstream notices and licensing are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [LICENSES/NAOMi-MIT.txt](LICENSES/NAOMi-MIT.txt).
