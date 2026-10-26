# MaoMi

**Modular anatomy and optics for Microscopy.** Faster simulation, open optics, programmable ground truth.

**Open Beta 1 · 0.8.0b1** — [Download](https://github.com/TomaKakuei/MaoMi/releases/tag/v0.8.0-beta.1)
· [Install and start](docs/GITHUB_RELEASE.md)
· [Four-page report](docs/MaoMi_RANA_4Page_Report_EN.pdf)
· [Report an issue](https://github.com/TomaKakuei/MaoMi/issues/new/choose)

**Author: Yueze Wang**

MaoMi = **M**odular **a**natomy and **o**ptics for **Mi**croscopy.

Built on **NAOMi (Neural Anatomy and Optical Microscopy)** by Song et al. (2021).
See [citation and upstream credit](#citation-and-upstream-credit).

## 1. Speed

**49.1× measured optical-stage acceleration. 0.00667% full-volume relative L2 difference from MATLAB.**

| Implementation | Propagation sites | Optical-stage time |
| --- | --- | --- |
| Original MATLAB CPU | 4225 | 179.644 s |
| MaoMi Standard CUDA | 4225 | 3.662 s |


Same 512 x 512 x 60 tissue, scattering input, pupil and 4225 sites. Full-volume relative L2: 0.00667%. Three serial calls, median; Ryzen 9 7950X / RTX 3080, CUDA batch 8. Optical stage includes phase preparation, propagation, both tails and mask assembly; input generation, static scanning, startup and I/O excluded. Standard-mode comparison, measured 19 October 2026 UTC.

The [26 October report](docs/MaoMi_RANA_4Page_Report_EN.pdf) adds component
benchmarks, cross-domain examples and controlled RANA comparisons. The 49.1×
result measures this matched optical stage, not end-to-end movie generation.

## 2. RANA and extensibility

**One workflow, one RANA switch.** Add `--rana` to the same `run` or `render`
command. Inputs, acquisition settings, outputs and movie resume work the same
way. Default MaoMi uses MATLAB-compatible optics; **RANA (Ray Aware Nonuniform
Acquisition)** uses complete local PSFs throughout the field of view.

**RANA carries local optics and acquisition detail into the predicted image.**
Composable providers retain refractive-index/depth aberration, high-NA vectorial
optics, finite-voxel/scan integration and coherent pulse/spectral response.
In the [controlled 256 × 256 × 16 comparison](docs/RANA_COMPARISON.md)
(NA 1.1, specimen/immersion index 1.38/1.33, depth 50–62 µm), removing all four
extensions changes the full volume by **30.31%**. Full RANA differs by **0.1030%**
from finer numerical quadrature. These quantify modeled effects and convergence;
accuracy against measured data remains to be established.

The practical advantage is connecting resolved optics to the specimen, activity,
exposure and detector in one forward experiment. Optical packages such as
RayOptics and HCIPy can supply fields or independent references. In the report's
separate RANA-X example, replacing the same 5 × 5 HCIPy PSF field with its mean
kernel changes the 1024² photon image by **12.52%**: retaining field dependence
matters even when the optical solver is unchanged.

RANA has its own [development core](docs/ARCHITECTURE.md) in `maomi/rana/`.
Its fields, physics and renderer can run independently; a dedicated adapter
connects them to MaoMi's current anatomy and acquisition pipeline.

New solvers implement the [PSF provider protocol](docs/PROVIDERS.md) without
changing MaoMi. Numerical fields, acquisition logic and CPU/CUDA backends have
separate responsibilities.

| Area | MaoMi | NAOMi.jl |
| --- | --- | --- |
| Execution | Batched CUDA propagation | CPU / FFTW |
| Native tissue optics | Scattering and multiple excitation modes | Cortical driver deferred; Gaussian example |

Julia: [pinned snapshot, 17 May 2026](https://github.com/JuliaNeuroscience/NAOMi.jl/tree/b657ac7cd1416af0274a25a3c230b21429da4d7f);
source comparison, with no Julia timing claim.

## 3. Connections and capabilities

| Connection | Available now |
| --- | --- |
| Original NAOMi MAT files | v5/v7.3 import, lazy input, ROI-safe sparse indices and imported activity |
| RayOptics / wave solvers | XYZ multiphoton fields, portable PSF banks and pupil coupling to tissue scattering |
| RANA physics | Layered vector fields, polarization, voxel/scan dwell, dispersion and spectra |
| Specimen and activity | Tissue presets, cell geometry and imported activity traces feeding the same imaging workflow |
| Detector and paired data | Clean signal and noisy readout, resumable movies, cell labels and latent ground truth for restoration and reconstruction |
| Time-resolved acquisition | Explicit raster/global/rolling clocks, within-frame activity and specimen motion, recorded timestamps |

Install from the checkout and run a complete small example:

```sh
python -m pip install ".[mat,tissue,examples]"
# Standard MaoMi
python -m maomi run --config examples/scene.json --output artifacts/standard
# Same scene, activity, acquisition and output format, with RANA
python -m maomi run --config examples/scene.json --rana --output artifacts/rana
# Static images use the same inputs and the same switch
python -m maomi render --config examples/scene.json --rana --output artifacts/static
# Resume restores the saved optics and settings automatically
python -m maomi resume artifacts/rana
```

The installed `maomi` command is equivalent to `python -m maomi`.
`maomi --help` lists **run, render, resume and inspect**.
The small example uses 3x3 optical sites; regular runs default to 65x65.
Select `--device cpu` or `--device cuda`; `auto` selects available CUDA with CuPy.
CUDA requires CUDA-enabled PyTorch and `python -m pip install ".[cuda]"`.

Native RANA retains the propagated local kernels automatically. To supply your
own focal optics, add `--psf-field bank.npz` to the same command. Fields must
contain complete XYZ kernels, match the voxel pitch and cover the whole FOV.
Depth banks must also cover the acquisition depths.

```sh
python -m maomi inspect source_tissue.mat
python -m maomi run --input source_tissue.mat --frames 300 --rana --output artifacts/imported
python -m maomi run --demo --config examples/rana_physics.json --rana --output artifacts/physics
python -m pip install ".[optics,mat]"
python examples/rayoptics_movie.py --route field --output artifacts/ray_movie
```

In Python, use `run_scene` / `render_scene` with `RunConfig(rana=True)`;
custom providers use `RanaAddon(field=...)`. Both operations share optical
preparation. See [commands and migration](docs/CLI.md), [workflow guide](docs/WORKFLOWS.md),
[RANA](docs/RANA_MODE.md), [PSF interfaces](docs/PSF_INTERFACE.md) and [API](docs/API.md).

For motion during a scan, add `acquisition.timing` to the same configuration.
The [time-resolved workflow and 512-pixel comparison](docs/TIME_RESOLVED_ACQUISITION.md)
integrate changing photon rates before detector noise, with one common frame
midpoint and complete local PSFs in RANA. The existing acquisition remains the
default when this optional section is omitted.

The [global cubic RBF reference comparison](docs/RANA_COMPARISON.md)
(256 x 256 x 16) records 0.1030% difference from higher quadrature,
0.206/0.088 s CPU/CUDA rendering and 23.42 s CPU PSF construction.
This reference operator is distinct from the compact positive cubic
interpolator used by the shared acquisition workflow.

Python 3.11+; Windows/Linux CPU CI covers 3.11 and 3.12. The optics extra pins
RayOptics 0.9.5 with `opticalglass>=1.1,<2`.
See the [one-page overview](docs/TECHNICAL_REPORT.pdf) and [release notes](docs/RELEASE_NOTES.md).
The [four-page report](docs/MaoMi_RANA_4Page_Report_EN.pdf) covers speed,
connections beyond optics, RANA comparisons and positioning among existing simulators.
Original datasets: **[NAOMi on OSF](https://osf.io/863j9/)**
([files](https://osf.io/863j9/files/), [usage guide](docs/DATASETS.md)).

## 4. What's next

**RANA is the primary development direction for new optical capabilities.**
MaoMi builds on NAOMi's anatomy and microscopy foundation. RANA develops
complementary optics and acquisition models through a modular core, preserving
NAOMi compatibility, attribution and one user workflow.

Extend editable 3-D cells and dynamics, connect more optics and acquisition
models, and calibrate simulations against real data. Generate programmable
ground truth for denoising and reconstruction, from custom cells through
optics to clean/noisy observations.

**Beyond microscopy: demonstrated in independent RANA-X development.**
RANA-X 0.1.0.dev1 connects named physical quantities, models and solvers while
retaining acquisition, state, noise and export. The report includes these
executed examples; they are separate from the MaoMi 0.8.0b1 package.

| Connection | Executed example |
| --- | --- |
| Biological and molecular response | Calcium kinetics, bleaching/FRAP and fluorescence lifetime gates |
| Optothermal materials | Temperature-dependent absorption coupled to heat balance; replace the cooling model and solver |
| Coatings and biosensing | Thickness, wavelength and analyte-index sweeps for reflection and plasmon response |
| Infrared detectors and camera electronics | Temperature/emissivity to black-phosphorus detector current; QE, charge, noise, gain and ADC readout |
| Electron microscopy | 200 keV elastic multislice through prescribed potentials using electron-wave and probability quantities |

For the thermal example, changing the material equations gives temperature rises
of **5.75 K, 8.58 K and 1.07 K** at 0.4 s; all 40-step traces agree with independent
heat-balance roots to **1.4 × 10⁻⁹ K**, and resume reproduces all five output arrays
bitwise. These are illustrative physical models.

New RANA-X component timings also show where reuse helps: prepared 1024² wave
propagation is **1.18×** faster than a cached TorchOptics control (**2.89×** versus
public calls); a 32,768-case coating sweep is **221.7×** faster than a tmm scalar-API
loop for the same reflectance observable. The latter also computes internal-layer
quantities. These are bounded component comparisons, with settings and numerical
agreement in the report, rather than rankings of complete simulators.

MIT licensed. See [source attribution](THIRD_PARTY_NOTICES.md).

### Citation and upstream credit

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
