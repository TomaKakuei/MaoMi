# MaoMi

**Modular Anatomy and Optics for Microscopy** — Faster simulation, open optics, programmable ground truth.

**Open Beta 1 · `0.8.0b1`** — [Download](#) · [Install and start](#3-capabilities-and-quickstart) · [Four-page report](#) · [Report an issue](#)  
**Author:** Yueze Wang  
Built on [NAOMi (Neural Anatomy and Optical Microscopy)](https://doi.org/10.1016/j.jneumeth.2021.109173) by Song et al. (2021). See [citation and upstream credit](#license-citation-and-upstream-credit).

---

## 1. Speed and Implementation Comparison

**49.1× measured optical-stage acceleration** with **0.00667% full-volume relative L2 difference** from the original MATLAB implementation.

| Implementation | Propagation Sites | Optical-Stage Time | Execution Backend | Native Tissue Optics |
| :--- | :---: | :---: | :--- | :--- |
| **Original MATLAB (`NAOMi-Sim`)** | 4,225 | 179.644 s | CPU | Scattering and multiple excitation modes |
| **MaoMi Standard** | 4,225 | **3.662 s** | Batched CUDA (`batch=8`) | Scattering and multiple excitation modes |
| **`NAOMi.jl`** *(source snapshot)* | — | — | CPU / FFTW | Cortical driver deferred; Gaussian example |

**Benchmark configuration:**
* **Volume & sites:** 512 × 512 × 60 tissue volume, scattering input, pupil function, and 65 × 65 (4,225) propagation sites.
* **Hardware & protocol:** AMD Ryzen 9 7950X, NVIDIA RTX 3080, median of 3 serial calls (CUDA batch size 8).
* **Scope:** Optical stage includes phase preparation, wave propagation, both tails, and mask assembly (excludes input generation, static scanning, startup, and I/O). See the four-page report for component-level benchmarks.

---

## 2. RANA (Ray-Aware Nonuniform Acquisition)

While default MaoMi reproduces MATLAB-compatible optics, passing `--rana` enables **RANA**: a modular optical engine (`maomi/rana/`) that propagates complete, spatially varying local PSFs across the entire field of view.

### Modeled Physical Effects & Numerical Convergence
Composable RANA providers model four primary physical effects:
1. Refractive-index mismatch and depth-dependent aberration
2. High-NA vectorial diffraction
3. Finite-voxel and scan-dwell integration
4. Coherent ultrashort pulse and spectral response

* **Ablation impact (256 × 256 × 16 volume, NA 1.1, specimen/immersion index 1.38/1.33, depth 50–62 µm):** Removing all four physical extensions alters the simulated volume by **30.31%**.
* **Field-dependence impact (1024² photon image):** Replacing a 5 × 5 spatially varying HCIPy PSF bank with its spatial mean changes the final image by **12.52%**.
* **Quadrature convergence & RBF reference:** Full RANA differs by **0.1030%** from higher-order numerical quadrature. On the 256 × 256 × 16 benchmark, the global cubic RBF reference operator requires 23.42 s for CPU PSF construction and renders in **0.206 s (CPU) / 0.088 s (CUDA)** (distinct from the compact positive cubic interpolator used in the default acquisition pipeline).

### Architecture & External Optical Solvers
RANA's field representation, physics modules, and renderer reside in `maomi/rana/` and connect to MaoMi's anatomy and acquisition pipeline via an adapter. External optical packages—such as **RayOptics** and **HCIPy**—or custom solvers can inject focal fields directly by implementing the PSF provider protocol without modifying core MaoMi code.

---

## 3. Capabilities and Quickstart

| Module | Supported Capabilities |
| :--- | :--- |
| **NAOMi `.mat` Compatibility** | MATLAB v5/v7.3 import, lazy loading, ROI-safe sparse indexing, imported neural activity traces |
| **External Wave & Ray Solvers** | XYZ multiphoton focal fields, portable `.npz` PSF banks, pupil coupling to tissue scattering |
| **RANA Physics** | Layered vector diffraction, polarization, voxel/scan dwell integration, dispersion, spectral weighting |
| **Specimen & Activity** | Tissue presets, 3D cell geometry, custom fluorescence kinetics |
| **Time-Resolved Acquisition** | Explicit raster/global/rolling shutter clocks, intra-frame motion and activity integration prior to detector noise, recorded timestamps |
| **Detector & Ground Truth** | Paired clean/noisy readouts, resumable movie generation, cell labels, latent ground truth for restoration/reconstruction |

### Installation & CLI Usage

Requires **Python 3.11+** (Windows/Linux CI covers 3.11 and 3.12). The `[optics]` extra pins `rayoptics==0.9.5` and `opticalglass>=1.1,<2.0`. CUDA acceleration requires a CUDA-enabled PyTorch build and CuPy (`python -m pip install ".[cuda]"`).

```bash
# Install core package with MAT import, tissue generation, and examples
python -m pip install ".[mat,tissue,examples]"

# 1. Standard NAOMi-compatible simulation (3x3 sites in example; 65x65 default)
python -m maomi run --config examples/scene.json --output artifacts/standard

# 2. Enable RANA spatially varying optics (--device cpu|cuda|auto)
python -m maomi run --config examples/scene.json --rana --output artifacts/rana

# 3. Render static images or resume interrupted runs
python -m maomi render --config examples/scene.json --rana --output artifacts/static
python -m maomi resume artifacts/rana

# 4. Inspect and run directly from original NAOMi .mat datasets
python -m maomi inspect source_tissue.mat
python -m maomi run --input source_tissue.mat --frames 300 --rana --output artifacts/imported

# 5. Supply custom external 3D PSF banks (must match voxel pitch and FOV/depth range)
python -m maomi run --config examples/scene.json --rana --psf-field bank.npz --output artifacts/custom_psf

# 6. Run RANA vectorial/dispersive physics and RayOptics integration examples
python -m pip install ".[optics,mat]"
python -m maomi run --demo --config examples/rana_physics.json --rana --output artifacts/physics
python examples/rayoptics_movie.py --route field --output artifacts/ray_movie
```

*(Note: The installed `maomi` CLI command is equivalent to `python -m maomi`.)*

### Python API & Time-Resolved Scanning

Use `run_scene` or `render_scene` with `RunConfig(rana=True)`, or pass custom optical field providers via `RanaAddon(field=...)`. To model intra-frame specimen motion or fast activity during scanning, add an `acquisition.timing` block to your scene configuration.

See the documentation for [Commands and migration](#), [Workflow guide](#), [RANA](#), [PSF interfaces](#), and [API reference](#). Original datasets: [NAOMi on OSF](#).

---

## 4. Roadmap and RANA-X Extensions

### MaoMi Roadmap
* Expand editable 3D cell morphologies and temporal dynamics.
* Calibrate forward optical and detector models against experimental two-photon datasets.
* Generate programmable ground truth for denoising and 3D reconstruction, from custom cells through optics to paired clean/noisy observations.

### Beyond Microscopy: `RANA-X` (`0.1.0.dev1`)
Evaluated alongside MaoMi in the four-page technical report, **RANA-X** generalizes RANA's forward acquisition, state management, and noise pipeline to broader multiphysics and optical domains (maintained separately from the `maomi 0.8.0b1` package):

| Domain | Executed Forward Model | Key Verification / Performance Metric |
| :--- | :--- | :--- |
| **Biological & Molecular Response** | Calcium kinetics, photobleaching/FRAP, fluorescence lifetime gates | Coupled state-space readout |
| **Optothermal Materials** | Temperature-dependent absorption coupled to heat balance | Matches independent heat-balance roots to 1.4 × 10⁻⁹ K across 40 steps (5.75 K, 8.58 K, 1.07 K at 0.4 s); bitwise-exact resume |
| **Coatings & Biosensing** | Thickness, wavelength, and analyte-index sweeps for reflection and plasmon response | **221.7×** faster on a 32,768-case reflectance sweep vs. scalar `tmm` API loops while computing internal-layer quantities |
| **Wave Propagation** | Prepared 1024² wave propagation | **1.18×** faster than cached `TorchOptics` control (**2.89×** vs. public API calls) |
| **IR Detectors & Readout** | Temperature/emissivity to black-phosphorus detector current, QE, charge, noise, gain, and ADC | End-to-end sensor electronics chain |
| **Electron Microscopy** | 200 keV elastic multislice propagation through prescribed potentials | Electron-wave and probability quantities |

---

## License, Citation, and Upstream Credit

MIT licensed. See [source attribution](#), `THIRD_PARTY_NOTICES.md`, and `LICENSES/NAOMi-MIT.txt`.

Please cite the original NAOMi paper when using MaoMi in research, alongside this repository and its release tag or commit:

> Song, A., Gauthier, J. L., Pillow, J. W., Tank, D. W., & Charles, A. S. (2021). Neural anatomy and optical microscopy (NAOMi) simulation for evaluating calcium imaging methods. *Journal of Neuroscience Methods*, 358, 109173. doi:10.1016/j.jneumeth.2021.109173

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
