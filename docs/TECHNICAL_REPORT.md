# MaoMi

**Modular anatomy and optics for Microscopy.** Faster simulation, open optics, programmable ground truth.

## 1. Speed

**49.1× measured optical-stage acceleration. 0.00667% image difference from MATLAB.**

| Implementation | Propagation sites | Optical-stage time |
| --- | --- | --- |
| Original MATLAB CPU | 4225 | 179.644 s |
| MaoMi Standard CUDA | 4225 | 3.662 s |


Same 512 x 512 x 60 tissue, scattering input, pupil and 4225 sites. Full-volume relative L2: 0.00667%. Three serial calls, median; Ryzen 9 7950X / RTX 3080, CUDA batch 8. Optical stage includes phase preparation, propagation, both tails and mask assembly; input generation, static scanning, startup and I/O excluded. Standard-mode comparison, measured 19 October 2026 UTC.

## 2. Extensibility

RANA (Ray Aware Nonuniform Acquisition) is an optional optical add-on. Native local PSFs and custom physics providers feed the same anatomy, activity, movie, noise and export workflow. Default MaoMi remains MATLAB-compatible.

**RANA: theoretically higher accuracy through four implemented extensions: refractive-index/depth aberration, high-NA vectorial optics, finite-voxel/scan integration and coherent pulse/spectral response.**

| Area | MaoMi | NAOMi.jl |
| --- | --- | --- |
| Execution | Batched CUDA propagation | CPU / FFTW |
| Native tissue optics | Scattering and multiple excitation modes | Cortical driver deferred; Gaussian example |


Julia: [pinned snapshot, 17 May 2026](https://github.com/JuliaNeuroscience/NAOMi.jl/tree/b657ac7cd1416af0274a25a3c230b21429da4d7f); source comparison, with no Julia timing claim.

## 3. Connections and capabilities

| Connection | Available now |
| --- | --- |
| RayOptics / wave solvers | Sampled local PSFs, pupil OPD/amplitude callbacks and portable field banks. |
| RANA physics | Layered vector fields, polarization, voxel/scan dwell, dispersion and spectra. |
| Cells and paired data | Editable cells, activity, clean/noisy movies and latent GT; shared acquisition/export. |


RANA standalone static control (256 x 256 x 16): 0.1030% full-volume difference from higher quadrature; CPU/CUDA rendering 0.206/0.088 s (2.3x). PSF construction: 23.42 s CPU, measured separately.

Original data: [NAOMi on OSF](https://osf.io/863j9/).

See [RANA add-on](RANA_MODE.md), [physics and units](RANA_PHYSICS.md),
[comparison and individual ablations](RANA_COMPARISON.md), [PSF interfaces](PSF_INTERFACE.md) and [API](API.md).

## 4. What's next

Extend editable 3-D cells and dynamics, calibrate optical/tissue parameters against real data, and generate paired ground truth for denoising and reconstruction. Add optical providers and acquisition models through the same Python interfaces.

Built on [NAOMi, Song et al. (2021)](https://doi.org/10.1016/j.jneumeth.2021.109173).
