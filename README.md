# MAOMi

Faster simulation. Open optical interfaces. Ground truth from a constructed world.

MAOMi builds on [NAOMi (Neural Anatomy and Optical Microscopy)](https://bitbucket.org/adamshch/naomi_sim/)
by Song et al. (2021), extending its simulation framework with GPU acceleration,
Python workflows and pluggable optics. See [citation and upstream credit](#citation-and-upstream-credit).

## 1. Speed

**188.7× measured acceleration. 0.05219% final-image error.**

| Implementation | Propagation sites | Optical-stage time |
| --- | --- | --- |
| Original MATLAB CPU | 4225 | 85.303 s |
| MAOMi CUDA | 1024 | 0.452 s |


Same tissue/scattering input and MATLAB pupil; full-image relative L2 error 0.05219%, below 0.2%. Median of three subsequent calls on Ryzen 9 7950X / RTX 3080. Optical stage includes phase preparation, propagation, both tails and mask assembly; input/pupil generation, static scanning, I/O and startup excluded. Measured 17 October 2026.

## 2. Extensibility

MAOMi combines MATLAB-referenced tissue, activity, scattering and movie workflows with replaceable optical providers. Physical PSF data and image formation have separate interfaces: change the optics without rewriting the renderer. External-PSF workflows run on CPU; fresh native tissue propagation uses CUDA.

| Area | MAOMi | NAOMi.jl |
| --- | --- | --- |
| Execution | Batched CUDA propagation and fused kernels. | CPU / FFTW implementation. |
| Tissue optics | Scattering propagation and multiple excitation modes. | Cortical driver deferred; example uses a Gaussian PSF. |


Julia comparison: [pinned snapshot, 17 May 2026](https://github.com/JuliaNeuroscience/NAOMi.jl/tree/b657ac7cd1416af0274a25a3c230b21429da4d7f);
[execution dependencies](https://github.com/JuliaNeuroscience/NAOMi.jl/blob/b657ac7cd1416af0274a25a3c230b21429da4d7f/Project.toml), [optical driver](https://github.com/JuliaNeuroscience/NAOMi.jl/blob/b657ac7cd1416af0274a25a3c230b21429da4d7f/src/optics/propagation.jl). Source comparison only; no Julia speed benchmark.

## 3. Connections and capabilities

| Connection | Available now |
| --- | --- |
| RayOptics | Working adapter: lens and wavefront traces become custom PSFs or local PSF fields. |
| Wave optics / measured PSFs | Callable and NPZ inputs accept sampled intensity kernels with explicit units and sampling. |
| Procedural cells / Python | Editable 2-D cells produce clean/noisy pairs, latent cell maps, seeds and scene-disjoint splits. |


The external PSF renderer currently forms 2-D intensity images; the native 3-D tissue route remains separate.


Run the CPU example:

```sh
python -m pip install ".[examples]"
python examples/spatial_psf.py --output artifacts/spatial_psf
python tools/verify_psf_field.py --folder artifacts/spatial_psf
```

For RayOptics and editable cells:

```sh
python -m pip install ".[optics,examples]"
python examples/rayoptics_custom_psf.py --cells examples/cell_recipe.json --output artifacts/my_cells
```

The `optics` extra pairs RayOptics 0.9.5 with `opticalglass>=1.1.0,<2`.
OpticalGlass 2.x changed the API used by this RayOptics release. After updating
an existing checkout, rerun `python -m pip install ".[optics,examples]"` to apply
the compatible dependency constraint.

For native CUDA propagation: install `.[cuda]`, then run
`python -m maomi --scene scattering --output artifacts/scattering`.
Requires Python 3.11+; use a CUDA-compatible PyTorch installation for GPU work.
CI covers the CPU and RayOptics examples on Python 3.11 and 3.12, on Linux and Windows.

Read the [PSF interface](docs/PSF_INTERFACE.md), [example guide](docs/PLATFORM_EXAMPLE.md),
[API](docs/API.md) and [one-page report](docs/TECHNICAL_REPORT.pdf).

## 4. What's next

Extend to editable 3-D cells and dynamics, depth- and wavelength-dependent optics, more wave solvers and detectors. Fit morphology, fluorescence and instrument parameters to real references; generate paired ground truth for denoising and reconstruction, then validate on held-out experiments. The goal is a programmable ground-truth platform for instrument design and imaging models. Full 3-D customization, real-data calibration and model training are planned capabilities.

MIT licensed. See [source attribution](THIRD_PARTY_NOTICES.md).

## Citation and upstream credit

Please cite the original NAOMi paper when using MAOMi in research:

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
To identify the MAOMi implementation used, also report
[this repository](https://github.com/TomaKakuei/MaoMi) and its release tag or commit.
Upstream notices and licensing are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [LICENSES/NAOMi-MIT.txt](LICENSES/NAOMi-MIT.txt).
