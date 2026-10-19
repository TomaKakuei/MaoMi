# Connect optics, cells and paired images

Install the optional example dependencies:

```sh
python -m pip install ".[optics,examples]"
```

This extra installs RayOptics 0.9.5 with OpticalGlass 1.x (`>=1.1.0,<2`).
RayOptics 0.9.5 still imports the old `opticalglass.glass` API, which is absent
from OpticalGlass 2.x. The explicit bound keeps Python 3.12 installations on a
compatible combination instead of restricting MaoMi to Python 3.11.
For an existing environment with the incompatible pair, run:

```sh
python -m pip install "opticalglass>=1.1.0,<2"
python -m pip check
```

After pulling a newer MaoMi checkout, rerun the extra installation command above
to install its updated requirements. CI checks clean installations on Python
3.11 and 3.12 on Linux and Windows, including an installed-package PSF trace from
outside the source directory. These CPU checks do not validate CUDA deployment.

Trace a RayOptics lens and generate procedural specimens with clean/noisy pairs:

```sh
python examples/rayoptics_custom_psf.py --output artifacts/rayoptics_demo
python tools/verify_platform_demo.py --folder artifacts/rayoptics_demo
```

The default example generates nine scenes and four photon-budget/noise variants
per scene. All variants of one scene stay in the same dataset split. The saved
arrays contain the latent fluorescence density, per-cell components, expected
photoelectrons and noisy observations. Seeds and physical sampling are recorded
with the output. `clean_expected_e` is the denoising target; `latent_density`
is a separate reconstruction target.

To customize cells without a source photograph, edit positions, soma axes,
orientation, brightness, nuclear depression and branch paths in the recipe:

```sh
python examples/rayoptics_custom_psf.py --cells examples/cell_recipe.json --output artifacts/my_cells
```

To use a PSF from another source, save a `SampledPSF` and pass its NPZ file:

```sh
python examples/rayoptics_custom_psf.py --psf my_psf.npz --output artifacts/external_optics
```

For spatially varying PSFs:

```sh
python examples/spatial_psf.py --output artifacts/spatial_psf
python tools/verify_psf_field.py --folder artifacts/spatial_psf
python examples/rayoptics_field.py --output artifacts/rayoptics_field.npz
```

The scalar-wave example supplies a field of local kernels. The RayOptics field
example traces three object heights and resamples them to common physical
sampling. See the [PSF interface](PSF_INTERFACE.md) for units and provider contracts.

These external-optics examples use 2-D intensity imaging. The RayOptics demo
uses near-axis scalar optics; it is not a calibrated high-NA microscope.
The native 3-D tissue-scattering engine is a separate route. Full 3-D cell
editing, automatic scattering composition and real-data model training are
future capabilities.

Run the self-contained tests with `python -m unittest discover -s tests -v`.
CUDA tests skip when CUDA/CuPy is unavailable; RayOptics tests skip when its
optional dependency is absent. Examples write only to the output folder you
choose, and reject overwriting an existing nonempty result folder.
