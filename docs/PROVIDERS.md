# Adding an optical provider

RANA accepts a sampled `FullFieldPSF`, a `DepthPSFBank`, or any object with
`build_field()` returning one of them. `PSFProvider` describes this structural
protocol: no base class, registration or change to MaoMi is needed.
`RanaConfig` is the built-in physics provider. A third-party provider can own
its solver, calibration and dependencies while using the same acquisition.

```python
from maomi import RanaAddon, run_scene
from your_optics import InstrumentProvider

provider = InstrumentProvider(calibration="instrument.json")
run_scene(scene, rana=RanaAddon(field=provider), output="artifacts/instrument")
```

`your_optics` above represents your own module. For a runnable implementation,
see [AstigmaticProvider](../examples/custom_rana_provider.py):

```sh
python examples/custom_rana_provider.py --output artifacts/custom_bank.npz
maomi render --demo --device cpu --rana --psf-field artifacts/custom_bank.npz --output artifacts/custom_static
maomi run --demo --device cpu --rana --psf-field artifacts/custom_bank.npz --output artifacts/custom_movie
```

The example supplies an analytic field with spatially varying astigmatism and
depth dependence. It matches the demo's geometry. Replace its `kernel()` with
your effective-intensity solver, retaining the field contract below.
For direct Python use, `RanaOperator(provider, z_origin_um=10).static(volume)`
also accepts the same provider.

## Numerical boundary

- Kernels are finite, nonnegative effective intensity, with explicit gain and
  excitation/detection meaning. Convert complex wave amplitudes and photon
  order in the provider. The renderer neither squares nor normalizes them.
- Use specimen micrometres, XYZ kernel axes, positive voxel pitch and a
  complete rectangular XY field covering the FOV. Depth banks also cover the
  requested acquisition depths and motion range.
- Record solver, calibration and response units in JSON-compatible provenance.
  Kernels retain their physical amplitude; source masks remain separate.

`resolve_field()` evaluates a provider during preparation. The frame loop uses
the resulting fields and cached FFT plans. For multiple channels it resolves
each channel separately; pass a prebuilt field to share an expensive result.
Provider errors propagate before acquisition; there is no mean-PSF fallback.

Archives store the resolved numerical fields and provenance, not provider
objects or code. Resume needs neither the external solver nor its installation.
For the CLI, export the field with `.save()` and pass the existing `--psf-field`
option. JSON continues to support `rana.field` and built-in `rana.physics`;
it does not dynamically import Python provider classes.

The host adapter preserves NAOMi-derived scattering, masks, tails, motion,
activity and detector stages. A custom focal provider supplies the local
kernel within that workflow. See [RANA integration](RANA_MODE.md) and the
[architecture map](ARCHITECTURE.md) for the separation of responsibilities.
