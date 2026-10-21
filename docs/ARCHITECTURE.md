# Architecture and development map

MaoMi exposes one workflow. RANA is a separately developed core connected to
that workflow through an adapter.

```text
maomi / python -m maomi
          |
        cli.py                  run / render / resume / inspect
          |
      workflow.py               Scene + RunConfig, preparation, archives
          |
          +-- native optics     NAOMi-derived propagation, masks, tails
          |       |
          |       +-- rana/adapters/naomi.py
          |                    |
          +-- local operator --+-- rana/operator.py
          |
       movie.py / scanning.py   acquisition, motion, detector
          |
       archive.py / dataset.py  source GT, paired chunks, exact resume
```

## Where to develop

| Area | Canonical implementation |
| --- | --- |
| Commands and help | `maomi/cli.py` |
| Common orchestration | `maomi/workflow.py` |
| Source contract / archives | `maomi/scene.py`, `maomi/archive.py` |
| RANA configuration | `maomi/rana/config.py` |
| Provider protocol and resolution | `maomi/rana/providers.py` |
| RANA local fields / depth banks | `maomi/rana/fields.py`, `maomi/rana/depth.py` |
| RANA optical physics | `maomi/rana/physics.py` |
| RANA acquisition operator / FFT cache | `maomi/rana/operator.py`, `maomi/rana/_fft.py` |
| Array operations / device selection | `maomi/rana/_backends.py` |
| Native host integration | `maomi/rana/adapters/naomi.py` |
| Global RBF reference experiments | `maomi/rana/reference.py` |
| External ray/wave providers | `maomi/integrations/` |
| Native CUDA propagation | `maomi/fixed.py`, `maomi/backends/cuda/` |
| Current tissue and acquisition host | `maomi/tissue_pipeline.py`, `maomi/movie.py` |

`__main__.py` and the old `workflow_cli.py` import one parser. The former
`enhanced.py`, `rana_depth.py`, `rana_optics.py`, `rana_addon.py` and
`_local_fft.py` are import shims, not parallel implementations. New RANA work
belongs in `maomi/rana/`.

## RANA boundary

The core accepts physical arrays and field objects. It does not import the
NAOMi-derived propagator, tissue generator, movie scanner or CLI.
Top-level MaoMi exports are lazy. Importing the core and rendering a CPU volume
does not load those host modules, PyTorch or CuPy. CUDA rendering loads PyTorch
when requested. The adapter converts native outputs to local fields and keeps
gain/mask conventions at that boundary.

```python
from maomi.rana import FullFieldPSF, RanaOperator

field = FullFieldPSF.load("field.npz")
image = RanaOperator(field, device="cpu").static(density_xyz)
```

This is a reusable optical component, not a second command-line workflow.
For a complete acquisition use the common Scene workflow.

## Extension boundaries

```text
RanaConfig / external PSFProvider / sampled fields
                       |
                resolve_field()            preparation only
                       |
          FullFieldPSF / DepthPSFBank        physical numerical data
                       |
                 RanaOperator               interpolation and acquisition
                       |
                 LocalFFTPlan               tiles and bounded FFT caches
                       |
              ArrayBackend contract         NumPy/SciPy or PyTorch CUDA
```

The provider protocol is structural: `build_field()` returns a validated field
or bank. No inheritance, central registration or host imports are required.
Providers own optical simulation; the operator owns interpolation and scanning.
Native PSF conversion and gain/mask conventions belong to the host adapter.
See the [provider guide](PROVIDERS.md) for a runnable example.

`ArrayBackend` is an internal typed contract. It owns array conversion, FFTs,
reductions and transfers. CPU/CUDA selection occurs per operator and plan;
there is no mutable process-wide backend setting. Tile scheduling, positive
cubic weights, depth interpolation and excitation/detection conventions are
shared. Future array backends implement this contract and extend the internal
factory, without changing physics providers or command parsing.

Archives serialize resolved fields and provenance, never provider instances,
backend objects or FFT caches. Resume reconstructs the operator using the
saved acquisition configuration. Existing archive schemas and the package
version contract are unchanged. A field is built before frame iteration and
the resulting operator reuses its bounded caches across frames.

## Development direction

RANA is the main development area for new optical capabilities. MaoMi builds
on NAOMi's tissue, propagation and acquisition work. New optical providers and
acquisition models extend that foundation through explicit module boundaries,
while users retain the same scene, command and export interfaces.

Preserve upstream attribution for all retained derived code. Package boundaries
describe dependencies; they do not change source provenance or licensing.
