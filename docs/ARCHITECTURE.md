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
| RANA local fields / depth banks | `maomi/rana/fields.py`, `maomi/rana/depth.py` |
| RANA optical physics | `maomi/rana/physics.py` |
| RANA acquisition operator / FFT cache | `maomi/rana/operator.py`, `maomi/rana/_fft.py` |
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

## Evolution

RANA is the main development area. Today MaoMi still hosts NAOMi-derived tissue,
propagation and scanning stages. New RANA implementations can replace those
components behind stable scene/field/acquisition contracts. The long-term goal
is an independent RANA simulator succeeding NAOMi-based MaoMi, with compatible
import/export and one coherent user workflow throughout the transition.

Preserve upstream attribution for all retained derived code. Package boundaries
describe dependencies; they do not change source provenance or licensing.
