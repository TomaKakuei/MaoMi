# Commands

Use `maomi` or `python -m maomi`. They invoke the same parser.

| Command | Purpose | Output |
| --- | --- | --- |
| `run` | Complete time-dependent acquisition | Paired clean/noisy frame chunks, source tissue and GT |
| `render` | Noiseless static focal XYZ stack | `image.npz`, source scene, reusable optics and metadata |
| `resume` | Continue a saved movie | Additional chunks in the same directory |
| `inspect` | Inspect MAT, scene, PSF field or output metadata | JSON on stdout |

## Standard and RANA

```sh
maomi run --config examples/scene.json --output artifacts/standard
maomi run --config examples/scene.json --rana --output artifacts/rana
maomi render --config examples/scene.json --output artifacts/static_standard
maomi render --config examples/scene.json --rana --output artifacts/static_rana
```

`--rana` is the only CLI switch for RANA. In JSON its equivalent is
`"run": {"rana": true}`. A provider configuration by itself does not enable
RANA: a disabled configuration with a supplied provider reports an error.
There is no separate RANA command set.

Both modes accept `--input`, `--config`, `--demo`, `--roi`, `--device`, `--seed`,
`--count-per-axis`, `--batch-size`, `--focal-only` and `--output`.
Run-specific activity/checkpoint options and render-specific `--volume` work
identically in both modes. `--output` always means a **new directory**.

Run `maomi run --help` or `maomi render --help` for the complete options.
`--demo` supplies a small complete phantom without the tissue extra. It uses
3x3 optical sites and eight movie frames unless overridden. For morphology
use a tissue config or imported scene.

## Sources and configuration

```sh
maomi inspect source.mat
maomi run --input source.mat --frames 300 --output artifacts/imported
maomi run --input source.mat --activity-mat activity.mat --activity-key neur_act --rana --output artifacts/imported_local
maomi render --input artifacts/imported/scene.h5 --rana --output artifacts/static
maomi render --volume density.npy --config geometry.json --rana --output artifacts/density
```

`--volume` accepts nonnegative XYZ fluorescence and needs `optical` geometry
in JSON matching its dimensions and pitch. Static rendering uses baseline
density; movie acquisition uses activity traces. Output formats differ by
task, never by whether RANA is enabled.

The [example config](../examples/scene.json) contains `optical`, `run`,
`activity` and `acquisition`. Other supported sections are `input`, `roi`,
`optical_overrides`, `preset`, `tissue`, `noise`, `spikes`, `tpm` and `rana`.
`optical`/`tissue` configure generated inputs; an imported scene supplies its
own geometry, with `optical_overrides` applied during MAT import.
CLI values override corresponding config values. Paths inside JSON are relative
to that JSON file; CLI paths are relative to the working directory.

The compact example explicitly sets `acquisition.tail_policy="nonnegative"`.
This projects negative tail-kernel resampling ringing onto nonnegative intensity
while preserving integrated gain, so sparse source tissue has valid photon
rates. The raw optical archive is unchanged. `"native"` retains raw MATLAB
resampling; it is standard mode's default. RANA defaults to `"nonnegative"`.
Both modes accept either policy through the same acquisition configuration.

## RANA optical providers

```sh
maomi run --config examples/scene.json --rana --psf-field bank.npz --output artifacts/custom_movie
maomi render --config examples/scene.json --rana --psf-field bank.npz --output artifacts/custom_static
maomi run --demo --config examples/rana_physics.json --rana --output artifacts/physics_movie
maomi render --demo --config examples/rana_physics.json --rana --output artifacts/physics_static
```

Without a custom provider RANA retains native per-position PSFs. A supplied
field must contain XYZ kernels, match the scene's voxel pitch and cover the
entire FOV. Depth banks must cover acquisition depths, including motion.
Native scattering, masks, tails, acquisition and detector processing stay
connected when a custom **focal** provider is selected.

`rana.field` is the JSON equivalent of `--psf-field`; `rana.physics` configures
`RanaConfig.from_dict`. Choose one provider. Optional `rana.origin_um` and
`rana.z_origin_um` declare coordinates; the latter defaults to the scene's
lower depth. See [physics and units](RANA_PHYSICS.md) for optical gain conventions.

## Checkpoint and resume

```sh
maomi run --config examples/scene.json --rana --output artifacts/movie --max-frames 3
maomi resume artifacts/movie
maomi inspect artifacts/movie
```

Resume restores RANA selection, local fields, device, seeds, motion and source
inputs. Do not supply mode/optics flags again. It starts at the first uncommitted
frame and requires the same package version. `--max-frames` also bounds the
next resume invocation. Initial preparation and static renders are not movie
checkpoints.

## Migration from older snapshots

The former `--mode rana` selector and top-level demo flags are removed.
Use `maomi render --demo --rana ...` or `maomi run --demo --rana ...`.
Replace a filename prefix with an output directory. The `maomi-fixed` and
`maomi-run` console scripts are retired; use `maomi render` and `maomi run`.

The former static selector used a global cubic RBF operator. Experiments needing
that exact operator should call `maomi.rana.reference.render_reference` or
`CubicFullFieldRenderer` / `DepthFullFieldRenderer`. Shared `run` and `render`
commands both use `RanaOperator` with compact positive cubic interpolation.
The historical comparison continues to describe the original reference operator.

Older Python imports remain compatibility shims. `simulate(mode="rana")`
emits a deprecation warning and calls the reference renderer; new acquisition
code uses `run_scene` / `render_scene`. It is not a second public CLI mode.
