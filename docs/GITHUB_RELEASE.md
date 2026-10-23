# MaoMi Open Beta 1

**Version 0.8.0b1 · tag `v0.8.0-beta.1` · author Yueze Wang**

MaoMi connects source anatomy to accelerated optical simulation and paired
clean/noisy movies. Standard uses MATLAB-compatible optics; adding `--rana`
selects complete local PSFs through the same commands and file formats.

## Install a fixed release

Use Python 3.11 or 3.12 in a fresh environment. Download and extract
`MaoMi-0.8.0b1-source.zip` from the
[Open Beta release](https://github.com/TomaKakuei/MaoMi/releases/tag/v0.8.0-beta.1),
then open a terminal in its `MaoMi` directory. Alternatively:

```sh
git clone --branch v0.8.0-beta.1 --depth 1 https://github.com/TomaKakuei/MaoMi.git
cd MaoMi
```

Install and run a small CPU example:

```sh
python -m pip install ".[mat,tissue,examples]"
python -m maomi --version
python -m maomi run --demo --device cpu --output artifacts/standard
python -m maomi run --demo --device cpu --rana --output artifacts/rana
```

`maomi` and `python -m maomi` use the same parser. Each output is a new directory
containing the archived scene/optics, paired movie chunks and metadata. Use
`python -m maomi inspect artifacts/rana` to inspect it. For the procedural-tissue
example, replace `--demo` with `--config examples/scene.json`.

The wheel is also available for installation without a source checkout:

```sh
python -m pip install "./maomi_optics_preview-0.8.0b1-py3-none-any.whl[mat,tissue]"
```

The distribution name remains `maomi-optics-preview`; the command/import is
`maomi`. Examples and reports are in the source ZIP. Install the `optics` extra
for RayOptics; its compatible OpticalGlass range is pinned by the package.
Native CUDA propagation needs CUDA-enabled PyTorch and the `cuda` extra.
The [workflow guide](WORKFLOWS.md) distinguishes GPU optical work from CPU
anatomy, detector and file operations.

## Time-resolved movies and resume

```sh
python -m maomi run --demo --config examples/timed_acquisition.json --rana --output artifacts/timed --max-frames 3
python -m maomi resume artifacts/timed
```

The optional clock models when pixels collect light. Activity and specimen
motion enter the optical calculation before photon integration and noise.
Omitting `acquisition.timing` keeps the existing acquisition. See the
[timing guide and 512-pixel comparison](TIME_RESOLVED_ACQUISITION.md).

This beta resumes compatible `0.8.0.dev0` archives and its own `0.8.0b1` archives.
The saved scene, PSFs, device, motion and seeds are restored. Future archive
versions require an explicit compatibility decision.

## Evidence and scope

- The documented Standard CUDA optical benchmark is 49.1x faster than the
  matched 4225-site MATLAB CPU task, with 0.00667% full-volume image L2
  difference. Workload, hardware and timing boundaries are in the README.
- RANA exposes independent providers, complete spatial/depth PSFs and four
  composable physics extensions, with theoretically higher accuracy.
- Local verification includes CUDA, installed-package execution, resume,
  32 unchanged legacy arrays and 513-to-1025 temporal refinement at 0.1889%
  image L2 change in the published moving-cell example.
- GitHub Actions exercises Windows/Linux CPU on Python 3.11/3.12. CUDA testing
  is local; the public runners do not provide a GPU.

Time refinement measures numerical convergence. Optical parameters and photon
budgets remain explicit model inputs. Large tissue volumes, dense local PSF
banks and fine temporal sampling require proportionate memory and computation.

## Downloads, feedback and documentation

The GitHub prerelease contains the source ZIP, wheel, one-page overview, timing
comparison PDF and `SHA256SUMS.txt`. `MANIFEST.sha256` verifies individual source
files. Local tissue, experimental outputs, logs, backups and environments are
excluded from the release.

Report reproducible problems through
[GitHub Issues](https://github.com/TomaKakuei/MaoMi/issues/new/choose).
Include your release/commit, OS, Python, device, command, configuration and
traceback. [Contribution guidance](../CONTRIBUTING.md) describes the development
boundaries and local checks.

[Commands](CLI.md) · [Workflows](WORKFLOWS.md) · [Architecture](ARCHITECTURE.md)
· [Providers](PROVIDERS.md) · [Original datasets](DATASETS.md)
· [NAOMi credit](../README.md#citation-and-upstream-credit)
