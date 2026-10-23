# MaoMi 0.8.0b1 — Open Beta 1 — 23 October 2026

- First tagged Open Beta: accelerated Standard optics, independent RANA core,
  external PSF providers, original MAT inputs and resumable paired movies.
- Includes time-resolved acquisition, a 512-pixel motion comparison and the
  shared English command/API/architecture guides.
- Existing `0.8.0.dev0` movie archives can resume in this beta; unknown versions
  remain rejected. The package distribution is still `maomi-optics-preview`,
  with the public command and import both named `maomi`.
- Source ZIP, wheel, overview, timing comparison and SHA-256 checksums are
  available in the GitHub prerelease. See [installation](GITHUB_RELEASE.md).

# Time-resolved acquisition - 22 October 2026

- Optional `acquisition.timing` uses explicit raster, rolling or global sample
  times through the same Standard/RANA movie commands and resume workflow.
- Activity and fluorescence motion enter the optical calculation before
  integration; complete local PSFs remain in use in RANA. Detector noise is
  sampled once after temporal integration and incoherent beam summation.
- Declared PSF dwell factors are converted to rates, avoiding double exposure.
- Stationary unchanged scenes reuse their optical image; temporal integration
  streams snapshots rather than keeping a temporal image cube in memory.
- Frame timestamps and timing conventions are stored with the paired data.
- Added a 512-pixel moving-cell comparison, separate field/ROI metrics, temporal
  refinement, analytic timing tests, and CPU/CUDA and resume checks.
- Existing acquisition stays the default. See the
  [time-resolved guide](TIME_RESOLVED_ACQUISITION.md).

# RANA provider and backend boundaries - 20 October 2026

- External optical providers implement `build_field()`; the same field can
  enter the core or the shared Scene workflow without modifying MaoMi.
- CPU/CUDA array operations live behind one internal backend contract. Optical
  physics, interpolation and bounded FFT scheduling remain separate.
- Commands, existing Python calls, numerical conventions and archive schemas
  are preserved. Archives store resolved fields; external solvers are not
  required for resume.
- Added a runnable [custom provider example](PROVIDERS.md). RANA's development
  direction emphasizes extensions built on NAOMi, with upstream credit retained.

# Unified commands and independent RANA core - 20 October 2026

- One `maomi` CLI: `run`, `render`, `resume`, `inspect`; only `--rana` enables RANA.
- Static and movie tasks share preparation, providers and directory conventions.
- RANA physics accepts declarative JSON as well as Python configuration.
- RANA lives in `maomi/rana/`; native coupling lives in its adapter. Existing
  module imports are shims. Numerical implementations are preserved.
- The old static selector is replaced by an explicitly named global RBF
  reference API. See [migration](CLI.md#migration-from-older-snapshots).
- The compact recipe explicitly selects a shared nonnegative tail policy to
  remove native resampling ringing before photon sampling. Raw optical
  descriptors and the standard default remain MATLAB-compatible.
- Existing 0.8.0.dev0 movie archives retain their resume version contract.

# MaoMi 0.8.0.dev0 - source-to-movie workflows

- Correct automatic Y-motion bounds so boundary shifts do not introduce an
  invalid final column before detector noise. Explicit supplied trajectories
  retain the existing row-scanner coordinate convention.

- Direct original NAOMi MAT v5/v7.3 import, lazy input arrays, physical ROI
  cropping and sparse-index remapping with preserved component IDs.
- Unified Scene/RunConfig execution on CPU/CUDA; complete procedural tissue
  presets, portable scene/optics archives and separate optical preparation.
- RANA compact batched convolution, bounded FFT caching and reusable sparse
  movie buffers; optional compiled ordered writes and chunked activity reads.
- RayOptics XYZ multiphoton fields and complex-pupil coupling to native
  scattering; a typed XYZ wave/intensity provider interface.
- Atomic paired-movie chunks, frame-indexed noise, verified resume and lazy
  MovieDataset access with anatomy/activity/motion/optics ground truth.
- Existing interfaces, upstream notices and the original MATLAB comparison
  retain their scope. [Complete guide](WORKFLOWS.md).

# MaoMi 0.7.0.dev1 — RANA acquisition add-on

- `rana=True` enables complete native local PSFs inside existing optics and
  acquisition entry points. Defaults remain MATLAB-compatible.
- Native scattering, pupils, excitation modes and multiple beams are retained.
  Movie scans share activity, nuclear labeling, tails, motion, binning, noise,
  analysis and streaming/export with the existing pipeline.
- Custom full-field/depth-bank providers and all four RANA physics extensions
  now feed the same movie scanner. `attach_rana` reuses computed optics.
- New [complete movie example](../examples/rana_movie.py), integration tests
  and [OSF download/usage guide](DATASETS.md).
- English README and four-section overview describe the optional add-on.
- The previous standalone custom-field API and its numerical comparison remain.
  Native MATLAB speed figures retain their original measurement scope.

Source ZIP and wheel contain runtime code, tests, examples and public docs.
Local experiments, source volumes, raw logs, private backups and development
environments are excluded.
