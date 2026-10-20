# MaoMi 0.8.0.dev0 - source-to-movie workflows

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
