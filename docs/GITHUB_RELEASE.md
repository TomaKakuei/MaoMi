# Public snapshot

MaoMi 0.7.0.dev1 connects **RANA — Ray Aware Nonuniform Acquisition** as an
optional add-on to the existing acquisition pipeline.
See the [README](../README.md), [one-page overview](TECHNICAL_REPORT.pdf),
[integration guide](RANA_MODE.md) and [OSF datasets](DATASETS.md).

```sh
python -m pip install ".[examples]"
python examples/rana_movie.py --rana --output artifacts/movie
python -m unittest discover -s tests -v
```

The source includes code, tests, examples and public documentation, with
checksums in `MANIFEST.sha256`. Local experiments, arrays, logs and backups
are excluded. Native CUDA propagation uses the `cuda` extra and CUDA-enabled
PyTorch; RayOptics uses the `optics` extra. CPU CI covers Python 3.11/3.12.
