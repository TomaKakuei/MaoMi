# Public snapshot

MaoMi 0.7.0.dev0 introduces **RANA — Ray Aware Nonuniform Acquisition**.
Use the [README](../README.md) and [one-page overview](TECHNICAL_REPORT.pdf).

The public source includes the implementation, tests, runnable examples,
equations, concise comparison results and the Rana pixel-cat asset. Source
checksums are listed in `MANIFEST.sha256`. Local experiments, arrays, logs,
backups and environment-specific runtime files are excluded.

```sh
python -m pip install ".[examples]"
python -m maomi --mode rana --physics-demo --no-status-window --output artifacts/rana
python -m unittest discover -s tests -v
```

Optional native CUDA propagation uses `.[cuda]` plus CUDA-enabled PyTorch.
RayOptics uses `.[optics]`; Python 3.11/3.12 are covered by CPU CI.
