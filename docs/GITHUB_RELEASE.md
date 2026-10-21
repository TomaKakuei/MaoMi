# Public snapshot

MaoMi 0.8.0.dev0 connects source anatomy to standard/RANA acquisition through
one CPU/CUDA workflow. Only `--rana` enables the add-on. RANA has a separate
development core and native adapter; see [architecture](ARCHITECTURE.md) and
[commands](CLI.md). Author: Yueze Wang.

```sh
python -m pip install ".[mat,tissue,examples]"
python -m maomi run --config examples/scene.json --output artifacts/movie
python -m unittest discover -s tests -v
```

See [README](../README.md), [workflow guide](WORKFLOWS.md), [original datasets](DATASETS.md)
and [one-page overview](TECHNICAL_REPORT.pdf). Source checksums are in
`MANIFEST.sha256`. Local source volumes, experimental outputs, logs, backups
and development environments are excluded. Native CUDA needs the CUDA extra
and CUDA-enabled PyTorch; RayOptics needs the optics extra. CPU CI covers
Python 3.11/3.12 on Windows/Linux.
