# Contributing to MaoMi

MaoMi Open Beta welcomes reproducible bug reports, new optical providers and
small, focused improvements. Start with the [architecture map](docs/ARCHITECTURE.md)
and [provider interface](docs/PROVIDERS.md).

For a bug report, include the release or commit, Python and OS versions, device
and dependency versions, the command and configuration, expected behavior and
the complete traceback. Prefer a small synthetic example that reproduces the
problem. Use [GitHub Issues](https://github.com/TomaKakuei/MaoMi/issues).

Before a pull request, run:

```sh
python -m pip install ".[mat,tissue,examples]"
python -m unittest discover -s tests -v
```

Add a focused numerical or integration check for changed behavior. Keep the
Standard MATLAB-compatible route, RANA core and host adapters separate, and
document units, array axes and boundary conditions. Preserve NAOMi citations
and notices. Do not include local experiment directories, environments, large
movies or source tissue files in a contribution.
