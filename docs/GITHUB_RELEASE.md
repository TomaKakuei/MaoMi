# Release packaging

Repository: [TomaKakuei/MaoMi](https://github.com/TomaKakuei/MaoMi).
Development version: `0.5.0.dev0`.
Suggested prerelease tag: `v0.5.0.dev0-platform-20261017`.

Build an installable wheel from the repository root:

```shell
python -m pip wheel . --no-deps --wheel-dir dist
```

The wheel contains the Python runtime. The repository also includes examples,
self-contained tests, API documentation and the [one-page English report](TECHNICAL_REPORT.pdf).
Use [RELEASE_NOTES.md](RELEASE_NOTES.md) for a GitHub prerelease description.
Attach the wheel, a source archive and the report when creating a release.
Local experimental records are excluded from the public source.

GitHub Actions runs CPU tests and the generated spatial-PSF example on Windows
and Ubuntu. Check the repository's Actions tab for the result of each commit.
